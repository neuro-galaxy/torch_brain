import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from temporaldata import Data

from torch_brain.data import pad, track_mask
from torch_brain.nn import Embedding, InfiniteVocabEmbedding
from torch_brain.registry import ModalitySpec
from torch_brain.utils import prepare_for_readout
from torch_brain.utils.binning import bin_spikes

# Removed parameters kept here for reference.
# lag: bool = False,
# bhvr_lag_bins: int = 0,
# Context-token masking against the main token stream happens in attention masks.


def get_ssl_mask(x, mask_ratio):
    T = len(x)
    n_mask = int(mask_ratio * T)

    mask_binary = np.zeros(T, dtype=np.bool_)
    mask_binary[:n_mask] = True
    np.random.shuffle(mask_binary)

    return mask_binary, n_mask


def create_attn_mask(time_idx, n_ctx_tokens, n_heads, is_causal):
    # (True = block attention)
    B = time_idx.shape[0]
    L = time_idx.shape[1] + n_ctx_tokens

    if is_causal:
        # Per-sample mask (varies across the batch), which typically disables Flash attention.
        # Non-context tokens use causal masking over non-context tokens.
        attn_mask = torch.zeros((B, L, L), dtype=torch.bool, device=time_idx.device)
        attn_mask[:, :n_ctx_tokens, n_ctx_tokens:] = True
        attn_mask = time_idx[:, :, None] < time_idx[:, None, :]
        attn_mask[:, n_ctx_tokens:, n_ctx_tokens:] = attn_mask

        # b l l -> (b enc_heads) l l
        attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(-1, n_heads, -1, -1)
        attn_mask = attn_mask.reshape(-1, L, L)

    else:
        # Shared mask (same for all samples), which is compatible with Flash attention.
        attn_mask = torch.zeros((L, L), dtype=torch.bool, device=time_idx.device)
        attn_mask[:n_ctx_tokens, n_ctx_tokens:] = True

    return attn_mask


class NDT2(nn.Module):
    """NDT2 model from `Ye et al. 2023, Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity <https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1>`_."""

    def __init__(
        self,
        is_ssl: bool,
        dim: int,
        units_per_patch: int,
        max_bincount: int,
        max_num_units: int,
        bin_time: float,
        ctx_time: float,
        tokenize_session: bool,
        tokenize_subject: bool,
        tokenize_task: bool,
        enc_depth: int,
        enc_heads: int,
        enc_ffn_mult: float,
        dec_depth: int,
        dec_heads: int,
        dec_ffn_mult: float,
        dropout: float,
        activation: str = "gelu",
        pre_norm: bool = False,
        is_causal: bool = False,
        # SSL params
        mask_ratio: Optional[float] = None,
        # Supervised params
        readout_spec: Optional[ModalitySpec] = None,
    ):
        super().__init__()

        if is_ssl and mask_ratio is None:
            raise ValueError("mask_ratio must be provided when is_ssl=True")
        if not is_ssl and readout_spec is None:
            raise ValueError("readout_spec must be provided when is_ssl=False")

        if not np.isclose(
            ctx_time - bin_time * np.round(ctx_time / bin_time), 0, atol=-1e6
        ):
            raise ValueError(
                f"ctx_time should be a multiple of bin_time (ctx_time:{ctx_time}, bin_time:{bin_time})"
            )

        # TODO Check NDT2 init_scale
        if units_per_patch > dim:
            raise ValueError(
                f"dim should be greater than units_per_patch (dim:{dim}, units_per_patch:{units_per_patch})"
            )
        if dim % units_per_patch != 0:
            raise ValueError(
                f"dim should be divisible by units_per_patch (dim:{dim}, units_per_patch:{units_per_patch})"
            )

        self.bin_time = bin_time
        bin_size = int(np.round(ctx_time / bin_time))
        self.bin_size = bin_size

        self.is_ssl = is_ssl
        self.mask_ratio = mask_ratio
        self.units_per_patch = units_per_patch
        self.max_bincount = max_bincount

        self.ctx_embedder = ContextEmbedder(
            dim, tokenize_session, tokenize_subject, tokenize_task
        )
        n_ctx_tokens = self.ctx_embedder.n_ctx_tokens
        self.n_ctx_tokens = n_ctx_tokens

        self.encoder = NDT2Encoder(
            is_ssl,
            n_ctx_tokens,
            max_bincount,
            units_per_patch,
            bin_size,
            max_num_units,
            dim,
            enc_depth,
            enc_heads,
            enc_ffn_mult,
            dropout,
            activation,
            pre_norm,
            is_causal,
        )

        self.decoder = NDT2Decoder(
            is_ssl,
            n_ctx_tokens,
            units_per_patch,
            bin_size,
            max_num_units,
            dim,
            dec_depth,
            dec_heads,
            dec_ffn_mult,
            dropout,
            activation,
            pre_norm,
            is_causal,
        )

        readout_dim = readout_spec.dim if readout_spec is not None else None
        self.readout_spec = readout_spec
        self.head = NDT2TaskHead(is_ssl, dim, units_per_patch, readout_dim)

    def _patch_units(self, data: Data):
        ### Prepare sequnce tokens
        n_units = len(data.units.id)

        # `self.max_bincount` acts as the padding index.
        # TODO update with the new version of bin_spikes
        units_bincount = bin_spikes(
            data.spikes, n_units, self.bin_time, right=True, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        n_units_patches = int(np.ceil(n_units / self.units_per_patch))
        extra_units = n_units_patches * self.units_per_patch - n_units

        if extra_units > 0:
            # Pad the final patch when unit count is not divisible by patch size.
            # Example: 5 units with patch size 2 requires 1 padded unit.
            bottom_pad = ((0, extra_units), (0, 0))
            units_bincount = np.pad(
                units_bincount,
                bottom_pad,
                mode="constant",
                constant_values=self.max_bincount,
            )

        n_bins = units_bincount.shape[1]

        # Flatten patches in time-major order to match original NDT2 layout.
        units_patched = rearrange(
            units_bincount,
            "(n p) t -> (t n) p",
            t=n_bins,
            n=n_units_patches,
            p=self.units_per_patch,
        )

        extra_units_mask = None
        if self.is_ssl:
            # Track padded units so SSL reconstruction ignores synthetic entries.
            extra_units_mask = np.ones(
                (n_bins, n_units_patches, self.units_per_patch), dtype=np.bool_
            )

            if extra_units > 0:
                extra_units_mask[:, -1, -extra_units:] = False

            extra_units_mask = rearrange(
                extra_units_mask,
                "t n p -> (t n) p",
                t=n_bins,
                n=n_units_patches,
                p=self.units_per_patch,
            )

        return units_patched, extra_units_mask, n_bins, n_units_patches

    def _get_ctx_idx(self, data: Data):
        ### Context tokens
        session_idx, subject_idx, task_idx = [], [], []
        ctx_embedder = self.ctx_embedder
        if ctx_embedder.tokenize_session:
            session_idx = ctx_embedder.session_emb.tokenizer(data.session.id)
        if ctx_embedder.tokenize_subject:
            subject_idx = ctx_embedder.subject_emb.tokenizer(data.subject.id)
        if ctx_embedder.tokenize_task:
            task_idx = ctx_embedder.task_emb.tokenizer(data.brainset.id)

        return session_idx, subject_idx, task_idx

    def _get_seq_idx(self, units_patched, n_bins, n_units_patches):
        # Time and space indices for flattened patches.
        time_idx = np.arange(n_bins, dtype=np.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=n_units_patches)

        space_idx = np.arange(n_units_patches, dtype=np.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=n_bins)

        n_seq_tokens = n_bins * n_units_patches
        n_ctx_tokens = self.n_ctx_tokens

        if self.is_ssl:
            ssl_mask, n_mask_tokens = get_ssl_mask(units_patched, self.mask_ratio)

            in_units_patched = units_patched[ssl_mask]
            in_time_idx = time_idx[ssl_mask]
            in_space_idx = space_idx[ssl_mask]
            in_mask = np.zeros(
                n_ctx_tokens + n_seq_tokens - n_mask_tokens, dtype=np.bool_
            )

            query_time_idx = time_idx[~ssl_mask]
            query_space_idx = space_idx[~ssl_mask]
            query_idx = np.zeros_like(query_time_idx)
            query_mask = np.zeros(n_mask_tokens, dtype=np.bool_)

        else:
            in_units_patched = units_patched
            in_time_idx = time_idx
            in_space_idx = space_idx
            in_mask = np.zeros(n_ctx_tokens + n_seq_tokens, dtype=np.bool_)

            self.task = "regression"
            if self.task == "classification":
                query_time_idx = np.array([self.bin_size - 1])
                query_mask = np.zeros((1), dtype=np.bool_)

            else:
                query_time_idx = np.arange(self.bin_size)
                query_mask = np.zeros((self.bin_size), dtype=np.bool_)

            query_idx = np.ones_like(query_time_idx)

            query_space_idx = np.zeros((1))

        return (
            in_units_patched,
            in_time_idx,
            in_space_idx,
            in_mask,
            query_time_idx,
            query_space_idx,
            query_idx,
            query_mask,
        )

    def tokenize(self, data: Data) -> Dict:
        units_patched, extra_units_mask, n_bins, n_units_patches = self._patch_units(
            data
        )

        session_idx, subject_idx, task_idx = self._get_ctx_idx(data)

        (
            in_units_patched,
            in_time_idx,
            in_space_idx,
            in_mask,
            query_time_idx,
            query_space_idx,
            query_idx,
            query_mask,
        ) = self._get_seq_idx(units_patched, n_bins, n_units_patches)

        ### Target tokens
        if self.is_ssl:
            target = pad(units_patched)

        else:
            bhvr = prepare_for_readout(data, self.readout_spec)[1]
            # TODO Hack to test while still hacing a problem with data
            # Will be removed soon
            if len(bhvr) != n_bins:
                bhvr = np.zeros((50, 2), dtype=bhvr.dtype)

            target = pad(bhvr)

        data_dict = {
            "model_inputs": {
                # Input sequence
                "in_units_patched": pad(in_units_patched),
                "in_time_idx": pad(in_time_idx),
                "in_space_idx": pad(in_space_idx),
                "in_mask": track_mask(in_mask),
                # Query sequence
                "query_idx": pad(query_idx),
                "query_time_idx": pad(query_time_idx),
                "query_space_idx": pad(query_space_idx),
                "query_mask": track_mask(query_mask),
                # Context
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
            },
            "target": target,
        }

        if self.is_ssl:
            data_dict["extra_units_mask"] = pad(extra_units_mask)

        return data_dict

    def forward(
        self,
        in_units_patched,
        in_time_idx,
        in_space_idx,
        in_mask,
        query_idx,
        query_time_idx,
        query_space_idx,
        query_mask,
        session_idx=None,
        subject_idx=None,
        task_idx=None,
    ):

        # Context & Input Embeddings
        ctx_emb = self.ctx_embedder(session_idx, subject_idx, task_idx)

        # Encode
        latents = self.encoder(
            in_units_patched, in_time_idx, in_space_idx, in_mask, ctx_emb
        )

        # Decode
        latents, dec_time_idx, dec_space_idx, dec_padding_mask = (
            self.decoder.prepare_decoder(
                latents,
                in_time_idx,
                in_space_idx,
                in_mask,
                query_idx,
                query_time_idx,
                query_space_idx,
                query_mask,
            )
        )
        dec_out = self.decoder(
            latents, dec_time_idx, dec_space_idx, dec_padding_mask, ctx_emb
        )

        # Project to Task Output
        n_query_tokens = query_idx.size(1)
        return self.head(dec_out, n_query_tokens)


# ---------------------------------------------------------
# 1. Embedders
# ---------------------------------------------------------
class ContextEmbedder(nn.Module):
    """Handles optional session, subject, and task embeddings."""

    def __init__(self, dim, tokenize_session, tokenize_subject, tokenize_task):
        super().__init__()

        ### Context tokens
        self.tokenize_session = tokenize_session
        self.tokenize_subject = tokenize_subject
        self.tokenize_task = tokenize_task

        self.n_ctx_tokens = 0
        if self.tokenize_session:
            self.session_emb = InfiniteVocabEmbedding(dim)
            self.session_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.n_ctx_tokens += 1
        if self.tokenize_subject:
            self.subject_emb = InfiniteVocabEmbedding(dim)
            self.subject_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.n_ctx_tokens += 1
        if self.tokenize_task:  # More tied to dataset identity than semantic task.
            self.task_emb = InfiniteVocabEmbedding(dim)
            self.task_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.n_ctx_tokens += 1

    def forward(self, session_idx, subject_idx, task_idx):
        if self.n_ctx_tokens == 0:
            return None
        ctx_tokens = []

        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
        if self.tokenize_task:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

        ctx_emb = torch.stack(ctx_tokens, dim=1)

        return ctx_emb


# ---------------------------------------------------------
# 2. Encoder & Decoder Blocks
# ---------------------------------------------------------
class NDT2Encoder(nn.Module):
    """The main transformer encoder and pooling logic."""

    def __init__(
        self,
        is_ssl,
        n_ctx_tokens,
        # patch and spacio-temporal embeding params
        max_bincount,
        units_per_patch,
        bin_size,
        max_num_units,
        # transformer params
        dim,
        depth,
        heads,
        ffn_mult,
        dropout,
        activation,
        pre_norm,
        is_causal,
    ):
        super().__init__()
        self.is_ssl = is_ssl
        self.n_ctx_tokens = n_ctx_tokens

        self.patch_emb = Embedding(
            max_bincount + 1, dim // units_per_patch, padding_idx=max_bincount
        )

        self.bin_size = bin_size
        self.time_emb = Embedding(bin_size, dim)
        self.space_emb = Embedding(max_num_units // units_per_patch, dim)

        self.is_causal = is_causal
        self.heads = heads

        self.enc_dropout_in = nn.Dropout(dropout)
        self.enc_dropout_out = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)

    def forward(self, in_units_patched, in_time_idx, in_space_idx, in_mask, ctx_emb):
        inputs = self.patch_emb(in_units_patched)
        inputs = rearrange(inputs, "b l p p_dim -> b l (p p_dim)")
        inputs = self.enc_dropout_in(inputs)

        inputs = inputs + self.time_emb(in_time_idx) + self.space_emb(in_space_idx)

        if ctx_emb is not None:
            inputs = torch.concatenate([ctx_emb, inputs], dim=1)

        enc_padding_mask = ~in_mask
        attn_mask = create_attn_mask(
            in_time_idx, self.n_ctx_tokens, self.heads, self.is_causal
        )

        latents = self.encoder(
            inputs, src_key_padding_mask=enc_padding_mask, mask=attn_mask
        )
        latents = latents[:, self.n_ctx_tokens :]  # Drop prepended context
        latents = self.enc_dropout_out(latents)

        return latents


class NDT2Decoder(nn.Module):
    """The MAE-style decoder logic."""

    def __init__(
        self,
        is_ssl,
        n_ctx_tokens,
        # spacio-temporal embeding params
        units_per_patch,
        bin_size,
        max_num_units,
        # transformer params
        dim,
        depth,
        heads,
        ffn_mult,
        dropout,
        activation,
        pre_norm,
        is_causal,
    ):
        super().__init__()
        self.is_ssl = is_ssl
        self.n_ctx_tokens = n_ctx_tokens

        # Learnable token appended for masked-token reconstruction in SSL.
        # Learnable query token used to decode behavior at selected time bins.
        # 0: Ssl Mask token, 1: Behavior token
        self.query_emb = Embedding(2, dim)

        self.bin_size = bin_size
        self.dec_time_emb = Embedding(bin_size, dim)
        if is_ssl:
            # SSL decoder keeps spatial tokens; supervised path spatially pools latents first.
            self.dec_space_emb = Embedding(max_num_units // units_per_patch, dim)

        ### Decoder
        self.is_causal = is_causal
        self.heads = heads

        self.dec_dropout_in = nn.Dropout(dropout)
        self.dec_dropout_out = nn.Dropout(dropout)

        self.layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm,
        )
        self.decoder = nn.TransformerEncoder(self.layer, depth)

    def _pool(self, latents, in_mask, in_time_idx):
        # Remove context tokens
        in_mask = in_mask[:, self.n_ctx_tokens :]

        # Average-pool latents across spatial patches for each time bin.
        # +1 handles padding bucket.
        B, D = latents.size(0), latents.size(-1)
        pooled_latents_size = (B, self.bin_size + 1, D)
        pooled_latents = torch.zeros(
            pooled_latents_size, dtype=latents.dtype, device=latents.device
        )

        # Route padded entries to the extra bucket.
        index = torch.where(in_mask, in_time_idx, self.bin_size)
        index = repeat(index, "b l -> b l d", d=D)
        pooled_latents.scatter_reduce_(
            dim=1, index=index, src=latents, reduce="mean", include_self=False
        )
        latents = pooled_latents[:, :-1]  # Drop the padding bucket.

        # Pooling the spatial component of in_mask correspond to
        # vector of one indicating that time is present everywher as we only pad on the varying spcial dimension
        in_mask = torch.ones(
            (B, self.n_ctx_tokens + self.bin_size),
            dtype=torch.bool,
            device=latents.device,
        )

        return latents, in_mask

    def prepare_decoder(
        self,
        latents,
        in_time_idx,
        in_space_idx,
        in_mask,
        query_idx,
        query_time_idx,
        query_space_idx,
        query_mask,
    ):
        if not self.is_ssl:
            latents, in_mask = self._pool(latents, in_mask, in_time_idx)

        B = latents.size(0)
        dec_padding_mask = torch.concat([~in_mask, ~query_mask], dim=1)
        if self.is_ssl:
            dec_time_idx = torch.concat([in_time_idx, query_time_idx], dim=1)
            dec_space_idx = torch.concat([in_space_idx, query_space_idx], dim=1)

        else:
            pooled_time_idx = torch.arange(self.bin_size, device=query_time_idx.device)
            pooled_time_idx = repeat(pooled_time_idx, "t -> b t", b=B)
            dec_time_idx = torch.concat([pooled_time_idx, query_time_idx], dim=1)
            dec_space_idx = None

        query_tokens = self.query_emb(query_idx)

        # Append mask tokens to encoder latents
        latents = torch.cat([latents, query_tokens], dim=1)

        return latents, dec_time_idx, dec_space_idx, dec_padding_mask

    def forward(self, latents, dec_time_idx, dec_space_idx, dec_padding_mask, ctx_emb):
        latents = self.dec_dropout_in(latents)

        # Add decoder temporal embeddings.
        latents = latents + self.dec_time_emb(dec_time_idx)

        if self.is_ssl:
            # Add decoder spatial embeddings in SSL mode.
            latents = latents + self.dec_space_emb(dec_space_idx)

        # Prepend context tokens to decoder inputs.
        if self.n_ctx_tokens > 0:
            if not self.is_ssl:
                # Keep context embeddings fixed during supervised finetuning.
                ctx_emb = ctx_emb.detach()

            latents = torch.cat([ctx_emb, latents], dim=1)

        ### Decoder attention mask
        dec_attn_mask = create_attn_mask(
            dec_time_idx, self.n_ctx_tokens, self.heads, self.is_causal
        )

        # Decoder forward pass.
        dec_output = self.decoder(
            latents, src_key_padding_mask=dec_padding_mask, mask=dec_attn_mask
        )

        # Remove prepended context tokens.
        output = dec_output[:, self.n_ctx_tokens :]
        output = self.dec_dropout_out(output)
        return output


# ---------------------------------------------------------
# 3. Task Heads
# ---------------------------------------------------------
class NDT2TaskHead(nn.Module):
    """Final projections based on task."""

    def __init__(self, is_ssl, dim, units_per_patch=None, readout_dim=None):
        super().__init__()
        self.is_ssl = is_ssl
        if is_ssl:
            self.proj = nn.Linear(dim, units_per_patch)
        else:
            self.proj = nn.Linear(dim, readout_dim)

    def forward(self, x, n_query_tokens=0):
        if self.is_ssl:
            return {"output": self.proj(x)}
        else:
            # Supervised only reads from the query tokens appended at the end
            return {"output": self.proj(x[:, -n_query_tokens:])}
