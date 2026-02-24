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
    mask_binary[:n_mask] = False
    np.random.shuffle(mask_binary)

    return mask_binary


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

        self.is_ssl = is_ssl
        self.is_causal = is_causal

        self.dim = dim

        self.ctx_time = ctx_time
        self.bin_time = bin_time
        if not np.isclose(
            ctx_time - bin_time * np.round(ctx_time / bin_time), 0, atol=-1e6
        ):
            raise ValueError(
                f"ctx_time should be a multiple of bin_time (ctx_time:{ctx_time}, bin_time:{bin_time})"
            )

        self.bin_size = int(np.round(ctx_time / bin_time))
        self.max_bincount = max_bincount
        self.units_per_patch = units_per_patch

        if self.is_ssl:
            self.mask_ratio = mask_ratio

        # TODO Check NDT2 init_scale
        if units_per_patch > dim:
            raise ValueError(
                f"dim should be greater than units_per_patch (dim:{dim}, units_per_patch:{units_per_patch})"
            )
        if dim % units_per_patch != 0:
            raise ValueError(
                f"dim should be divisible by units_per_patch (dim:{dim}, units_per_patch:{units_per_patch})"
            )
        self.patch_emb = Embedding(
            max_bincount + 1, dim // units_per_patch, padding_idx=max_bincount
        )

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

        ### Encoder
        self.enc_time_emb = Embedding(self.bin_size, dim)
        self.enc_space_emb = Embedding(max_num_units // units_per_patch, dim)

        self.enc_dropout_in = nn.Dropout(dropout)
        self.enc_dropout_out = nn.Dropout(dropout)

        self.enc_heads = enc_heads
        self.enc_depth = enc_depth
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=enc_heads,
            dim_feedforward=int(dim * enc_ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, enc_depth)

        # Learnable token appended for masked-token reconstruction in SSL.
        # Learnable query token used to decode behavior at selected time bins.
        # 0: Ssl Mask token, 1: Behavior token
        self.query_emb = nn.Embedding(2, dim)

        ### Decoder
        self.dec_time_emb = Embedding(self.bin_size, dim)
        if self.is_ssl:
            # SSL decoder keeps spatial tokens; supervised path spatially pools latents first.
            self.dec_space_emb = Embedding(max_num_units // units_per_patch, dim)

        self.dec_dropout_in = nn.Dropout(dropout)
        self.dec_dropout_out = nn.Dropout(dropout)

        self.dec_heads = dec_heads
        self.dec_depth = dec_depth
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=dec_heads,
            dim_feedforward=int(dim * dec_ffn_mult),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm,
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, dec_depth)

        ### Readout
        if self.is_ssl:
            self.output_to_units = nn.Linear(dim, units_per_patch)

        else:
            self.task = "regression"
            self.readout_spec = readout_spec
            # Behavior readout.
            self.output_to_bhvr = nn.Linear(dim, readout_spec.dim)

    def tokenize(self, data: Data) -> Dict:
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

        # Time and space indices for flattened patches.
        time_idx = np.arange(n_bins, dtype=np.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=n_units_patches)

        space_idx = np.arange(n_units_patches, dtype=np.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=n_bins)

        ### Target tokens
        if self.is_ssl:
            target = pad(units_patched)

        else:
            bhvr = prepare_for_readout(data, self.readout_spec)[1]
            # TODO Hack to test while still hacing a problem with data
            # Will be removed soon
            if len(bhvr) != n_bins:
                bhvr = np.zeros((50, 2), dtype=bhvr.dtype)

            target = bhvr

        ### Context tokens
        session_idx, subject_idx, task_idx = [], [], []
        if self.tokenize_session:
            session_idx = self.session_emb.tokenizer(data.session.id)
        if self.tokenize_subject:
            subject_idx = self.subject_emb.tokenizer(data.subject.id)
        if self.tokenize_task:
            task_idx = self.task_emb.tokenizer(data.brainset.id)

        n_ctx_tokens = self.n_ctx_tokens

        if self.is_ssl:
            ssl_mask = get_ssl_mask(units_patched, self.mask_ratio)

            in_units_patched = units_patched[ssl_mask]
            in_time_idx = time_idx[ssl_mask]
            in_space_idx = space_idx[ssl_mask]
            in_mask = np.zeros(n_ctx_tokens + in_time_idx.shape[0], dtype=np.bool_)

            query_time_idx = time_idx[~ssl_mask]
            query_space_idx = space_idx[~ssl_mask]
            query_idx = np.zeros_like(query_time_idx)
            query_mask = np.zeros(query_time_idx.shape[0], dtype=np.bool_)

        else:
            in_units_patched = units_patched
            in_time_idx = time_idx
            in_space_idx = space_idx
            in_mask = np.zeros(n_ctx_tokens + time_idx.shape[0], dtype=np.bool_)

            if self.task == "classification":
                query_time_idx = np.array([self.bin_size - 1])

            else:
                query_time_idx = np.arange(self.bin_size)

            query_idx = np.ones_like(query_time_idx)
            query_mask = np.zeros(query_time_idx.shape[0], dtype=np.bool_)

            query_space_idx = np.zeros((1))

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

    def create_ctx_emb(self, session_idx, subject_idx, task_idx):
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

    def create_attn_mask(self, time_idx, n_heads):
        # (True = block attention)
        B, L = time_idx.size()
        n_ctx_tokens = self.n_ctx_tokens

        L = L + n_ctx_tokens

        if self.is_causal:
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

    def pool(self, latents, in_mask, in_time_idx):
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

        return latents

    def prepare_encoder_inputs(
        self, in_units_patched, in_time_idx, in_space_idx, in_mask, ctx_emb
    ):

        # Convert unit patches to encoder input tokens.
        inputs = self.patch_emb(in_units_patched)
        inputs = rearrange(inputs, "b l p p_dim -> b l (p p_dim)")
        inputs = self.enc_dropout_in(inputs)

        # Add temporal and spatial position embeddings.
        inputs = (
            inputs + self.enc_time_emb(in_time_idx) + self.enc_space_emb(in_space_idx)
        )

        # Prepend context tokens to the sequnce tokens.
        if self.n_ctx_tokens > 0:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        enc_padding_mask = ~in_mask

        ### Encoder attention mask
        enc_attn_mask = self.create_attn_mask(in_time_idx, self.enc_heads)

        return inputs, enc_padding_mask, enc_attn_mask

    def encoder_forward(self, inputs, enc_padding_mask, enc_attn_mask):
        # Encoder forward pass.
        enc_output = self.encoder(
            inputs, src_key_padding_mask=enc_padding_mask, mask=enc_attn_mask
        )

        # Remove prepended context tokens.
        latents = enc_output[:, self.n_ctx_tokens :]
        latents = self.enc_dropout_out(latents)

        return latents

    def prepare_decoder_inputs(
        self,
        latents,
        in_time_idx,
        in_space_idx,
        in_mask,
        query_idx,
        query_time_idx,
        query_space_idx,
        query_mask,
        ctx_emb,
    ):
        B = latents.size(0)
        dec_padding_mask = torch.concat([~in_mask, ~query_mask], dim=1)
        if self.is_ssl:
            dec_time_idx = torch.concat([in_time_idx, query_time_idx], dim=1)
            dec_space_idx = torch.concat([in_space_idx, query_space_idx], dim=1)

        else:
            pooled_time_idx = torch.arange(self.bin_size, device=query_time_idx.device)
            pooled_time_idx = repeat(pooled_time_idx, "t -> b t", b=B)
            dec_time_idx = torch.concat([pooled_time_idx, query_time_idx], dim=1)

        if not self.is_ssl:
            latents = self.pool(latents, in_mask[:, self.n_ctx_tokens :], in_time_idx)

        query_tokens = self.query_emb(query_idx)

        # Append mask tokens to encoder latents
        latents = torch.cat([latents, query_tokens], dim=1)

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

        return latents, dec_time_idx, dec_padding_mask

    def decoder_forward(self, latents, dec_time_idx, dec_padding_mask):
        ### Decoder attention mask
        dec_attn_mask = self.create_attn_mask(dec_time_idx, self.dec_heads)

        # Decoder forward pass.
        dec_output = self.decoder(
            latents, src_key_padding_mask=dec_padding_mask, mask=dec_attn_mask
        )

        # Remove prepended context tokens.
        output = dec_output[:, self.n_ctx_tokens :]
        output = self.dec_dropout_out(output)
        return output

    def forward(
        self,
        in_units_patched: torch.Tensor,
        in_time_idx: torch.Tensor,
        in_space_idx: torch.Tensor,
        in_mask: torch.Tensor,
        query_idx: torch.Tensor,
        query_time_idx: torch.Tensor,
        query_space_idx: torch.Tensor,
        query_mask: torch.Tensor,
        session_idx: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        task_idx: Optional[torch.Tensor] = None,
    ):
        # TODO update
        """
        Args:
            units_patched: (b, len_seq, patch, units_per_patch) tokenized unit-count patches.
            time_idx: (b, len_seq) token time indices.
            space_idx: (b, len_seq) token space indices.
            unpadding_mask: (b, len_seq) True for valid (non-padding) encoder tokens.
            session_idx: (b,) optional session token indices.
            subject_idx: (b,) optional subject token indices.
            task_idx: (b,) optional task token indices.

        Returns:
            Dict[str, Tensor]: Output dictionary containing model predictions.
        """
        ctx_emb = self.create_ctx_emb(session_idx, subject_idx, task_idx)

        inputs, enc_padding_mask, enc_attn_mask = self.prepare_encoder_inputs(
            in_units_patched, in_time_idx, in_space_idx, in_mask, ctx_emb
        )

        latents = self.encoder_forward(inputs, enc_padding_mask, enc_attn_mask)

        latents, dec_time_idx, dec_padding_mask = self.prepare_decoder_inputs(
            latents,
            in_time_idx,
            in_space_idx,
            in_mask,
            query_idx,
            query_time_idx,
            query_space_idx,
            query_mask,
            ctx_emb,
        )

        out = self.decoder_forward(latents, dec_time_idx, dec_padding_mask)

        if self.is_ssl:
            # Reconstruct per-unit rates for masked tokens.
            # output = output[mask]
            rates = self.output_to_units(output)

            return {"output": rates}

        else:
            # Predict behavior outputs.
            n_bhvr_tokens = query_idx.size(1)
            output = output[:, -n_bhvr_tokens:]
            bhvr = self.output_to_bhvr(output)

            return {"output": bhvr}
