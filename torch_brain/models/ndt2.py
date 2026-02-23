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


def get_batch_masking_indices(padding_mask, mask_ratio):
    """
    Args:
        padding_mask (Tensor): (B, L) boolean, True for VALID tokens, False for PAD.
        mask_ratio (float): Ratio of tokens to MASK.
    """
    device = padding_mask.device
    B, L = padding_mask.shape

    # Generate random noise
    noise = torch.rand(B, L, device=device)

    # Force padding tokens to the end of the sort order by making their noise negative
    # Tokens with high noise will be masked.
    noise = torch.where(padding_mask, noise, torch.tensor(-1.0, device=device))

    # Sort noise to get indices
    ids_shuffle = torch.argsort(noise, dim=1, descending=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Determine how many valid tokens to mask per sample
    num_valid = padding_mask.sum(dim=1)
    num_mask = (num_valid * mask_ratio).int()

    # Create the mask: 1 is MASK, 0 is KEEP
    # We create a mask of the top N indices where N is num_mask
    mask = torch.zeros((B, L), device=device)
    for i in range(B):
        mask[i, : num_mask[i]] = 1

    # Unshuffle the mask to align with original sequence order
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask.bool(), ids_restore


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

        if self.is_ssl:
            # Learnable token appended for masked-token reconstruction in SSL.
            self.masked_emb = nn.Parameter(torch.randn(dim))

        else:
            # Learnable query token used to decode behavior at selected time bins.
            self.bhvr_emb = nn.Parameter(torch.randn(dim))

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

        data_dict = {
            "model_inputs": {
                # Sequence
                "units_patched": pad(units_patched),
                "time_idx": pad(time_idx),
                "space_idx": pad(space_idx),
                "unpadding_mask": track_mask(time_idx),
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
        units_patched: torch.Tensor,
        time_idx: torch.Tensor,
        space_idx: torch.Tensor,
        unpadding_mask: torch.Tensor,
        session_idx: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        task_idx: Optional[torch.Tensor] = None,
    ) -> Dict:
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
        # Build optional context tokens.

        B, L = units_patched.size(0), units_patched.size(1)
        D = self.dim
        device = units_patched.device
        n_ctx_tokens = self.n_ctx_tokens

        # Convert unit patches to encoder input tokens.
        inputs = self.enc_dropout_in(units_patched)
        inputs = self.patch_emb(units_patched)
        inputs = rearrange(inputs, "b l p p_dim -> b l (p p_dim)")

        ctx_tokens = []
        if n_ctx_tokens > 0:
            if self.tokenize_session:
                ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
            if self.tokenize_subject:
                ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
            if self.tokenize_task:
                ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

            # Prepend context tokens to the sequnce tokens.
            ctx_emb = torch.stack(ctx_tokens, dim=1)
            inputs = torch.cat([ctx_emb, inputs], dim=1)

            ctx_pad_mask = torch.zeros(
                (B, n_ctx_tokens), dtype=torch.bool, device=device
            )

        else:
            ctx_pad_mask = torch.empty(dtype=torch.bool, device=device)

        if self.is_ssl:
            # Generate mask (True = MASKED, False = KEEP)
            mask, ids_restore = get_batch_masking_indices(
                unpadding_mask, self.mask_ratio
            )

            # Extract "Keep" tokens for Encoder
            # We use a trick: gather tokens based on sorted noise
            ids_keep = torch.argsort(
                torch.where(mask, -1.0, torch.rand_like(mask.float())),
                dim=1,
                descending=True,
            )

            # Handle variable length specifically
            num_keep = L - mask.sum(dim=1).max()

            # Filter inputs for encoder
            index = ids_keep[:, :num_keep]
            inputs = torch.gather(
                inputs, dim=1, index=repeat(index, "b l -> b l d", d=D)
            )

            ### Encoder spacio-temporal index
            enc_time_idx = torch.gather(time_idx, dim=1, index=index)
            enc_space_idx = torch.gather(space_idx, dim=1, index=index)

            ### Encoder padding mask (True = pad token)
            enc_padding_mask = torch.gather(~unpadding_mask, dim=1, index=index)
            enc_padding_mask = torch.cat([ctx_pad_mask, enc_padding_mask], dim=1)

            ### Decoder spacio-temporal index
            dec_time_idx = time_idx
            dec_space_idx = space_idx

            ### Decoder padding mask
            dec_padding_mask = torch.cat([ctx_pad_mask, ~unpadding_mask], dim=1)

            # Append learned masked tokens for decoder-side reconstruction.
            mask_tokens = repeat(self.masked_emb, "d -> b l d", b=B, l=L - num_keep)

        else:
            ### Encoder spacio-temporal index
            enc_time_idx = time_idx
            enc_space_idx = space_idx

            ### Encoder padding mask (True = pad token)
            enc_padding_mask = torch.cat([ctx_pad_mask, ~unpadding_mask], dim=1)

            ### Decoder temporal index (no spatial as temporal is pooled)
            pooled_time_idx = torch.arange(
                self.bin_size, dtype=time_idx.dtype, device=device
            )
            if self.task == "classification":
                bhvr_time_idx = torch.tensor(
                    [self.bin_size - 1], dtype=time_idx.dtype, device=device
                )
            else:
                bhvr_time_idx = pooled_time_idx.clone()

            bhvr_time_idx = repeat(bhvr_time_idx, "l -> b l")
            dec_time_idx = torch.cat([pooled_time_idx, bhvr_time_idx], dim=1)

            n_bhvr_tokens = bhvr_time_idx.size(1)
            bhvr_tokens = repeat(self.bhvr_emb, "h -> b l d", b=B, l=n_bhvr_tokens)

            bhvr_pad_mask = torch.zeros_like(bhvr_time_idx)
            dec_padding_mask = torch.cat(
                [ctx_pad_mask, ~unpadding_mask, bhvr_pad_mask], dim=1
            )

        # Add temporal and spatial position embeddings.
        inputs = (
            inputs + self.enc_time_emb(enc_time_idx) + self.enc_space_emb(enc_space_idx)
        )

        ### Encoder attention mask (True = block attention)
        # Context tokens cannot attend to non-context tokens.

        L = inputs.size(1)
        if self.is_causal:
            # Per-sample mask (varies across the batch), which typically disables Flash attention.
            # Non-context tokens use causal masking over non-context tokens.
            enc_attn_mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)
            enc_attn_mask[:, :n_ctx_tokens, n_ctx_tokens:] = True
            enc_causal_mask = enc_time_idx[:, :, None] < enc_time_idx[:, None, :]
            enc_attn_mask[:, n_ctx_tokens:, n_ctx_tokens:] = enc_causal_mask

            # b l l -> (b enc_heads) l l
            enc_attn_mask = enc_attn_mask.unsqueeze(1)
            enc_attn_mask = enc_attn_mask.expand(-1, self.enc_heads, -1, -1)
            enc_attn_mask = enc_attn_mask.reshape(-1, L, L)

        else:
            # Shared mask (same for all samples), which is compatible with Flash attention.
            L = inputs.size(1)
            enc_attn_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
            enc_attn_mask[:n_ctx_tokens, n_ctx_tokens:] = True

        # Encoder forward pass.
        enc_output = self.encoder(
            inputs, mask=enc_attn_mask, src_key_padding_mask=enc_padding_mask
        )

        # Remove prepended context tokens.
        latents = enc_output[:, n_ctx_tokens:]
        latents = self.enc_dropout_out(latents)

        # Append mask tokens to encoder latents
        if self.is_ssl:
            latents = torch.cat([latents, mask_tokens], dim=1)

            # Unshuffle to original positions
            latents = torch.gather(
                latents, dim=1, index=repeat(ids_restore, "b l -> b l d", d=D)
            )

        else:
            # Average-pool latents across spatial patches for each time bin.
            # +1 handles padding bucket.
            pooled_latents_size = (B, self.bin_size + 1, D)
            pooled_latents = torch.zeros(
                pooled_latents_size, dtype=latents.dtype, device=device
            )

            # Route padded entries to the extra bucket.
            index = torch.where(unpadding_mask, enc_time_idx, self.bin_size)
            index = repeat(index, "b l -> b l d", d=D)
            pooled_latents.scatter_reduce_(
                dim=1, index=index, src=latents, reduce="mean", include_self=False
            )
            latents = pooled_latents[:, :-1]  # Drop the padding bucket.

            latents = torch.cat([latents, bhvr_tokens], dim=1)

        latents = self.dec_dropout_in(latents)
        # Add decoder temporal embeddings.
        latents = latents + self.dec_time_emb(dec_time_idx)

        if self.is_ssl:
            # Add decoder spatial embeddings in SSL mode.
            latents = latents + self.dec_space_emb(dec_space_idx)

        # Prepend context tokens to decoder inputs.
        if n_ctx_tokens > 0:
            if not self.is_ssl:
                # Keep context embeddings fixed during supervised finetuning.
                ctx_emb = ctx_emb.detach()

            latents = torch.cat([ctx_emb, latents], dim=1)

        ### Decoder attention mask
        # Context tokens cannot attend to non-context tokens.

        L = latents.size(1)
        if self.is_causal:
            # Per-sample mask (varies across the batch), which typically disables Flash attention.
            # Non-context tokens use causal masking over non-context tokens.
            dec_attn_mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)
            dec_attn_mask[:, :n_ctx_tokens, n_ctx_tokens:] = True
            dec_causal_mask = dec_time_idx[:, :, None] < dec_time_idx[:, None, :]
            dec_attn_mask[:, n_ctx_tokens:, n_ctx_tokens:] = dec_causal_mask

            # b l l -> (b dec_heads) l l
            dec_attn_mask = dec_attn_mask.unsqueeze(1)
            dec_attn_mask = dec_attn_mask.expand(-1, self.dec_heads, -1, -1)
            dec_attn_mask = dec_attn_mask.reshape(-1, L, L)

        else:
            # Shared mask (same for all samples), which is compatible with Flash attention.
            dec_attn_mask = torch.zeros((L, L), dtype=torch.bool, device=device)
            dec_attn_mask[:n_ctx_tokens, n_ctx_tokens:] = True

        # Decoder forward pass.
        dec_output = self.decoder(
            latents, mask=dec_attn_mask, src_key_padding_mask=dec_padding_mask
        )

        # Remove prepended context tokens.
        output = dec_output[:, n_ctx_tokens:]
        output = self.dec_dropout_out(output)

        if self.is_ssl:
            # Reconstruct per-unit rates for masked tokens.
            output = output[mask]
            rates = self.output_to_units(output)

            return {"output": rates}

        else:
            # Predict behavior outputs.
            output = output[:, -n_bhvr_tokens:]
            bhvr = self.output_to_bhvr(output)

            return {"output": bhvr}
