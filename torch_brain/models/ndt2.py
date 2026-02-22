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


def get_masking_indices(input_tensor, mask_ratio):
    """
    Generates random indices for sample-level masking based on input shape.

    Args:
        input_tensor (Tensor): The input data, where dimension 0 is the sequence length.
        mask_ratio (float): The fraction of tokens to mask (e.g., 0.75).

    Returns:
        idx_keep (Tensor): 1D tensor of indices fed to the encoder.
        idx_mask (Tensor): 1D tensor of indices held out for the decoder.
    """
    # Number of tokens available for masking.
    total_bins = input_tensor.shape[0]

    # Number of visible tokens kept for the encoder.
    encoder_frac = int((1 - mask_ratio) * total_bins)

    # Random permutation used to split keep vs. mask tokens.
    # TODO be careful with seed set up
    shuffle = np.random.permutation(total_bins)

    # Partition into encoder-visible and decoder-reconstructed indices.
    idx_keep = shuffle[:encoder_frac]
    idx_mask = shuffle[encoder_frac:]

    return idx_keep, idx_mask


class NDT2(nn.Module):
    """NDT2 model from `Ye et al. 2023, Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity <https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1>`_."""

    def __init__(
        self,
        is_ssl: bool,
        dim: int,
        units_per_patch: int,
        max_bincount: int,
        max_time_patches: int,
        max_space_patches: int,
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
        self.is_ssl = is_ssl
        self.is_causal = is_causal

        self.ctx_time = ctx_time
        self.bin_time = bin_time
        if not np.isclose(
            ctx_time - bin_time * np.round(ctx_time / bin_time), 0, atol=1e6
        ):
            raise ValueError(
                f"ctx_time should be a multiple of bin_time (ctx_time:{ctx_time}, bin_time:{bin_time})"
            )

        self.bin_size = int(np.round(ctx_time / bin_time))
        self.max_bincount = max_bincount
        self.units_per_patch = units_per_patch
        self.max_time_patches = max_time_patches
        self.max_space_patches = max_space_patches

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
        self.enc_time_emb = Embedding(max_time_patches, dim)
        self.enc_space_emb = Embedding(max_space_patches, dim)

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
        self.dec_time_emb = Embedding(max_time_patches, dim)
        if self.is_ssl:
            # SSL decoder keeps spatial tokens; supervised path spatially pools latents first.
            self.dec_space_emb = Embedding(max_space_patches, dim)

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
        ### Context tokens
        session_idx, subject_idx, task_idx = [], [], []
        if self.tokenize_session:
            session_idx = self.session_emb.tokenizer(data.session.id)
        if self.tokenize_subject:
            subject_idx = self.subject_emb.tokenizer(data.subject.id)
        if self.tokenize_task:
            task_idx = self.task_emb.tokenizer(data.brainset.id)

        ### Prepare encoder_tokens
        n_units = len(data.units.id)

        # `self.max_bincount` acts as the padding index.
        units_bincount = bin_spikes(
            data.spikes, n_units, self.bin_time, right=True, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        n_units_patches = int(np.ceil(n_units / self.units_per_patch))
        extra_units = n_units_patches * self.units_per_patch - n_units

        if extra_units > 0:
            # Pad the final patch when unit count is not divisible by patch size.
            # Example: 3 units with patch size 2 requires 1 padded unit.
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

        if self.is_ssl:

            # ViT-style split: visible tokens to encoder, masked tokens to decoder.
            idx_keep, idx_mask = get_masking_indices(units_patched, self.mask_ratio)

            ### Encoder tokens
            enc_units_patched = units_patched[idx_keep]
            enc_time_idx = time_idx[idx_keep]
            enc_space_idx = space_idx[idx_keep]

            ### Decoder tokens
            dec_time_idx = time_idx[idx_mask]
            dec_space_idx = space_idx[idx_mask]

            ### Target tokens
            target = pad(units_patched[idx_mask])
            extra_units_mask = extra_units_mask[idx_mask]

        else:
            ### Encoder tokens
            enc_time_idx = time_idx
            enc_space_idx = space_idx
            enc_units_patched = units_patched

            ### Decoder tokens
            pooled_time_idx = np.arange(n_bins, dtype=np.int32)
            if self.task == "classification":
                bhvr_time_idx = np.array([n_bins - 1], dtype=np.int32)
            else:
                bhvr_time_idx = np.arange(n_bins, dtype=np.int32)

            dec_time_idx = np.concatenate([pooled_time_idx, bhvr_time_idx], axis=0)

            ### Target tokens
            bhvr = prepare_for_readout(data, self.readout_spec)[1]
            # TODO Hack to test while still hacing a problem with data
            # Will be removed soon
            if len(bhvr) != n_bins:
                bhvr = np.zeros((50, 2), dtype=bhvr.dtype)

            target = bhvr

        data_dict = {
            "model_inputs": {
                # context
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder
                "enc_units_patched": pad(enc_units_patched),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_unpadding_mask": track_mask(enc_time_idx),
                # decoder
                "dec_time_idx": pad(dec_time_idx),
                "dec_unpadding_mask": track_mask(dec_time_idx),
            },
            "target": target,
        }

        if self.is_ssl:
            data_dict["model_inputs"]["dec_space_idx"] = pad(dec_space_idx)
            data_dict["extra_units_mask"] = pad(extra_units_mask)

        return data_dict

    def forward(
        self,
        enc_units_patched: torch.Tensor,
        enc_time_idx: torch.Tensor,
        enc_space_idx: torch.Tensor,
        enc_unpadding_mask: torch.Tensor,
        dec_time_idx: torch.Tensor,
        dec_unpadding_mask: torch.Tensor,
        session_idx: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        task_idx: Optional[torch.Tensor] = None,
        dec_space_idx: Optional[torch.Tensor] = None,  # Only for SSL
    ) -> Dict:
        # TODO update
        """
        Args:
            enc_units_patched: (b, max_enc_l, units_per_patch) tokenized unit-count patches.
            enc_time_idx: (b, max_enc_l) encoder token time indices.
            enc_space_idx: (b, max_enc_l) encoder token space indices.
            enc_unpadding_mask: (b, max_enc_l) True for valid (non-padding) encoder tokens.
            dec_time_idx: (b, max_dec_l) decoder token time indices.
            dec_unpadding_mask: (b, max_dec_l) True for valid (non-padding) decoder tokens.
            session_idx: (b,) optional session token indices.
            subject_idx: (b,) optional subject token indices.
            task_idx: (b,) optional task token indices.
            dec_space_idx: (b, max_dec_l) decoder space indices (SSL only).

        Returns:
            Dict[str, Tensor]: Output dictionary containing model predictions.
        """
        # Build optional context tokens.
        ctx_tokens = []
        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
        if self.tokenize_task:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

        if self.n_ctx_tokens > 0:
            ctx_emb = torch.stack(ctx_tokens, dim=1)

        # Convert unit patches to encoder input tokens.
        inputs = self.patch_emb(enc_units_patched)
        inputs = rearrange(inputs, "... p p_dim -> ... (p p_dim)")
        inputs = self.enc_dropout_in(inputs)

        # Add temporal and spatial position embeddings.
        inputs = (
            inputs + self.enc_time_emb(enc_time_idx) + self.enc_space_emb(enc_space_idx)
        )

        # Prepend context tokens.
        if self.n_ctx_tokens > 0:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        ### Encoder attention mask (True = block attention)
        # Context tokens cannot attend to non-context tokens.

        if self.is_causal:
            # Per-sample mask (varies across the batch), which typically disables Flash attention.
            # Non-context tokens use causal masking over non-context tokens.
            b, n_tokens = inputs.size(0), inputs.size(1)
            enc_attn_mask = torch.zeros(
                (b, n_tokens, n_tokens), dtype=torch.bool, device=inputs.device
            )
            enc_attn_mask[:, : self.n_ctx_tokens, self.n_ctx_tokens :] = True
            enc_causal_mask = enc_time_idx[:, :, None] < enc_time_idx[:, None, :]
            enc_attn_mask[:, self.n_ctx_tokens :, self.n_ctx_tokens :] = enc_causal_mask

            enc_attn_mask = repeat(
                enc_attn_mask,
                "b n_1 n_2 -> (b enc_heads) n_1 n_2",
                enc_heads=self.enc_heads,
            )

        else:
            # Shared mask (same for all samples), which is compatible with Flash attention.
            b, n_tokens = inputs.size(0), inputs.size(1)
            enc_attn_mask = torch.zeros(
                (n_tokens, n_tokens), dtype=torch.bool, device=inputs.device
            )
            enc_attn_mask[: self.n_ctx_tokens, self.n_ctx_tokens :] = True

        ### Encoder padding mask (True = pad token)
        enc_padding_mask = ~enc_unpadding_mask
        if self.n_ctx_tokens > 0:
            ctx_pad_mask = torch.zeros(
                (b, self.n_ctx_tokens), dtype=torch.bool, device=inputs.device
            )
            enc_padding_mask = torch.cat([ctx_pad_mask, enc_padding_mask], dim=1)

        # Encoder forward pass.
        enc_output = self.encoder(
            inputs, mask=enc_attn_mask, src_key_padding_mask=enc_padding_mask
        )

        # Remove prepended context tokens.
        latents = enc_output[:, self.n_ctx_tokens :]
        latents = self.enc_dropout_out(latents)

        if self.is_ssl:
            # Append learned masked tokens for decoder-side reconstruction.
            b, n_masked_tokens = dec_time_idx.size(0), dec_time_idx.size(1)
            query_tokens = repeat(self.masked_emb, "h -> b n h", b=b, n=n_masked_tokens)

            # Padding is applied by the collater, so concatenate encoder/decoder indices here.
            dec_time_idx = torch.cat([enc_time_idx, dec_time_idx], dim=1)
            dec_space_idx = torch.cat([enc_space_idx, dec_space_idx], dim=1)

            dec_padding_mask = torch.cat([enc_padding_mask, ~dec_unpadding_mask], dim=1)

        else:
            # Average-pool latents across spatial patches for each time bin.
            b, _, h = latents.size()

            # +1 handles padding bucket.
            pooled_latents_size = (b, self.bin_size + 1, h)
            pooled_latents = torch.zeros(
                pooled_latents_size, device=latents.device, dtype=latents.dtype
            )

            # Route padded entries to the extra bucket.
            index = torch.where(enc_unpadding_mask, enc_time_idx, self.bin_size)
            index = repeat(index, "b enc_l -> b enc_l h", h=h)
            pooled_latents.scatter_reduce_(
                dim=1, index=index, src=latents, reduce="mean", include_self=False
            )
            latents = pooled_latents[:, :-1]  # Drop the padding bucket.

            n_bhvr_tokens = dec_time_idx.size(1) - self.bin_size
            query_tokens = repeat(self.bhvr_emb, "h -> b n h", b=b, n=n_bhvr_tokens)

            dec_padding_mask = ~dec_unpadding_mask
            if self.n_ctx_tokens > 0:
                ctx_pad_mask = torch.zeros(
                    (b, self.n_ctx_tokens), dtype=torch.bool, device=inputs.device
                )
                dec_padding_mask = torch.cat([ctx_pad_mask, dec_padding_mask], dim=1)

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

        ### Decoder attention mask
        # Context tokens cannot attend to non-context tokens.

        if self.is_causal:
            # Per-sample mask (varies across the batch), which typically disables Flash attention.
            # Non-context tokens use causal masking over non-context tokens.
            b, n_tokens = latents.size(0), latents.size(1)
            dec_attn_mask = torch.zeros(
                (b, n_tokens, n_tokens), dtype=torch.bool, device=inputs.device
            )
            dec_attn_mask[:, : self.n_ctx_tokens, self.n_ctx_tokens :] = True
            dec_causal_mask = dec_time_idx[:, :, None] < dec_time_idx[:, None, :]
            dec_attn_mask[:, self.n_ctx_tokens :, self.n_ctx_tokens :] = dec_causal_mask

            dec_attn_mask = repeat(
                dec_attn_mask,
                "b n_1 n_2 -> (b dec_heads) n_1 n_2",
                dec_heads=self.dec_heads,
            )

        else:
            # Shared mask (same for all samples), which is compatible with Flash attention.
            n_tokens = latents.size(1)
            dec_attn_mask = torch.zeros(
                (n_tokens, n_tokens), dtype=torch.bool, device=inputs.device
            )
            dec_attn_mask[: self.n_ctx_tokens, self.n_ctx_tokens :] = True

        # Decoder forward pass.
        dec_output = self.decoder(
            latents, mask=dec_attn_mask, src_key_padding_mask=dec_padding_mask
        )

        # Remove prepended context tokens.
        output = dec_output[:, self.n_ctx_tokens :]
        output = self.dec_dropout_out(output)

        if self.is_ssl:
            # Reconstruct per-unit rates for masked tokens.
            output = output[:, -n_masked_tokens:]
            rates = self.output_to_units(output)

            return {"output": rates}

        else:
            # Predict behavior outputs.
            output = output[:, -n_bhvr_tokens:]
            bhvr = self.output_to_bhvr(output)

            return {"output": bhvr}
