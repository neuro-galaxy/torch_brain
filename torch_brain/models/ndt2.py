import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from temporaldata import Data
from torchtyping import TensorType

from torch_brain.data import pad, pad2d, track_mask
from torch_brain.nn import Embedding, InfiniteVocabEmbedding
from torch_brain.registry import ModalitySpec
from torch_brain.utils import prepare_for_readout
from torch_brain.utils.binning import bin_spikes

# note we removed this
# lag: bool = False,
# bhvr_lag_bins: int = 0,


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
    # Dynamically grab the total number of bins from the first dimension
    total_bins = input_tensor.shape[0]

    # Calculate how many tokens go to the encoder
    encoder_frac = int((1 - mask_ratio) * total_bins)

    # Generate a random permutation of indices
    # TODO check if can use umpy with the dataloader seed
    shuffle = torch.randperm(total_bins)

    # Split the indices
    idx_keep = shuffle[:encoder_frac]
    idx_mask = shuffle[encoder_frac:]

    return idx_keep, idx_mask


class NDT2(nn.Module):
    """NDT2 model from `Ye et al. 2023, Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity <https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1>`_."""

    def __init__(
        self,
        dim,
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
        # SSL params
        mask_ratio: Optional[float] = None,
        # Supervised params
        readout_spec: Optional[ModalitySpec] = None,
    ):
        super().__init__()
        self.ctx_time = ctx_time
        self.bin_time = bin_time
        float_modulo_test = lambda x, y, eps=1e-6: np.abs(x - y * np.round(x / y)) < eps
        if not float_modulo_test(ctx_time, bin_time):
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
        if self.tokenize_task:  # more about dataset than task
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
            # Between encoder and decoder we used masked token to let them be reconstructed
            self.masked_emb = nn.Parameter(torch.randn(dim))

        else:
            # Between encoder and decoder we used query token to decode the behavior at a given time
            self.bhvr_emb = nn.Parameter(torch.randn(dim))

        ### Decoder
        self.dec_time_emb = Embedding(max_time_patches, dim)
        if self.is_ssl:
            # No need supervised regime as decoder tokens are spacially avg. pooled
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
            # Bhvr readout
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

        # self.max_bincount is used as the padding input
        units_bincount = bin_spikes(
            data.spikes, n_units, self.bin_time, right=True, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        n_units_patches = int(np.ceil(n_units / self.units_per_patch))
        extra_units = n_units_patches * self.units_per_patch - n_units

        if extra_units > 0:
            bottom_pad = ((0, extra_units), (0, 0))
            units_bincount = np.pad(
                units_bincount,
                bottom_pad,
                mode="constant",
                constant_values=self.max_bincount,
            )

        n_bins = units_bincount.shape[1]

        # flattened patches, here major hack to have time before space, as in o.g. NDT2 (n_units, time_length)
        units_patched = rearrange(
            units_bincount,
            "(n p) t -> (t n) p",
            t=n_bins,
            n=n_units_patches,
            p=self.units_per_patch,
        )

        if self.is_ssl:
            # last patches may have fewer units
            # ex: [unit_1, unit_2, unit_3] with patch size of 2 need 1 extra unit
            extra_units_mask = np.ones(
                (n_bins, n_units_patches, self.units_per_patch), dtype=np.bool_
            )
            extra_units_mask[:, -1, -extra_units:] = False
            extra_units_mask = rearrange(
                extra_units_mask,
                "t n p -> (t n) p",
                t=n_bins,
                n=n_units_patches,
                p=self.units_per_patch,
            )

        # time and space indices for flattened patches
        time_idx = np.arange(n_bins, dtype=np.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=n_units_patches)
        space_idx = np.arange(n_units_patches, dtype=np.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=n_bins)

        if self.is_ssl:

            # ViT style: First part of the tokens will be fed to the encoder
            # the last ones will be masked and reconstructed by the decoder
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

            ### Decoder tokens
            pooled_time_idx = np.arange(n_bins, dtype=np.int32)
            if self.task == "classification":
                bhvr_time_idx = np.array([n_bins - 1], dtype=np.int32)
            else:
                bhvr_time_idx = np.arange(n_bins, dtype=np.int32)

            dec_time_idx = np.concatenate([pooled_time_idx, bhvr_time_idx], axis=0)

            ### Target tokens
            bhvr = prepare_for_readout(data, self.readout_spec)[1]
            target = bhvr

        data_dict = {
            "model_inputs": {
                # context
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder
                "units_patched": pad(enc_units_patched),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_time_pad_idx": (
                    track_mask(enc_time_idx) if not self.is_ssl else []
                ),
                # decoder
                "dec_time_idx": pad(dec_time_idx) if self.is_ssl else dec_time_idx,
                "dec_space_idx": pad(dec_space_idx) if self.is_ssl else [],
            },
            "target": target,
            "extra_units_mask": pad(extra_units_mask) if self.is_ssl else [],
        }
        return data_dict

    def forward(
        self,
        units_patched: torch.Tensor,
        enc_time_idx: torch.Tensor,
        enc_space_idx: torch.Tensor,
        dec_time_idx: torch.Tensor,
        session_idx: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        task_idx: Optional[torch.Tensor] = None,
        dec_space_idx: Optional[torch.Tensor] = None,  # Only for SSL
        enc_time_pad_idx: Optional[torch.Tensor] = None,  # Only for Superv
    ) -> Dict:
        """
        units_patched: (b max_enc_l units_per_patch)
        enc_time_idx: (b max_enc_l)
        enc_space_idx: (b max_enc_l)
        dec_time_idx: (b max_dec_l)
        session_idx: (b)
        subject_idx: (b)
        task_idx: (b)
        dec_space_idx: (b max_dec_l)
        enc_time_pad_idx: (b max_enc_l)
        Prepare context tokens
        """
        ctx_tokens = []
        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
        if self.tokenize_task:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

        if self.n_ctx_tokens > 0:
            ctx_emb = torch.stack(ctx_tokens, dim=1)

        # From unit patches to input tokens
        inputs = self.patch_emb(units_patched)
        inputs = rearrange(inputs, "... p p_dim -> ... (p p_dim)")
        inputs = self.enc_dropout_in(inputs)

        # add the spacio-tempo embedings
        inputs = (
            inputs + self.enc_time_emb(enc_time_idx) + self.enc_space_emb(enc_space_idx)
        )

        # append context tokens at the start of the sequence
        if self.n_ctx_tokens > 0:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        ### Encoder attention mask
        # Context token can attend to non context token
        # Non context token cannot attend to context token
        # Non context token should follow a causal mask
        # No token can attend pad tokens as their value are 0
        # Note the stictly inferior plays an important role both for attention
        # TODO Check
        n_tokens = inputs.size(1)
        enc_attn_mask = torch.zeros(
            (n_tokens, n_tokens), dtpye=torch.bool, device=self.device
        )
        enc_attn_mask[:, : self.n_ctx_tokens] = True
        enc_attn_mask[self.n_ctx_tokens :, self.n_ctx_tokens :] = (
            enc_time_idx[:, None] < enc_time_idx[None, :]
        )

        # Encoder forward pass
        attn_mask = repeat(
            enc_attn_mask,
            "b n_1 n_2 -> (b enc_heads) n_1 n_2",
            enc_heads=self.enc_heads,
        )
        enc_output = self.encoder(inputs, mask=attn_mask)

        # remove context tokens at the start of the sequence
        latents = enc_output[:, self.n_ctx_tokens :]
        latents = self.enc_dropout_out(latents)

        if self.is_ssl:
            # Note that enc_space/time_idx dec_time/space_idx are padded in the collater
            # Thus, we need to concatenate them here and not in the tokenizer
            dec_time_idx = torch.cat([enc_time_idx, dec_time_idx], dim=1)
            dec_space_idx = torch.cat([enc_space_idx, dec_space_idx], dim=1)

            # Append the masked tokens to the encoder output (i.e. latents)
            b, n_masked_tokens = dec_time_idx.size(0), dec_time_idx.size(1)
            query_tokens = repeat(self.masked_emb, "h -> b n h", b=b, n=n_masked_tokens)

        else:
            # Spatial pooling of the latents
            b, _, h = latents.size()
            pooled_latents_size = (b, self.bin_size + 1, h)  # +1 to handdle padding
            pooled_latents = torch.zeros(
                pooled_latents_size, device=latents.device, dtype=latents.dtype
            )

            # Padding handdling
            # TODO check this
            index = torch.where(enc_time_pad_idx, enc_time_idx, self.bin_size)
            index = repeat(index, "b enc_l -> h enc_l h", h=h)
            pooled_latents.scatter_reduce_(
                dim=1, index=index, src=latents, reduce="mean", include_self=False
            )
            latents = pooled_latents[:, :-1]  # remove padding

            n_bhvr_tokens = dec_time_idx.size(1) - self.bin_size
            query_tokens = repeat(self.bhvr_emb, "h -> b n h", b=b, n=n_bhvr_tokens)

        latents = torch.cat([latents, query_tokens], dim=1)

        # Decoder forward pass
        latents = self.dec_dropout_in(latents)
        # add the temporal embedings
        latents = latents + self.dec_time_emb(dec_time_idx)

        if self.is_ssl:
            # add the spatial embedings
            latents = latents + self.dec_space_emb(dec_space_idx)

        # append context tokens at the start of the sequence
        if self.n_ctx_tokens > 0:
            if self.is_ssl:
                # detach the context tokens because don't want to uncalibrate the ctx_emb from the SSL pretraining
                ctx_emb = ctx_emb.detach()

            latents = torch.cat([ctx_emb, latents], dim=1)

        ### Decoder attention mask
        # Context token can attend to non context token
        # Non context token cannot attend to context token
        # Non context token should follow a causal mask
        n_tokens = latents.size(1)
        dec_attn_mask = torch.zeros(
            (n_tokens, n_tokens), dtpye=torch.bool, device=self.device
        )
        dec_attn_mask[:, : self.n_ctx_tokens] = True
        dec_attn_mask[self.n_ctx_tokens :, self.n_ctx_tokens :] = (
            dec_time_idx[:, None] < dec_time_idx[None, :]
        )

        dec_attn_mask = repeat(
            dec_attn_mask,
            "b n_1 n_2 -> (b dec_heads) n_1 n_2",
            dec_heads=self.dec_heads,
        )
        dec_output = self.decoder(latents, mask=dec_attn_mask)

        # remove context tokens at the start of the sequence
        output = dec_output[:, self.n_ctx_tokens :]
        output = self.dec_dropout_out(output)

        if self.is_ssl:
            # compute rates
            output = output[:, -n_masked_tokens:]
            rates = self.output_to_units(output)

            return {"output": rates}

        else:
            # compute behavior
            output = output[:, -n_bhvr_tokens:]
            bhvr = self.output_to_bhvr(output)

            return {"output": bhvr}
