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


class NDT2(nn.Module):
    def __init__(
        self,
        dim,
        units_per_patch: int,
        max_bincount: int,
        max_time_patches: int,
        max_space_patches: int,
        bin_time: float,
        ctx_time: float,
        mask_ratio: float,
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

        # Context tokens
        self.tokenize_session = tokenize_session
        self.tokenize_subject = tokenize_subject
        self.tokenize_task = tokenize_task
        self.nb_ctx_tokens = 0
        if self.tokenize_session:
            self.session_emb = InfiniteVocabEmbedding(dim)
            self.session_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1
        if self.tokenize_subject:
            self.subject_emb = InfiniteVocabEmbedding(dim)
            self.subject_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1
        if self.tokenize_task:  # more about dataset than task
            self.task_emb = InfiniteVocabEmbedding(dim)
            self.task_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1

        # Encoder
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

        # Between encoder and decoder we used masked token to let them be reconstructed
        self.masked_emb = nn.Parameter(torch.randn(dim))

        # Decoder
        self.dec_time_emb = Embedding(max_time_patches, dim)
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

        # SSL readout and loss
        self.output_to_units = nn.Linear(dim, units_per_patch)

    def tokenize(self, data: Data) -> Dict:
        ### Context tokens
        session_idx, subject_idx, task_idx = [], [], []
        if self.tokenize_session:
            session_idx = self.session_emb.tokenizer(data.session.id)
        if self.tokenize_subject:
            subject_idx = self.subject_emb.tokenizer(data.subject.id)
        if self.tokenize_task:
            task_idx = self.task_emb.tokenizer(data.brainset.id)

        ### Encoder_tokens
        nb_units = len(data.units.id)

        # self.max_bincount is used as the padding input
        units_bincount = bin_spikes(
            data.spikes, nb_units, self.bin_time, right=True, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        nb_units_patches = int(np.ceil(nb_units / self.units_per_patch))
        extra_units = nb_units_patches * self.units_per_patch - nb_units

        if extra_units > 0:
            bottom_pad = ((0, extra_units), (0, 0))
            units_bincount = np.pad(
                units_bincount,
                bottom_pad,
                mode="constant",
                constant_values=self.max_bincount,
            )

        nb_bins = units_bincount.shape[1]

        # flattened patches, here major hack to have time before space, as in o.g. NDT2 (nb_units, time_length)
        units_patch = rearrange(
            units_bincount,
            "(n pn) t -> (t n) pn",
            n=nb_units_patches,
            pn=self.units_per_patch,
            t=nb_bins,
        )

        # last patches may have fewer units
        extra_units_mask = np.ones(
            (nb_bins, nb_units_patches, self.units_per_patch), dtype=np.bool8
        )
        extra_units_mask[:, -1, -extra_units:] = False
        extra_units_mask = rearrange(
            extra_units_mask,
            "t n pn -> (t n) pn",
            t=nb_bins,
            n=nb_units_patches,
            pn=self.units_per_patch,
        )

        # time and space indices for flattened patches
        time_idx = np.arange(nb_bins, dtype=np.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=nb_units_patches)
        space_idx = np.arange(nb_units_patches, dtype=np.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=nb_bins)

        total_bins = nb_bins * nb_units_patches
        encoder_frac = int((1 - self.mask_ratio) * total_bins)

        # first part of the tokens are fed to the encoder
        # last will be masked and reconstructed by the decoder
        shuffle = torch.randperm(total_bins)

        units_patch_shuffled = units_patch[shuffle]
        time_idx_shuffled = time_idx[shuffle]
        space_idx_shuffled = space_idx[shuffle]
        extra_units_mask_shuffled = extra_units_mask[shuffle]

        enc_units_patch = units_patch_shuffled[:encoder_frac]
        enc_time_idx = time_idx_shuffled[:encoder_frac]
        enc_space_idx = space_idx_shuffled[:encoder_frac]

        enc_attn_mask = enc_time_idx[:, None] < enc_time_idx[None, :]
        top_pad = ((self.nb_ctx_tokens, 0), (0, 0))
        enc_attn_mask = np.pad(enc_attn_mask, top_pad, "constant", constant_values=True)
        left_pad = ((0, 0), (self.nb_ctx_tokens, 0))
        enc_attn_mask = np.pad(
            enc_attn_mask, left_pad, "constant", constant_values=False
        )

        ### Decoder tokens
        masked_time_idx = time_idx_shuffled[encoder_frac:]
        masked_space_idx = space_idx_shuffled[encoder_frac:]

        enc_dec_attn_mask = enc_time_idx[:, None] < masked_time_idx[None, :]
        top_pad = ((self.nb_ctx_tokens, 0), (0, 0))
        enc_dec_attn_mask = np.pad(
            enc_dec_attn_mask, top_pad, "constant", constant_values=True
        )

        dec_enc_attn_mask = masked_time_idx[:, None] < enc_time_idx[None, :]
        left_pad = ((0, 0), (self.nb_ctx_tokens, 0))
        dec_enc_attn_mask = np.pad(
            dec_enc_attn_mask, left_pad, "constant", constant_values=False
        )

        dec_dec_attn_mask = masked_time_idx[:, None] < masked_time_idx[None, :]

        ### Target tokens
        tgt_units_patch = units_patch_shuffled[encoder_frac:]
        extra_units_mask = extra_units_mask_shuffled[encoder_frac:]

        data_dict = {
            "model_inputs": {
                # context_tokens
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder_tokens
                "units_patch": pad(enc_units_patch),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_attn_mask": pad2d(enc_attn_mask),
                # decoder_tokens
                "masked_time_idx": pad(masked_time_idx),
                "masked_space_idx": pad(masked_space_idx),
                "enc_dec_attn_mask": pad2d(enc_dec_attn_mask),
                "dec_enc_attn_mask": pad2d(dec_enc_attn_mask),
                "dec_dec_attn_mask": pad2d(dec_dec_attn_mask),
            },
            "target": pad(tgt_units_patch),
            "extra_units_mask": pad(extra_units_mask),
        }
        return data_dict

    def forward(
        self,
        session_idx: Optional[TensorType["b", int]],
        subject_idx: Optional[TensorType["b", int]],
        task_idx: Optional[TensorType["b", int]],
        units_patch: TensorType["b", "r * max_l", "patch_dim", int],
        enc_time_idx: TensorType["b", "r * max_l", int],
        enc_space_idx: TensorType["b", "r * max_l", int],
        enc_attn_mask: TensorType["b", "n_ctx + r * max_l", "n_ctx + r * max_l", int],
        masked_time_idx: TensorType["b", "(1-r) * max_l", int],
        masked_space_idx: TensorType["b", "(1-r) * max_l", int],
        enc_dec_attn_mask: TensorType["b", "n_ctx + r * max_l", "(1-r) * max_l", int],
        dec_enc_attn_mask: TensorType["b", "(1-r) * max_l", "n_ctx + r * max_l", int],
        dec_dec_attn_mask: TensorType["b", "(1-r) * max_l", "(1-r) * max_l", int],
    ) -> Dict:
        # Prepare context tokens
        ctx_tokens = []
        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
        if self.tokenize_task:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

        if self.nb_ctx_tokens > 0:
            ctx_emb = torch.stack(ctx_tokens, dim=1)

        # From unit patches to input tokens
        inputs = self.patch_emb(units_patch).flatten(-2, -1)
        inputs = self.enc_dropout_in(inputs)

        # add the spacio-tempo embedings
        inputs = (
            inputs + self.enc_time_emb(enc_time_idx) + self.enc_space_emb(enc_space_idx)
        )

        # append context tokens at the start of the sequence
        if self.nb_ctx_tokens > 0:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        # Encoder forward pass
        attn_mask = repeat(
            enc_attn_mask,
            "b n_1 n_2 -> (b enc_heads) n_1 n_2",
            enc_heads=self.enc_heads,
        )
        enc_output = self.encoder(inputs, mask=attn_mask)

        # remove context tokens at the start of the sequence
        latents = enc_output[:, self.nb_ctx_tokens :]
        latents = self.enc_dropout_out(latents)

        # append the masked tokens to the encoder output (i.e. latents)
        b, nb_masked_tokens = masked_time_idx.size(0), masked_time_idx.size(1)
        masked_tokens = repeat(self.masked_emb, "h -> b n h", b=b, n=nb_masked_tokens)
        latents = torch.cat([latents, masked_tokens], dim=1)
        dec_time_idx = torch.cat([enc_time_idx, masked_time_idx], dim=1)
        dec_space_idx = torch.cat([enc_space_idx, masked_space_idx], dim=1)

        # Decoder forward pass
        latents = self.dec_dropout_in(latents)
        # add the spacio-tempo embedings
        latents = (
            latents
            + self.dec_time_emb(dec_time_idx)
            + self.dec_space_emb(dec_space_idx)
        )

        # append context tokens at the start of the sequence
        if self.nb_ctx_tokens > 0:
            latents = torch.cat([ctx_emb, latents], dim=1)

        # dec_attn_mask = [A, B
        #                  C, D]
        # with A=enc_attn_mask, B=enc_dec_attn_mask, C=dec_enc_attn_mask, and D=dec_dec_attn_mask
        attn_mask_low = torch.cat([enc_attn_mask, enc_dec_attn_mask], dim=2)
        attn_mask_up = torch.cat([dec_enc_attn_mask, dec_dec_attn_mask], dim=2)
        dec_attn_mask = torch.cat([attn_mask_low, attn_mask_up], dim=1)

        dec_attn_mask = repeat(
            dec_attn_mask,
            "b n_1 n_2 -> (b dec_heads) n_1 n_2",
            dec_heads=self.dec_heads,
        )
        dec_output = self.decoder(latents, mask=dec_attn_mask)

        # remove context tokens at the start of the sequence
        output = dec_output[:, self.nb_ctx_tokens :]
        output = self.dec_dropout_out(output)

        # compute rates
        output = output[:, -nb_masked_tokens:]
        rates = self.output_to_units(output)

        return {"rates": rates}


class NDT2_Supervised(nn.Module):
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
        readout_spec: ModalitySpec,
        activation: str = "gelu",
        pre_norm: bool = False,
        pool="mean",
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

        # Context tokens
        self.tokenize_session = tokenize_session
        self.tokenize_subject = tokenize_subject
        self.tokenize_task = tokenize_task
        self.nb_ctx_tokens = 0
        if self.tokenize_session:
            self.session_emb = InfiniteVocabEmbedding(dim)
            self.session_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1
        if self.tokenize_subject:
            self.subject_emb = InfiniteVocabEmbedding(dim)
            self.subject_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1
        if self.tokenize_task:  # more about dataset than task
            self.task_emb = InfiniteVocabEmbedding(dim)
            self.task_bias = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
            self.nb_ctx_tokens += 1

        # Encoder
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

        # Between encoder and decoder we used query token to decode the behavior at a given time
        self.bhvr_emb = nn.Parameter(torch.randn(dim))

        # Decoder
        self.dec_time_emb = Embedding(max_time_patches, dim)
        # dec_space_emb is not used in the supervised decoder because we dont encode the neurons position (they are pooled)

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

        self.task = "regression"
        self.pool = pool
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

        ### Encoder_tokens
        nb_units = len(data.units.id)

        # self.max_bincount is used as the padding input
        units_bincount = bin_spikes(
            data.spikes, nb_units, self.bin_time, right=True, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        nb_units_patches = int(np.ceil(nb_units / self.units_per_patch))
        extra_units = nb_units_patches * self.units_per_patch - nb_units

        if extra_units > 0:
            bottom_pad = ((0, extra_units), (0, 0))
            units_bincount = np.pad(
                units_bincount,
                bottom_pad,
                mode="constant",
                constant_values=self.max_bincount,
            )

        nb_bins = units_bincount.shape[1]

        # flattened patches, here major hack to have time before space, as in o.g. NDT2 (nb_units, time_length)
        units_patch = rearrange(
            units_bincount,
            "(n pn) t -> (t n) pn",
            n=nb_units_patches,
            pn=self.units_per_patch,
            t=nb_bins,
        )

        # last patches may have fewer units
        extra_units_mask = np.ones(
            (nb_bins, nb_units_patches, self.units_per_patch), dtype=np.bool8
        )
        extra_units_mask[:, -1, -extra_units:] = False
        extra_units_mask = rearrange(
            extra_units_mask,
            "t n pn -> (t n) pn",
            t=nb_bins,
            n=nb_units_patches,
            pn=self.units_per_patch,
        )

        # time and space indices for flattened patches
        enc_time_idx = np.arange(nb_bins, dtype=np.int32)
        enc_time_idx = repeat(enc_time_idx, "t -> (t n)", n=nb_units_patches)
        enc_space_idx = np.arange(nb_units_patches, dtype=np.int32)
        enc_space_idx = repeat(enc_space_idx, "n -> (t n)", t=nb_bins)

        enc_attn_mask = enc_time_idx[:, None] < enc_time_idx[None, :]
        top_pad = ((self.nb_ctx_tokens, 0), (0, 0))
        enc_attn_mask = np.pad(enc_attn_mask, top_pad, "constant", constant_values=True)
        left_pad = ((0, 0), (self.nb_ctx_tokens, 0))
        enc_attn_mask = np.pad(
            enc_attn_mask, left_pad, "constant", constant_values=False
        )

        ### Decoder tokens
        pooled_time_idx = np.arange(nb_bins, dtype=np.int32)
        if self.task == "classification":
            bhvr_time_idx = np.array([nb_bins - 1], dtype=np.int32)
        else:
            bhvr_time_idx = np.arange(nb_bins, dtype=np.int32)
            # self.bhvr_lag_bins = 0
            # if self.lag:
            #     # allow looking N-bins of neural data into the "future";
            #     # we back-shift during the actual decode comparison.
            #     bhvr_time_idx = bhvr_time_idx + self.bhvr_lag_bins
            # TODO move it before (tokenizer)
            # if self.lag:
            #     # exclude the last N-bins
            #     bhvr = bhvr[:, : -self.bhvr_lag_bins]
            #     # add to the left N-bins to match the lag
            #     bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

        dec_time_idx = np.concatenate([pooled_time_idx, bhvr_time_idx], axis=0)

        enc_enc_attn_mask = pooled_time_idx[:, None] < pooled_time_idx[None, :]
        enc_enc_attn_mask = np.pad(
            enc_enc_attn_mask, top_pad, "constant", constant_values=True
        )
        enc_enc_attn_mask = np.pad(
            enc_enc_attn_mask, left_pad, "constant", constant_values=False
        )

        enc_dec_attn_mask = pooled_time_idx[:, None] < bhvr_time_idx[None, :]
        enc_dec_attn_mask = np.pad(
            enc_dec_attn_mask, top_pad, "constant", constant_values=True
        )

        dec_enc_attn_mask = bhvr_time_idx[:, None] < pooled_time_idx[None, :]
        dec_enc_attn_mask = np.pad(
            dec_enc_attn_mask, left_pad, "constant", constant_values=False
        )

        dec_dec_attn_mask = bhvr_time_idx[:, None] < bhvr_time_idx[None, :]
        dec_attn_mask_up = np.concatenate(
            [enc_enc_attn_mask, enc_dec_attn_mask], axis=1
        )
        dec_attn_mask_low = np.concatenate(
            [dec_enc_attn_mask, dec_dec_attn_mask], axis=1
        )
        dec_attn_mask = np.concatenate([dec_attn_mask_up, dec_attn_mask_low], axis=0)

        ### Readout/Target tokens
        output_timestamps, output_values, output_weights, eval_mask = (
            prepare_for_readout(data, self.readout_spec)
        )
        # TODO binning (with interpolation)
        bin = lambda x, t: x[::20]

        tgt = bin(output_values, output_timestamps)

        data_dict = {
            "model_inputs": {
                # context_tokens
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder_tokens
                "units_patch": pad(units_patch),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_attn_mask": pad2d(enc_attn_mask),
                "enc_time_pad_idx": track_mask(enc_time_idx),
                # decoder_tokens
                "dec_time_idx": dec_time_idx,
                "dec_attn_mask": dec_attn_mask,
            },
            "target": tgt,
        }
        return data_dict

    def forward(
        self,
        session_idx: Optional[TensorType["b", int]],
        subject_idx: Optional[TensorType["b", int]],
        task_idx: Optional[TensorType["b", int]],
        units_patch: TensorType["b", "max_l", "patch_dim", int],
        enc_time_idx: TensorType["b", "max_l", int],
        enc_space_idx: TensorType["b", "max_l", int],
        enc_attn_mask: TensorType["b", "n_ctx + max_l", "n_ctx + max_l", int],
        enc_time_pad_idx: TensorType["b", "max_l", int],
        dec_time_idx: TensorType["b", "nb_bins + tgt_l", int],
        dec_attn_mask: TensorType["b", "n_ctx + 2*tgt_l", "n_ctx + 2*tgt_l", int],
    ) -> Dict:
        # Prepare context tokens
        ctx_tokens = []
        if self.tokenize_session:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_bias)
        if self.tokenize_subject:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_bias)
        if self.tokenize_task:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_bias)

        if self.nb_ctx_tokens > 0:
            ctx_emb = torch.stack(ctx_tokens, dim=1)

        # From unit patches to input tokens
        inputs = self.patch_emb(units_patch).flatten(-2, -1)
        inputs = self.enc_dropout_in(inputs)

        # add the spacio-tempo embedings
        inputs = (
            inputs + self.enc_time_emb(enc_time_idx) + self.enc_space_emb(enc_space_idx)
        )

        # append context tokens at the start of the sequence
        if self.nb_ctx_tokens > 0:
            inputs = torch.cat([ctx_emb, inputs], dim=1)

        # Encoder forward pass
        attn_mask = repeat(
            enc_attn_mask,
            "b n_in_1 n_in_2 -> (b enc_heads) n_in_1 n_in_2",
            enc_heads=self.enc_heads,
        )
        enc_output = self.encoder(inputs, mask=attn_mask)

        # remove context tokens at the start of the sequence
        latents = enc_output[:, self.nb_ctx_tokens :]
        latents = self.enc_dropout_out(latents)

        # TODO check the t + 1 for padding
        # encoder_out = pooled_features[:, :-1]  # remove padding

        # Pooling
        b, _, h = latents.size()
        size = (b, self.bin_size, h)
        pooled_latents = torch.zeros(size, device=latents.device, dtype=latents.dtype)
        index = repeat(enc_time_pad_idx, "b max_l -> b max_l h", h=h).to(torch.long)
        pooled_latents = pooled_latents.scatter_reduce(
            dim=1, index=index, src=latents, reduce=self.pool, include_self=False
        )

        bhv_l = dec_time_idx.size(1) - self.bin_size
        bhvr_tokens = repeat(self.bhvr_emb, "h -> b bhv_l h", b=b, bhv_l=bhv_l)

        latents = torch.cat([pooled_latents, bhvr_tokens], dim=1)

        # Decoder forward pass
        latents = self.dec_dropout_in(latents)
        # add the temporal embedings (postions are not used with this decoder)
        latents = latents + self.dec_time_emb(dec_time_idx)

        # append context tokens at the start of the sequence
        if self.nb_ctx_tokens > 0:
            # detach the context tokens because don't to uncalibrate the ctx_emb from the SSL pretraining
            ctx_emb = ctx_emb.detach()
            latents = torch.cat([ctx_emb, latents], dim=1)

        dec_attn_mask = repeat(
            dec_attn_mask,
            "b n_in_1 n_in_2 -> (b dec_heads) n_in_1 n_in_2",
            dec_heads=self.dec_heads,
        )
        dec_output = self.decoder(latents, mask=dec_attn_mask)
        output = dec_output[:, self.nb_ctx_tokens :]
        output = self.dec_dropout_out(output)

        # compute behavior
        output = output[:, -bhv_l:]
        bhvr = self.output_to_bhvr(output)

        # TODO if lagging
        # if self.lag:
        #     # exclude the last N-bins
        #     bhvr = bhvr[:, : -self.bhvr_lag_bins]
        #     # add to the left N-bins to match the lag
        #     bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

        return {"output": bhvr}

        if self.task == "classification":
            tgt = bhvr_tgt.argmax(dim=-1).cpu()
            pred = bhvr.argmax(dim=-1).cpu()
            acc = accuracy_score(tgt, pred)
            balanced_acc = balanced_accuracy_score(tgt, pred)
            return {
                "loss": loss,
                "acc": acc,
                "balanced_acc": balanced_acc,
                "pred": bhvr,
            }

        raise NotImplementedError

        # pass
        # # b, t_enc = encoder_out.size()[:2]
        # time = torch.arange(encoder_frac)

        # if self.task == "classification":
        #     query_time = torch.tensor([encoder_frac])
        # else:
        #     t = bhvr.size(1)
        #     query_time = torch.arange(t)

        # if self.lag:
        #     # allow looking N-bins of neural data into the "future";
        #     # we back-shift during the actual decode comparison.
        #     query_time = time + self.bhvr_lag_bins

        # length_mask = ~self.temporal_pad_mask(decoder_out, max_length)
        # if self.lag:
        #     length_mask[:, : self.bhvr_lag_bins] = False

        # latent_time_idx = torch.cat([time, query_time], dim=1)
        # latent_space_idx = torch.zeros_like(latent_time_idx)

        # latent_mask = torch.ones(encoder_frac, dtype=bool).float()
        # latent_mask = latent_mask.scatter_reduce(
        #     src=torch.zeros_like(time_idx).float(),
        #     dim=1,
        #     index=time_idx.to(torch.long),
        #     reduce="prod",
        #     include_self=False,
        # ).bool()

        # if encoder_out.size(1) < bhvr.size(2):
        #     to_add = bhvr.size(2) - latent_mask.size(1)
        #     latent_mask = F.pad(latent_mask, (0, to_add), value=True)

        # # TODO check where max_length is from
        # max_lenght = batch["bhvr_mask"].sum(1, keepdim=True)
        # token_position = torch.arange(latent_mask.size(1))
        # token_position = rearrange(token_position, "t -> () t")
        # query_pad_mask = token_position >= max_lenght

        # # TODO check but this should be easier to do
        # query_pad_mask = batch["bhvr_mask"]

        # latent_mask = torch.cat([latent_mask, query_pad_mask], dim=1)

        # decoder_attn_mask = ~input_causality

        # if not self.is_ssl:
        #     # -- Behavior
        #     # TODO add a callable in the config to handle this access to the bhvr data
        #     bhvr = getattr(data, self.bhvr_key)
        #     try:
        #         bhvr = getattr(bhvr, self.bhvr_key)
        #         # One hot encoding of the behavior
        #         bhvr = np.eye(self.bhvr_dim)[bhvr]
        #     except:
        #         pass

        #     # TODO should be more general
        #     if self.ibl_binning:
        #         intervals = np.c_[data.trials.start, data.trials.end]
        #         params = {
        #             "interval_len": 2,
        #             "binsize": 0.02,
        #             "single_region": False,
        #             "align_time": "stimOn_times",
        #             "time_window": (-0.5, 1.5),
        #             "fr_thresh": 0.5,
        #         }

        #         # TODO use mask_dict and refactor
        #         bhvr_data = getattr(data, self.bhvr_key)
        #         bhvr_value = bhvr_data.values

        #         behave_dict, mask_dict = bin_behaviors(
        #             bhvr_data.timestamps,
        #             bhvr_value.squeeze(),
        #             intervals=intervals,
        #             beh=self.bhvr_key,
        #             **params,
        #         )
        #         bhvr = behave_dict[self.bhvr_key][:, None]

        #     batch["bhvr"] = pad(bhvr)
        #     batch["bhvr_mask"] = track_mask(bhvr)

    # def forward(
    #     self,
    #     latents: TensorType["b", "n_in + n_lat", "dim", int],
    #     time_idx: TensorType["b", "n_in + n_lat", int],
    #     space_idx: TensorType["b", "n_in + n_lat", int],
    #     decoder_attn_mask: TensorType["b", "n_in + n_lat", "n_in + n_lat", bool],
    #     ctx_emb: Optional[TensorType["b", "n_ctx", "dim", float]],
    #     target: TensorType["b", "n_lat", "patch_dim", int],
    #     extra_units_mask: Optional[TensorType["b", "n_lat", "patch_dim", bool]],
    # ) -> Dict:
    #     """
    #     TODO update w/ eval_mode if needed
    #     """
    # prepare decoder input

    # TODO for bhvr
    # latent_time_idx: torch.Tensor
    # latent_space_idx: torch.Tensor
    # pad_mask = latent_mask

    # times = input_time_idx

    # b, nb_tokens, h = encoder_out.shape
    # b = encoder_out.shape[0]
    # t = times.max() + 1
    # h = encoder_out.shape[-1]
    # dev = encoder_out.device
    # pool = self.decode_time_pool

    # # t + 1 for padding
    # pooled_features = torch.zeros(
    #     b, t + 1, h, device=dev, dtype=encoder_out.dtype
    # )

    # time_with_pad_marked = torch.where(pad_mask, t, times)
    # index = repeat(time_with_pad_marked, "b t -> b t h", h=h).to(torch.long)
    # pooled_features = pooled_features.scatter_reduce(
    #     src=encoder_out, dim=1, index=index, reduce=pool, include_self=False
    # )
    # encoder_out = pooled_features[:, :-1]  # remove padding

    # bhvr_tgt = batch["bhvr"]

    # b, t = bhvr.size()[:2]
    # query_tokens = repeat(self.query_token, "h -> b t h", b=b, t=t)
    # if encoder_out.size(1) < t:
    #     to_add = t - encoder_out.size(1)
    #     encoder_out = F.pad(encoder_out, (0, 0, 0, to_add), value=0)
    # decoder_in = torch.cat([encoder_out, query_tokens], dim=1)

    # else:
    #     # compute behavior
    #     nb_query_tokens = bhvr_tgt.size(1)
    #     decoder_out = decoder_out[:, -nb_query_tokens:]
    #     bhvr = self.out(decoder_out)

    #     # TODO move it before (tokenizer)
    #     if self.lag:
    #         # exclude the last N-bins
    #         bhvr = bhvr[:, : -self.bhvr_lag_bins]
    #         # add to the left N-bins to match the lag
    #         bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

    #     # Compute loss & r2
    #     loss_mask = loss_mask
    #     no_nan_mask_decoder_out = ~torch.isnan(decoder_out).any(-1)
    #     no_nan_mask_target = ~torch.isnan(bhvr_tgt).any(-1)
    #     no_nan_mask = no_nan_mask_decoder_out & no_nan_mask_target
    #     loss_mask = loss_mask & no_nan_mask
    #     bhvr_tgt = bhvr_tgt.to(bhvr.dtype)  # TODO make it cleanner
    #     loss = self.loss(bhvr, bhvr_tgt)
    #     # TODO this way of computing a loss
    #     # self.modality_spec.loss_fn
    #     loss = loss[loss_mask].mean()

    #     if self.task == "regression":
    #         tgt = bhvr_tgt[loss_mask].float().detach().cpu()
    #         pred = bhvr[loss_mask].float().detach().cpu()
    #         r2 = r2_score(tgt, pred, multioutput="raw_values")
    #         if r2.mean() < -10:
    #             r2 = np.zeros_like(r2)
    #         return {"loss": loss, "r2": r2, "pred": bhvr}

    #     if self.task == "classification":
    #         tgt = bhvr_tgt.argmax(dim=-1).cpu()
    #         pred = bhvr.argmax(dim=-1).cpu()
    #         acc = accuracy_score(tgt, pred)
    #         balanced_acc = balanced_accuracy_score(tgt, pred)
    #         return {
    #             "loss": loss,
    #             "acc": acc,
    #             "balanced_acc": balanced_acc,
    #             "pred": bhvr,
    #         }

    #     raise NotImplementedError
