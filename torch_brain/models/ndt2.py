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
    """NDT2 model from `Ye et al. 2023, Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity <https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1>`_.

    TODO

    Args:
        dim: Dimension of all embeddings,
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
    """

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

        ### Prepare encoder_tokens
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
            (nb_bins, nb_units_patches, self.units_per_patch), dtype=np.bool_
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

        ### Encoder tokens
        enc_units_patch = units_patch_shuffled[:encoder_frac]
        enc_time_idx = time_idx_shuffled[:encoder_frac]
        enc_space_idx = space_idx_shuffled[:encoder_frac]

        ### Encoder attention mask
        # need to pad at the beginning the attention mask to take the context tokens into account
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

        ### Decoder attention masks
        # need to pad at the attention masks to take the context tokens into account
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
                # context
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder
                "units_patch": pad(enc_units_patch),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_attn_mask": pad2d(enc_attn_mask),
                # decoder
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
    """NDT2 model from `Ye et al. 2023, Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity <https://www.biorxiv.org/content/10.1101/2023.09.18.558113v1>`_.
    TODO
    """

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
        dec_time_pool="mean",
        activation: str = "gelu",
        pre_norm: bool = False,
        lag: bool = False,
        bhvr_lag_bins: int = 0,
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
        self.pool = dec_time_pool
        self.readout_spec = readout_spec
        self.lag = lag
        if lag:
            self.bhvr_lag_bins = bhvr_lag_bins

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
            (nb_bins, nb_units_patches, self.units_per_patch), dtype=np.bool_
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

        ### Encoder attention mask
        # need to pad at the beginning the attention mask to take the context tokens into account
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
            if self.lag:
                # allow looking N-bins of neural data into the "future";
                # we back-shift during the actual decode comparison.
                bhvr_time_idx = bhvr_time_idx + self.bhvr_lag_bins
                bhvr_time_idx[-self.bhvr_lag_bins :] = 0

        dec_time_idx = np.concatenate([pooled_time_idx, bhvr_time_idx], axis=0)

        ### Decoder attention masks
        # need to pad at the attention masks to take the context tokens into account
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
        bhvr_timestamps, bhvr_values, _, _ = prepare_for_readout(
            data, self.readout_spec
        )

        # TODO binning (with interpolation)
        def interpolation(x, t):
            missing = 50 - t.shape[0]
            if missing == 0:
                return x
            else:
                pad_width = (0, missing) if x.ndim == 1 else ((0, missing), (0, 0))
                return np.pad(x, pad_width, mode="constant", constant_values=1.0)

        bhvr = interpolation(bhvr_values, bhvr_timestamps)

        eval_mask = np.ones(len(bhvr), dtype=np.bool_)
        if self.lag:
            # exclude the last N-bins (N=self.bhvr_lag_bins)
            bhvr[..., : -self.bhvr_lag_bins] = 0
            eval_mask[: -self.bhvr_lag_bins] = False

        data_dict = {
            "model_inputs": {
                # context
                "session_idx": session_idx,
                "subject_idx": subject_idx,
                "task_idx": task_idx,
                # encoder
                "units_patch": pad(units_patch),
                "enc_time_idx": pad(enc_time_idx),
                "enc_space_idx": pad(enc_space_idx),
                "enc_attn_mask": pad2d(enc_attn_mask),
                "enc_time_pad_idx": track_mask(enc_time_idx),
                # decoder
                "dec_time_idx": pad(dec_time_idx),
                "dec_attn_mask": pad2d(dec_attn_mask),
            },
            "target": bhvr,
            "eval_mask": eval_mask,
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

        # Pooling
        b, _, h = latents.size()
        size = (b, self.bin_size + 1, h)  # +1 to handdle padding
        pooled_latents = torch.zeros(size, device=latents.device, dtype=latents.dtype)

        # handdle padding
        index = torch.where(enc_time_pad_idx, enc_time_idx, self.bin_size)
        index = index.unsqueeze(-1).expand(-1, -1, h)
        pooled_latents.scatter_reduce_(
            dim=1, index=index, src=latents, reduce=self.pool, include_self=False
        )
        pooled_latents = pooled_latents[:, :-1]  # remove padding

        bhv_l = dec_time_idx.size(1) - self.bin_size
        bhvr_tokens = repeat(self.bhvr_emb, "h -> b bhv_l h", b=b, bhv_l=bhv_l)

        latents = torch.cat([pooled_latents, bhvr_tokens], dim=1)

        # Decoder forward pass
        latents = self.dec_dropout_in(latents)
        # add the temporal embedings (postions are not used with this decoder)
        latents = latents + self.dec_time_emb(dec_time_idx)

        # append context tokens at the start of the sequence
        if self.nb_ctx_tokens > 0:
            # detach the context tokens because don't want to uncalibrate the ctx_emb from the SSL pretraining
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

        return {"output": bhvr}
