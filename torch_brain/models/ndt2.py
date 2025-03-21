import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from temporaldata import ArrayDict, Data
from torch.nn import Transformer

# from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from torchtyping import TensorType

from torch_brain.data import chain, pad, track_mask
from torch_brain.nn import Embedding, InfiniteVocabEmbedding

# from torch_brain.utils.binning import bin_behaviors, bin_spikes
from torch_brain.utils.binning import bin_spikes


class NDT2(nn.Module):
    def __init__(
        self,
        is_ssl: bool,
        dim,
        units_per_patch: int,
        max_bincount: int,
        pad_value: int,
        max_time_patches: int,
        max_space_patches: int,
        bin_time: float,
        ctx_time: float,
        mask_ratio: float,
        tokenize_session: bool,
        tokenize_subject: bool,
        tokenize_task: bool,
        depth: int,
        heads: int,
        dropout: float,
        ffn_mult: float,
        causal: bool = True,
        activation: str = "gelu",
        pre_norm: bool = False,
        predictor_cfg: Dict = None,
        bhv_decoder_cfg: Dict = None,
        unsorted=False,
        bhvr_key="finger.vel",
        bhvr_dim=2,
        ibl_binning=False,
    ):
        super().__init__()
        # not sure about this TODO check
        self.max_bincount = max_bincount
        self.patch_emb = Embedding(
            max_bincount + 1, dim // units_per_patch, padding_idx=max_bincount
        )
        self.time_emb = Embedding(max_time_patches, dim)
        self.space_emb = Embedding(max_space_patches, dim)

        self.session_emb, self.subject_emb, self.task_emb = None, None, None
        if tokenize_session:
            self.session_emb = InfiniteVocabEmbedding(dim)
            self.session_flag = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        if tokenize_subject:
            self.subject_emb = InfiniteVocabEmbedding(dim)
            self.subject_flag = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        if tokenize_task:  # more about dataset than task
            self.task_emb = InfiniteVocabEmbedding(dim)
            self.task_flag = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

        self.query_emb = nn.Parameter(torch.randn(dim))

        # Encoder
        self.heads = heads
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, depth)

        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        self.bin_time = bin_time
        self.ctx_time = ctx_time
        float_modulo_test = lambda x, y, eps=1e-6: np.abs(x - y * np.round(x / y)) < eps
        assert float_modulo_test(ctx_time, bin_time)

        self.bin_size = int(np.round(ctx_time / bin_time))
        self.units_per_patch = units_per_patch
        self.mask_ratio = mask_ratio
        self.pad_value = pad_value

        self.unsorted = unsorted
        self.is_ssl = is_ssl
        self.bhvr_key = bhvr_key
        self.ibl_binning = ibl_binning
        self.bhvr_dim = bhvr_dim

        # Decoder
        self.decoder = Decoder(
            dim,
            depth,
            heads,
            dropout,
            max_time_patches,
            max_space_patches,
            ffn_mult,
            units_per_patch,
            decode_time_pool="mean",
            behavior_dim=bhvr_dim,
            bin_time=self.bin_time,
            causal=True,
            activation="gelu",
            pre_norm=False,
            behavior_lag=None,
            task="regression",
        )

    def forward(
        self,
        units_patch: TensorType["b", "n_in", "patch_dim", int],
        input_time_idx: TensorType["b", "n_in", int],
        input_space_idx: TensorType["b", "n_in", int],
        input_mask: TensorType["b", "n_in", int],
        encoder_attn_mask: TensorType["b x encoder_heads", "n_in", "n_in", int],
        latent_time_idx: TensorType["b", "n_lat", int],
        latent_space_idx: TensorType["b", "n_lat", int],
        latent_mask: TensorType["b", "n_in + n_lat", int],
        decoder_attn_mask: TensorType[
            "b x decoder_heads", "n_in + n_lat", "n_in + n_lat", int
        ],
        session_idx: Optional[TensorType["b", int]],
        subject_idx: Optional[TensorType["b", int]],
        task_idx: Optional[TensorType["b", int]],
        target: TensorType["b", "n_lat", "tgt_dim", int],
        target_mask: TensorType["b", "n_lat", int],
        channel_mask: Optional[TensorType["b", "n_lat", "n_unit_patch", bool]],
    ) -> Dict:
        # make input tokens
        inputs = self.patch_emb(units_patch).flatten(-2, -1)
        inputs = self.dropout_in(inputs)
        inputs = (
            inputs + self.time_emb(input_time_idx) + self.space_emb(input_space_idx)
        )

        # add context tokens at the end of the sequence
        ctx_tokens = []
        if self.session_emb is not None:
            ctx_tokens.append(self.session_emb(session_idx) + self.session_flag)
        if self.subject_emb is not None:
            ctx_tokens.append(self.subject_emb(subject_idx) + self.subject_flag)
        if self.task_emb is not None:
            ctx_tokens.append(self.task_emb(task_idx) + self.task_flag)

        nb_ctx_tokens = len(ctx_tokens)
        if nb_ctx_tokens > 0:
            ctx_emb = torch.stack(ctx_tokens, dim=1)
            inputs = torch.cat([inputs, ctx_emb], dim=1)

        # encoder forward pass
        latents = self.encoder(
            inputs, mask=encoder_attn_mask, src_key_padding_mask=input_mask
        )

        latents = latents[:, :-nb_ctx_tokens]
        latents = self.dropout_out(latents)

        b, t = latent_time_idx.size(0), latent_time_idx.size(1)
        query_emb = repeat(self.query_emb, "h -> b t h", b=b, t=t)
        latents = torch.cat([latents, query_emb], dim=1)
        latents_time_idx = torch.cat([input_time_idx, latent_time_idx], dim=1)
        latents_space_idx = torch.cat([input_space_idx, latent_space_idx], dim=1)

        return self.decoder(
            latents,
            latents_time_idx,
            latents_space_idx,
            latent_mask,
            decoder_attn_mask,
            ctx_emb,
            target,
            target_mask,
            channel_mask,
        )

    def tokenize(self, data: Data) -> Dict:
        num_units = len(data.units.id)

        # if self.unsorted:
        #     chan_nb_mapper = self.extract_chan_nb(data.units)
        #     spikes.unit_index = chan_nb_mapper.take(spikes.unit_index)
        #     # TODO do not work need to find an hack
        #     # nb_units = chan_nb_mapper.max() + 1
        #     num_units = 96
        units_bincount = bin_spikes(
            data.spikes, num_units, self.bin_time, dtype=np.int32
        )
        units_bincount = np.clip(units_bincount, 0, self.max_bincount - 1)

        nb_units = units_bincount.shape[0]
        nb_units_patches = int(np.ceil(nb_units / self.units_per_patch))
        extra_units = nb_units_patches * self.units_per_patch - nb_units

        if extra_units > 0:
            units_bincount = np.pad(
                units_bincount,
                [(0, extra_units), (0, 0)],
                mode="constant",
                constant_values=self.max_bincount,
            )

        nb_bins = units_bincount.shape[1]

        # major hack to have time before space, as in o.g. NDT2 (nb_units, time_length)
        # TODO could be mutch more cleaner
        units_patch = rearrange(
            units_bincount,
            "(n pn) t -> (t n) pn",
            n=nb_units_patches,
            pn=self.units_per_patch,
            t=nb_bins,
        )

        # time and space indices for flattened patches
        time_idx = np.arange(nb_bins, dtype=np.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=nb_units_patches)
        space_idx = np.arange(nb_units_patches, dtype=np.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=nb_bins)

        extra_units_mask = np.ones((nb_bins, self.units_per_patch), dtype=np.bool8)

        # last patch may have fewer units
        if extra_units > 0:
            first_idx = nb_units_patches - 1
            stride = nb_units_patches
            extra_units_mask[first_idx::stride, -extra_units:] = False

        encoder_frac = int((1 - self.mask_ratio) * nb_bins)
        target_frac = nb_bins - encoder_frac

        if self.is_ssl:
            shuffle = torch.randperm(nb_bins)

            units_patch_shuffled = units_patch[shuffle]
            time_idx_shuffled = time_idx[shuffle]
            space_idx_shuffled = space_idx[shuffle]
            extra_units_mask_shuffled = extra_units_mask[shuffle]

            in_units_patch = units_patch_shuffled[:encoder_frac]
            in_time_idx = time_idx_shuffled[:encoder_frac]
            in_space_idx = space_idx_shuffled[:encoder_frac]

            model_inputs = {
                "units_patch": pad(in_units_patch),
                "time_idx": pad(in_time_idx),
                "space_idx": pad(in_space_idx),
                "pad_mask": track_mask(in_units_patch),
            }

            lat_time_idx = time_idx_shuffled[encoder_frac:]
            lat_space_idx = space_idx_shuffled[encoder_frac:]

            model_latents = {
                "time_idx": pad(lat_time_idx),
                "space_idx": pad(lat_space_idx),
                "pad_mask": track_mask(lat_time_idx),
            }

            tgt_units_patch = units_patch_shuffled[encoder_frac:]
            extra_units_mask = extra_units_mask_shuffled[encoder_frac:]

            # TODO need to check that target_units has shape t' c
            # loss_mask = torch.ones(target_units.size(), dtype=torch.bool)

            # tmp = torch.arange(target_units.size(-1))
            # comparison = repeat(tmp, "c -> 1 t c", t=target_frac)
            # channel_mask = comparison < batch["channel_counts_target"].unsqueeze(-1)
            # loss_mask = loss_mask & channel_mask

            # target_mask = units_mask_shuffled[:, encoder_frac:]
            # loss_mask = loss_mask & target_mask.unsqueeze(-1)

            model_tagets = {
                "target": pad(tgt_units_patch),
                "pad_mask": track_mask(tgt_units_patch),
                "extra_units_mask": extra_units_mask,
            }

        else:
            pass
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

        # TODO add channel_counts
        # TODO be careful of track_mask that might not be anymore valid
        data_dict = {
            "model_inputs": model_inputs,
            "model_latents": model_latents,
            "model_tagets": model_tagets,
        }

        session_idx, subject_idx, task_idx = [], [], []
        if self.session_emb is not None:
            session_idx = self.session_emb.tokenizer(data.session.id)
        if self.subject_emb is not None:
            subject_idx = self.subject_emb.tokenizer(data.subject.id)
        if self.task_emb is not None:
            task_idx = self.task_emb.tokenizer(data.brainset.id)

        data_dict["session_idx"] = session_idx
        data_dict["subject_idx"] = subject_idx
        data_dict["task_idx"] = task_idx

        # output_timestamps, output_values, output_weights, eval_mask = (
        #     prepare_for_readout(data, self.readout_spec)
        # )

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

        return data_dict

    def attn_mask(self, batch, nb_ctx_tokens, encoder_heads, decoder_heads):
        input_time_idx = batch["model_inputs"]["time_idx"]
        latent_time_idx = batch["model_latents"]["time_idx"]
        latent_time_idx = torch.cat([input_time_idx, latent_time_idx], dim=1)

        encoder_attn_mask = input_time_idx[:, :, None] < input_time_idx[:, None, :]
        encoder_attn_mask = F.pad(
            encoder_attn_mask, (0, 0, 0, nb_ctx_tokens), value=True
        )
        encoder_attn_mask = F.pad(encoder_attn_mask, (0, nb_ctx_tokens), value=False)
        encoder_attn_mask = repeat(
            encoder_attn_mask, "b t1 t2 -> (b h) t1 t2", h=encoder_heads
        )

        decoder_attn_mask = latent_time_idx[:, :, None] < latent_time_idx[:, None, :]
        decoder_attn_mask = F.pad(
            decoder_attn_mask, (0, 0, 0, nb_ctx_tokens), value=True
        )
        decoder_attn_mask = F.pad(decoder_attn_mask, (0, nb_ctx_tokens), value=False)
        decoder_attn_mask = repeat(
            decoder_attn_mask, "b t1 t2-> (b h) t1 t2", h=decoder_heads
        )

        return encoder_attn_mask, decoder_attn_mask

    def pad_mask(self, batch, nb_ctx_tokens):
        input_mask = batch["model_inputs"]["pad_mask"]
        latent_mask = batch["model_latents"]["pad_mask"]
        latent_mask = torch.cat([input_mask, latent_mask], dim=1)

        input_mask = F.pad(input_mask, (0, nb_ctx_tokens), value=False)
        latent_mask = F.pad(latent_mask, (0, nb_ctx_tokens), value=False)

        return input_mask, latent_mask

    def extract_chan_nb(self, units: ArrayDict):
        channel_names = units.channel_name
        res = [int(chan_name.split(b" ")[-1]) for chan_name in channel_names]
        return np.array(res) - 1


class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dropout,
        max_time_patches,
        max_space_patches,
        ffn_mult,
        units_per_patch,
        decode_time_pool,
        behavior_dim,
        bin_time,
        causal=True,
        activation="gelu",
        pre_norm=False,
        behavior_lag=None,
        task="regression",
    ):
        super().__init__()
        self.is_ssl = True
        self.dim = dim
        self.units_per_patch = units_per_patch

        self.time_emb = nn.Embedding(max_time_patches, dim)
        self.space_emb = nn.Embedding(max_space_patches, dim)

        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=pre_norm,
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, depth)
        self.query_token = nn.Parameter(torch.randn(dim))

        self.output_to_units = nn.Linear(dim, units_per_patch)
        self.ssl_loss = nn.PoissonNLLLoss(reduction="none", log_input=True)

        # self.dim = dim
        # self.causal = causal
        # self.bin_time = bin_time
        # self.lag = behavior_lag
        # self.decode_time_pool = decode_time_pool
        # self.behavior_dim = behavior_dim
        # self.task = task
        # if self.lag:
        #     self.bhvr_lag_bins = round(self.lag / bin_time)

        # self.query_token = nn.Parameter(torch.randn(dim))

        # self.decoder = Transformer(
        #     dim=dim,
        #     depth=depth,
        #     heads=heads,
        #     dropout=dropout,
        #     max_time_patches=max_time_patches,
        #     max_space_patches=max_space_patches,
        #     ffn_mult=ffn_mult,
        #     causal=causal,
        #     activation=activation,
        #     pre_norm=pre_norm,
        #     allow_embed_padding=True,
        # )
        # self.out = nn.Linear(dim, self.behavior_dim)

        # if self.task == "regression":
        #     self.loss = nn.MSELoss(reduction="none")
        # elif self.task == "classification":
        #     self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        latents: TensorType["b", "n_in + n_lat", "dim", int],
        time_idx: TensorType["b", "n_in + n_lat", int],
        space_idx: TensorType["b", "n_in + n_lat", int],
        latent_mask: TensorType["b", "n_in + n_lat", bool],
        decoder_attn_mask: TensorType[
            "b x decoder_heads", "n_in + n_lat", "n_in + n_lat", bool
        ],
        ctx_emb: Optional[TensorType["b", "n_ctx", "dim", float]],
        target: TensorType["b", "n_lat", "n_unit_patch", int],
        target_mask: TensorType["b", "n_lat", bool],
        extra_units_mask: Optional[TensorType["b", "n_lat", "n_unit_patch", bool]],
    ) -> Dict:
        """
        TODO update w/ eval_mode if needed
        """
        # prepare decoder input
        # TODO should be considering computing query_tokens on the fly

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

        latents = self.dropout_in(latents)
        latents = latents + self.time_emb(time_idx) + self.space_emb(space_idx)

        # TODO Check if they are the same ctx_tokens as in enc
        nb_ctx_tokens = 0
        if ctx_emb is not None:
            nb_ctx_tokens = ctx_emb.size(1)  # TODO check

            # detach context to avoid losing context calibradion from SSL
            if not self.is_ssl:
                ctx_emb = ctx_emb.detach()
            latents = torch.cat([latents, ctx_emb], dim=1)

        decoder_out = self.decoder(
            latents, mask=decoder_attn_mask, src_key_padding_mask=latent_mask
        )

        output = decoder_out[:, :-nb_ctx_tokens]
        output = self.dropout_out(output)

        if self.is_ssl:
            # compute rates
            nb_query_toknes = target.size(1)
            output = output[:, -nb_query_toknes:]
            rates = self.output_to_units(output)

            # compute loss
            loss = self.ssl_loss(rates, target)
            loss_mask = target_mask.unsqueeze(-1) & extra_units_mask
            loss = loss[loss_mask].mean()

            return {"loss": loss}

        else:
            # compute behavior
            nb_query_tokens = bhvr_tgt.size(1)
            decoder_out = decoder_out[:, -nb_query_tokens:]
            bhvr = self.out(decoder_out)

            # TODO move it before (tokenizer)
            if self.lag:
                # exclude the last N-bins
                bhvr = bhvr[:, : -self.bhvr_lag_bins]
                # add to the left N-bins to match the lag
                bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

            # Compute loss & r2
            loss_mask = loss_mask
            no_nan_mask_decoder_out = ~torch.isnan(decoder_out).any(-1)
            no_nan_mask_target = ~torch.isnan(bhvr_tgt).any(-1)
            no_nan_mask = no_nan_mask_decoder_out & no_nan_mask_target
            loss_mask = loss_mask & no_nan_mask
            bhvr_tgt = bhvr_tgt.to(bhvr.dtype)  # TODO make it cleanner
            loss = self.loss(bhvr, bhvr_tgt)
            loss = loss[loss_mask].mean()

            if self.task == "regression":
                tgt = bhvr_tgt[loss_mask].float().detach().cpu()
                pred = bhvr[loss_mask].float().detach().cpu()
                r2 = r2_score(tgt, pred, multioutput="raw_values")
                if r2.mean() < -10:
                    r2 = np.zeros_like(r2)
                return {"loss": loss, "r2": r2, "pred": bhvr}

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
            raise NotImplementedError
