import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from torchtyping import TensorType

from temporaldata import ArrayDict, Data
from torch_brain.data import pad, track_mask
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_behaviors, bin_spikes


class NDT2(nn.Module):
    def __init__(
        self,
        is_ssl: bool,
        mask_ratio: float,
        dim,
        ctx_keys: List[str],
        units_per_patch: int,
        max_bincount: int,
        spike_pad: int,
        max_time_patches: int,
        max_space_patches: int,
        bin_time: float,
        depth: int,
        heads: int,
        dropout: float,
        ffn_mult: float,
        causal: bool = True,
        activation: str = "gelu",
        pre_norm: bool = False,
        predictor_cfg: Dict = None,
        bhv_decoder_cfg: Dict = None,
    ):
        super().__init__()
        spike_embed_dim = round(dim / units_per_patch)
        self.bincount_emb = nn.Embedding(max_bincount, spike_embed_dim, padding_idx=pad)
        self.time_emb = nn.Embedding(max_time_patches, dim)
        self.space_emb = nn.Embedding(max_space_patches, dim)
        self.session_emb = InfiniteVocabEmbedding(dim)
        self.subject_emb = InfiniteVocabEmbedding(dim)
        self.task_emb = InfiniteVocabEmbedding(dim)  # more about dataset than task

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, depth)

        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

        # Decoder
        if is_ssl:
            decoder = SslDecoder(
                dim=dim,
                max_time_patches=max_time_patches,
                max_space_patches=max_space_patches,
                patch_size=units_per_patch,
                **predictor_cfg,
            )
        else:
            decoder = BhvrDecoder(
                dim=dim,
                max_time_patches=max_time_patches,
                max_space_patches=max_space_patches,
                bin_time=bin_time,
                **bhv_decoder_cfg,
            )

        self.decoder = decoder

    def forward(
        self,
        input_patch_bincount: TensorType["batch", "n_in", "patch_dim", int],
        input_time_index: TensorType["batch", "n_in", int],
        input_space_index: TensorType["batch", "n_in", int],
        input_mask: TensorType["batch", "n_in", int],
        encoder_attn_mask: TensorType["batch", "n_in", "n_in", int],
        session_index: Optional[TensorType["batch", int]],
        subject_index: Optional[TensorType["batch", int]],
        task_index: Optional[TensorType["batch", int]],
    ):
        # make input tokens
        inputs = self.bincount_emb(input_patch_bincount).flatten(-2, -1)
        inputs = self.dropout_in(inputs)
        inputs = (
            inputs + self.time_emb(input_time_index) + self.space_emb(input_space_index)
        )

        # add context tokens at the end of the sequence
        nb_ctx_tokens = 0
        ctx_tokens = []
        if session_index is not None:
            ctx_tokens.append(self.session_emb(session_index))
            nb_ctx_tokens += 1
        if subject_index is not None:
            ctx_tokens.append(self.subject_emb(subject_index))
            nb_ctx_tokens += 1
        if task_index is not None:
            ctx_tokens.append(self.subject_emb(task_index))
            nb_ctx_tokens += 1

        if nb_ctx_tokens > 0:
            ctx_emb = torch.stack([ctx_tokens], dim=1)
            inputs = torch.cat([inputs, ctx_emb], dim=1)
            input_mask = F.pad(input_mask, (0, nb_ctx_tokens), value=True)
            encoder_attn_mask = F.pad(
                encoder_attn_mask, (0, nb_ctx_tokens, 0, nb_ctx_tokens), value=True
            )

        # encoder forward pass
        latents = self.encoder(
            inputs, mask=encoder_attn_mask, src_key_padding_mask=input_mask
        )
        latents = latents[:, :-nb_ctx_tokens]
        latents = self.dropout_out(latents)

        # TODO update this
        return self.decoder(latents, context_emb, batch)


class NDT2Tokenizer:
    def __init__(
        self,
        bin_time: float,
        ctx_time: float,
        units_per_patch: int,
        pad_value: int,
        ctx_tokenizer: Dict[str, InfiniteVocabEmbedding],
        unsorted=True,
        is_ssl=True,
        bhvr_key="finger.vel",
        bhvr_dim=2,
        ibl_binning=False,
        eval=False,
    ):
        self.bin_time: float = bin_time
        self.ctx_time: float = ctx_time
        self.bin_size: int = int(np.round(ctx_time / bin_time))
        self.units_per_patch: int = units_per_patch

        def float_modulo_test(x, y, eps=1e-6):
            return np.abs(x - y * np.round(x / y)) < eps

        assert float_modulo_test(self.ctx_time, self.bin_time)

        self.pad_value: int = pad_value
        self.unsorted: bool = unsorted
        self.is_ssl: bool = is_ssl
        self.bhvr_key: str = bhvr_key
        self.ibl_binning: bool = ibl_binning
        self.bhvr_dim: int = bhvr_dim
        self.ctx_tokenizer = ctx_tokenizer
        self.session_tokenizer = None
        self.subject_tokenizer = None
        self.task_tokenizer = None
        self.eval = eval

    def __call__(self, data: Data) -> Dict:
        num_units = len(data.units.id)

        if self.unsorted:
            chan_nb_mapper = self.extract_chan_nb(data.units)
            spikes.unit_index = chan_nb_mapper.take(spikes.unit_index)
            # TODO do not work need to find an hack
            # nb_units = chan_nb_mapper.max() + 1
            num_units = 96

        binned_spikes = bin_spikes(data.spikes, num_units, self.bin_size)
        binned_spikes = np.clip(binned_spikes, 0, self.pad_value - 1)

        nb_units = binned_spikes.shape[0]
        num_spatial_patches = int(np.ceil(nb_units / self.units_per_patch))
        extra_units = num_spatial_patches * self.units_per_patch - nb_units

        if extra_units > 0:
            binned_spikes = np.pad(
                binned_spikes,
                [(0, extra_units)],
                mode="constant",
                constant_values=self.pad_value,
            )

        num_temporal_patches = binned_spikes.shape[1]

        # major hack to have time before space, as in o.g. NDT2(nb_units, time_length)
        # TODO could be mutch more cleaner
        binned_spikes = rearrange(
            binned_spikes,
            "(n pn) (t pt) -> (t n) pn pt",
            n=num_spatial_patches,
            t=num_temporal_patches,
            pn=self.units_per_patch,
            pt=1,
        )

        # time and space indices for flattened patches
        time_idx = torch.arange(num_temporal_patches, dtype=torch.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=num_spatial_patches)
        space_idx = torch.arange(num_spatial_patches, dtype=torch.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=num_temporal_patches)

        if self.mask_ratio is not None:
            keys = ["spike_tokens", "time_idx", "space_idx", "channel_counts"]
            spikes = batch["spike_tokens"]
            # TODO should be carefull here

            # TODO Check eval mode (not used for ibl)
            if self.eval:
                batch["shuffle"] = torch.arange(spikes.size(1), device=spikes.device)

                batch["encoder_frac"] = spikes.size(1)
                for k in keys:
                    batch[f"{k}_target"] = batch[k]
                return batch

            shuffle = torch.randperm(spikes.size(1), device=spikes.device)
            encoder_frac = int((1 - self.mask_ratio) * spikes.size(1))
            for k in keys:
                # applying mask at the sequence level (not batch)
                t = batch[k].transpose(1, 0)[shuffle].transpose(1, 0)

                batch[k] = t[:, :encoder_frac]
                batch[f"{k}_target"] = t[:, encoder_frac:]

            # TODO should be removed, we should have all necessary info in the batch
            batch["encoder_frac"] = encoder_frac
            batch["shuffle"] = shuffle

        batch["spike_tokens"] = rearrange(
            batch["spike_tokens"], "bs T Pn Pt -> bs T (Pn Pt)"
        )

        shape = (num_temporal_patches, num_spatial_patches)
        units_count = torch.full(shape, self.units_per_patch, dtype=torch.long)

        # last patch may have fewer units
        if num_units % num_spatial_patches != 0:
            units_count[:, -1] = self.units_per_patch - extra_units

        units_count = rearrange(
            units_count, "t n -> (t n)", n=num_spatial_patches, t=num_temporal_patches
        )

        session_idx = self.session_tokenizer(data.session.id)
        subject_idx = self.subject_tokenizer(data.subject.id)
        task_idx = self.task_tokenizer(data.id)

        batch = {
            "spike_tokens": pad(binned_spikes),
            "spike_tokens_mask": track_mask(spikes),
            "time_idx": pad(time_idx),
            "space_idx": pad(space_idx),
            "units_count": pad(units_count),
            "session_idx": session_idx,
            "subject_idx": subject_idx,
            "task_index": task_idx,
        }

        if not self.is_ssl:
            # -- Behavior
            # TODO add a callable in the config to handle this access to the bhvr data
            bhvr = getattr(data, self.bhvr_key)
            try:
                bhvr = getattr(bhvr, self.bhvr_key)
                # One hot encoding of the behavior
                bhvr = np.eye(self.bhvr_dim)[bhvr]
            except:
                pass

            # TODO should be more general
            if self.ibl_binning:
                intervals = np.c_[data.trials.start, data.trials.end]
                params = {
                    "interval_len": 2,
                    "binsize": 0.02,
                    "single_region": False,
                    "align_time": "stimOn_times",
                    "time_window": (-0.5, 1.5),
                    "fr_thresh": 0.5,
                }

                # TODO use mask_dict and refactor
                bhvr_data = getattr(data, self.bhvr_key)
                bhvr_value = bhvr_data.values

                behave_dict, mask_dict = bin_behaviors(
                    bhvr_data.timestamps,
                    bhvr_value.squeeze(),
                    intervals=intervals,
                    beh=self.bhvr_key,
                    **params,
                )
                bhvr = behave_dict[self.bhvr_key][:, None]

            batch["bhvr"] = pad(bhvr)
            batch["bhvr_mask"] = track_mask(bhvr)

        return batch

    def extract_chan_nb(self, units: ArrayDict):
        channel_names = units.channel_name
        res = [int(chan_name.split(b" ")[-1]) for chan_name in channel_names]
        return np.array(res) - 1

    def make_src_mask(
        self, times: torch.Tensor, nb_ctx_token: int, causal=True
    ) -> torch.Tensor:
        # TODO REMOVE
        cond = times[:, :, None] >= times[:, None, :]
        src_mask = torch.where(cond, 0.0, float("-inf"))

        # deal with context tokens
        src_mask = F.pad(src_mask, (0, 0, 0, nb_ctx_token), value=float("-inf"))
        src_mask = F.pad(src_mask, (0, nb_ctx_token), value=0)

        if src_mask.ndim == 3:
            src_mask = repeat(src_mask, "b t1 t2 -> (b h) t1 t2", h=self.heads)
        return src_mask

    def get_temporal_padding_mask(
        self, ref: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # TODO REMOVE
        if "shuffle" in batch:
            token_position = batch["shuffle"]
            token_position = token_position[: batch["encoder_frac"]]
        else:
            # TODO spike_tokens_mask can be returned directly
            token_position = torch.arange(ref.shape[1], device=ref.device)
        token_position = rearrange(token_position, "t -> () t")
        token_length = batch["spike_tokens_mask"].sum(1, keepdim=True)
        return token_position >= token_length


class SslDecoder(Decoder):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dropout,
        max_time_patches,
        max_space_patches,
        ffn_mult,
        patch_size,
        causal=True,
        activation="gelu",
        pre_norm=False,
    ):
        super().__init__()

        self.dim = dim
        self.neurons_per_token = patch_size[0]

        self.decoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            max_time_patches=max_time_patches,
            max_space_patches=max_space_patches,
            ffn_mult=ffn_mult,
            causal=causal,
            activation=activation,
            pre_norm=pre_norm,
        )

        self.query_token = nn.Parameter(torch.randn(dim))
        self.out = nn.Sequential(nn.Linear(dim, self.neurons_per_token))
        self.loss = nn.PoissonNLLLoss(reduction="none", log_input=True)

    def forward(
        self,
        encoder_output: torch.Tensor,
        ctx_emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        TODO update w/ eval_mode if needed
        """
        # prepare decoder input
        b, t = batch["spike_tokens_target"].shape[:2]
        decoder_query_tokens = repeat(self.query_token, "h -> b t h", b=b, t=t)
        decoder_input = torch.cat([encoder_output, decoder_query_tokens], dim=1)

        # get time, space, and context
        time = torch.cat([batch["time_idx"], batch["time_idx_target"]], 1)
        space = torch.cat([batch["space_idx"], batch["space_idx_target"]], 1)

        # get temporal padding mask
        token_position = rearrange(batch["shuffle"], "t -> () t")
        token_length = batch["spike_tokens_mask"].sum(1, keepdim=True)
        pad_mask = token_position >= token_length

        # decoder forward
        decoder_out: torch.Tensor
        decoder_out = self.decoder(decoder_input, ctx_emb, time, space, pad_mask)

        target = batch["spike_tokens_target"].squeeze(-1)

        # compute rates
        decoder_out = decoder_out[:, -target.shape[1] :]
        rates = self.out(decoder_out)

        # compute loss
        loss: torch.Tensor = self.loss(rates, target)
        loss_mask = self.get_loss_mask(batch, loss)
        loss = loss[loss_mask]
        return {"loss": loss.mean()}

    def get_loss_mask(self, batch: Dict[str, torch.Tensor], loss: torch.Tensor):
        loss_mask = torch.ones(loss.shape, device=loss.device, dtype=torch.bool)

        tmp = torch.arange(loss.shape[-1], device=loss.device)
        comparison = repeat(tmp, "c -> 1 t c", t=loss.shape[1])
        channel_mask = comparison < batch["channel_counts_target"].unsqueeze(-1)
        loss_mask = loss_mask & channel_mask

        token_position = batch["shuffle"][batch["encoder_frac"] :]
        token_position = rearrange(token_position, "t -> () t")
        token_length = batch["spike_tokens_mask"].sum(1, keepdim=True)
        length_mask = token_position < token_length

        return loss_mask & length_mask.unsqueeze(-1)


class BhvrDecoder(Decoder):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dropout,
        max_time_patches,
        max_space_patches,
        ffn_mult,
        decode_time_pool,
        behavior_dim,
        bin_time,
        behavior_lag=None,
        causal=True,
        activation="gelu",
        pre_norm=False,
        task="regression",
    ):
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.bin_time = bin_time
        self.lag = behavior_lag
        self.decode_time_pool = decode_time_pool
        self.behavior_dim = behavior_dim
        self.task = task
        if self.lag:
            self.bhvr_lag_bins = round(self.lag / bin_time)

        self.query_token = nn.Parameter(torch.randn(dim))
        self.decoder = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            max_time_patches=max_time_patches,
            max_space_patches=max_space_patches,
            ffn_mult=ffn_mult,
            causal=causal,
            activation=activation,
            pre_norm=pre_norm,
            allow_embed_padding=True,
        )
        self.out = nn.Linear(dim, self.behavior_dim)

    def forward(
        self,
        encoder_out: torch.Tensor,
        ctx_emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ):
        # prepare decoder input and temporal padding mask

        time = batch["time_idx"]
        token_length = batch["spike_tokens_mask"].sum(1, keepdim=True)
        pad_mask = self.temporal_pad_mask(encoder_out, token_length)
        encoder_out, pad_mask = self.temporal_pool(time, encoder_out, pad_mask)

        bhvr_tgt = batch["bhvr"]
        bhvr_length = batch["bhvr_mask"].sum(1, keepdim=True)
        decoder_in, pad_mask = self.prepare_decoder_input(
            bhvr_tgt, encoder_out, pad_mask, bhvr_length
        )

        # get time, space
        time, space = self.get_time_space(encoder_out, bhvr_tgt)

        # decoder forward
        decoder_out: torch.Tensor
        # detach context to avoid gradient flow and lose context calibradion from SSL
        ctx_emb = ctx_emb.detach()

        decoder_out = self.decoder(decoder_in, ctx_emb, time, space, pad_mask)

        # compute behavior
        nb_injected_tokens = bhvr_tgt.shape[1]
        decoder_out = decoder_out[:, -nb_injected_tokens:]
        bhvr = self.get_bhvr(decoder_out)

        # Compute loss & r2
        length_mask = self.get_length_mask(decoder_out, bhvr_tgt, token_length)
        bhvr_tgt = bhvr_tgt.to(bhvr.dtype)  # TODO make it cleanner
        loss = self.loss(bhvr, bhvr_tgt, length_mask)

        if self.task == "regression":
            tgt = bhvr_tgt[length_mask].float().detach().cpu()
            pred = bhvr[length_mask].float().detach().cpu()
            r2 = r2_score(tgt, pred, multioutput="raw_values")
            if r2.mean() < -10:
                r2 = np.zeros_like(r2)
            return {"loss": loss, "r2": r2, "pred": bhvr}

        elif self.task == "classification":
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
        else:
            raise NotImplementedError

    def temporal_pad_mask(
        self, ref: torch.Tensor, max_lenght: torch.Tensor
    ) -> torch.Tensor:
        token_position = torch.arange(ref.shape[1], device=ref.device)
        token_position = rearrange(token_position, "t -> () t")
        return token_position >= max_lenght

    def temporal_pool(
        self,
        times: torch.Tensor,
        encoder_out: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, nb_tokens, h = encoder_out.shape
        b = encoder_out.shape[0]
        t = times.max() + 1
        h = encoder_out.shape[-1]
        dev = encoder_out.device
        pool = self.decode_time_pool

        # t + 1 for padding
        pooled_features = torch.zeros(b, t + 1, h, device=dev, dtype=encoder_out.dtype)

        time_with_pad_marked = torch.where(pad_mask, t, times)
        index = repeat(time_with_pad_marked, "b t -> b t h", h=h).to(torch.long)
        pooled_features = pooled_features.scatter_reduce(
            src=encoder_out, dim=1, index=index, reduce=pool, include_self=False
        )
        encoder_out = pooled_features[:, :-1]  # remove padding

        nb_tokens = encoder_out.shape[1]
        new_pad_mask = torch.ones(b, nb_tokens, dtype=bool, device=dev).float()
        src = torch.zeros_like(times).float()

        times = times.to(torch.long)
        new_pad_mask = new_pad_mask.scatter_reduce(
            src=src, dim=1, index=times, reduce="prod", include_self=False
        ).bool()

        return encoder_out, new_pad_mask

    def prepare_decoder_input(
        self,
        bhvr: torch.Tensor,
        encoder_out: torch.Tensor,
        pad_mask: torch.Tensor,
        max_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t = bhvr.shape[:2]
        query_tokens = repeat(self.query_token, "h -> b t h", b=b, t=t)
        if encoder_out.shape[1] < t:
            to_add = t - encoder_out.shape[1]
            encoder_out = F.pad(encoder_out, (0, 0, 0, to_add), value=0)
        decoder_in = torch.cat([encoder_out, query_tokens], dim=1)

        if encoder_out.shape[1] < t:
            to_add = t - pad_mask.shape[1]
            pad_mask = F.pad(pad_mask, (0, to_add), value=True)
        query_pad_mask = self.temporal_pad_mask(query_tokens, max_length)
        pad_mask = torch.cat([pad_mask, query_pad_mask], dim=1)

        return decoder_in, pad_mask

    def get_time_space(
        self, encoder_out: torch.Tensor, bhvr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t_enc = encoder_out.size()[:2]
        dev = encoder_out.device
        time = repeat(torch.arange(t_enc, device=dev), "t -> b t", b=b)

        if self.task == "classification":
            query_time = repeat(torch.tensor([t_enc], device=dev), "t -> b t", b=b)
        else:
            t = bhvr.shape[1]
            query_time = repeat(torch.arange(t, device=dev), "t -> b t", b=b)
        if self.causal and self.lag:
            # allow looking N-bins of neural data into the "future";
            # we back-shift during the actual decode comparison.
            query_time = time + self.bhvr_lag_bins
        time = torch.cat([time, query_time], dim=1)

        # Do use space for this decoder
        space = torch.zeros_like(time)

        return time, space

    def get_bhvr(self, decoder_out: torch.Tensor) -> torch.Tensor:
        bhvr = self.out(decoder_out)

        if self.lag:
            # exclude the last N-bins
            bhvr = bhvr[:, : -self.bhvr_lag_bins]
            # add to the left N-bins to match the lag
            bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)
        return bhvr

    def get_length_mask(
        self,
        decoder_out: torch.Tensor,
        bhvr_tgt: torch.Tensor,
        max_length: torch.Tensor,
    ) -> torch.Tensor:
        length_mask = ~self.temporal_pad_mask(decoder_out, max_length)
        no_nan_mask = ~torch.isnan(decoder_out).any(-1) & ~torch.isnan(bhvr_tgt).any(-1)
        length_mask = length_mask & no_nan_mask
        if self.lag:
            length_mask[:, : self.bhvr_lag_bins] = False

        return length_mask

    def loss(
        self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor, length_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.task == "regression":
            loss = F.mse_loss(bhvr, bhvr_tgt, reduction="none")
            return loss[length_mask].mean()
        elif self.task == "classification":
            loss = F.binary_cross_entropy_with_logits(bhvr, bhvr_tgt, reduction="none")
            return loss[length_mask].mean()
        else:
            raise NotImplementedError
