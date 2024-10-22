import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from torch import Tensor

from torch_brain.nn import InfiniteVocabEmbedding


class NDT2_MAE_MaskManager(nn.Module):
    def __init__(self, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(
        self, batch: Dict[str, torch.Tensor], eval_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        # -- Mask
        if self.mask_ratio is None:
            return batch

        spikes = batch["spike_tokens"]
        if eval_mode:
            batch["shuffle"] = torch.arange(spikes.size(1), device=spikes.device)
            batch["encoder_frac"] = spikes.size(1)
            for key in ["spike_tokens", "time_idx", "space_idx"]:
                batch[f"{key}_target"] = batch[key]
            return batch

        shuffle = torch.randperm(spikes.size(1), device=spikes.device)
        encoder_frac = int((1 - self.mask_ratio) * spikes.size(1))
        for key in ["spike_tokens", "time_idx", "space_idx"]:
            t = batch[key].transpose(1, 0)[shuffle].transpose(1, 0)

            batch[key] = t[:, :encoder_frac]
            batch[f"{key}_target"] = t[:, encoder_frac:]
        batch["encoder_frac"] = encoder_frac
        batch["shuffle"] = shuffle

        return batch


class NDT2_SpikesPatchifier(nn.Module):
    # TODO transfer at the cfg file
    def __init__(self, dim, patch_size=(32, 1), max_neuron_count=21, pad=5):
        """
        Args:
            dim (Int): Dimension of the output patchified tensor
            patch_size (Tuple[Int, Int]): (num_neurons, num_time_bins)
            max_time_patches (Int): Maximum number of time patches
            max_space_patches (Int): Maximum number of space patches
        """
        super().__init__()
        spike_embed_dim = round(dim / patch_size[0])
        self.readin = nn.Embedding(max_neuron_count, spike_embed_dim, padding_idx=pad)

    def forward(self, spikes):
        """
        Args:
            x (torch.Tensor): Binned spikes (bs, T, patch_size[0], patch_size[1])
        Returns: (NxT, D)
        """
        x = rearrange(spikes, "bs T Pn Pt -> bs T (Pn Pt)")
        x = self.readin(x).flatten(-2, -1)
        return x


class NDT2_ContextManager(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_keys: Optional[List[str]] = ["session", "subject"],
    ):
        super().__init__()
        self.keys = ctx_keys
        for k in self.keys:
            setattr(self, f"{k}_emb", InfiniteVocabEmbedding(dim, init_scale=1.0))
            setattr(self, f"{k}_flag", nn.Parameter(torch.randn(dim) / math.sqrt(dim)))

    def get_ctx(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch.get(f"{k}_idx") for k in self.keys}

    def get_emb(self, key: str, token: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"{key}_emb")(token) + getattr(self, f"{key}_flag")

    def init_vocab(self, vocab: Dict[str, List[str]]):
        for k, ids in vocab.items():
            getattr(self, f"{k}_emb").initialize_vocab(ids)

    def extend_vocab(self, vocab: Dict[str, List[str]]):
        for k, ids in vocab.items():
            getattr(self, f"{k}_emb").extend_vocab(ids, exist_ok=True)

    def forward(
        self, batch: Dict[str, torch.Tensor], type: torch.Tensor
    ) -> torch.Tensor:
        ctx_emb = [
            self.get_emb(ctx_key, ctx_token).to(dtype=type)
            for ctx_key, ctx_token in self.get_ctx(batch).items()
        ]
        return torch.stack(ctx_emb, dim=1)


class NDT2_PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_time_patches: int,
        max_space_patches: int,
        allow_embed_padding: bool,
    ):
        # TODO make max_time_patches and max_space_patches in cfg
        super().__init__()
        if allow_embed_padding:
            self.time_emb = nn.Embedding(
                max_time_patches + 1, dim, padding_idx=max_time_patches
            )
            self.space_emb = nn.Embedding(
                max_space_patches + 1, dim, padding_idx=max_space_patches
            )
        else:
            self.time_emb = nn.Embedding(max_time_patches, dim)
            self.space_emb = nn.Embedding(max_space_patches, dim)

    def forward(self, times: torch.Tensor, spaces: torch.Tensor) -> torch.Tensor:
        return self.time_emb(times) + self.space_emb(spaces)


class NDT2_Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dropout,
        max_time_patches,
        max_space_patches,
        in_encoder=True,
        ffn_mult=1,
        causal=True,
        activation="gelu",
        pre_norm=False,
        allow_embed_padding=False,
    ):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            depth (Int): Number of Attention layers
            heads (Int): Number of heads for MHA
            inter_dim (Int): Dimension of the intermediate MLP layers
            dropout (Float): Dropout rate in Attention layers
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

        self.max_space_patches = max_space_patches
        self.max_time_patches = max_time_patches

        self.activation = activation
        self.pre_norm = pre_norm
        self.ffn_mult = ffn_mult
        self.causal = causal

        self.in_encoder = in_encoder

        enc_layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=int(dim * ffn_mult),
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=pre_norm,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, depth)

        self.positional_encoding = NDT2_PositionalEncoding(
            dim, max_time_patches, max_space_patches, allow_embed_padding
        )

        self.dropout_in = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        ctx_emb: torch.Tensor,
        times: torch.Tensor,
        spaces: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        src = self.dropout_in(src)
        src = src + self.positional_encoding(times, spaces)

        try:
            nb_ctx_token = ctx_emb[0].shape[1]
            src = torch.cat([src, ctx_emb], dim=1)
            pad_mask = F.pad(pad_mask, (0, nb_ctx_token), value=False)
        except:
            nb_ctx_token = 0

        src_mask = self.make_src_mask(times, nb_ctx_token)

        # encoder forward
        out = self.transformer_encoder(src, src_mask, src_key_padding_mask=pad_mask)

        encoder_out = out[:, :-nb_ctx_token]
        return self.dropout_out(encoder_out)

    def make_src_mask(
        self, times: torch.Tensor, nb_ctx_token: int, causal=True
    ) -> torch.Tensor:
        # TODO update if casusal is False
        cond = times[:, :, None] >= times[:, None, :]
        src_mask = torch.where(cond, 0.0, float("-inf"))

        # deal with context tokens
        src_mask = F.pad(src_mask, (0, 0, 0, nb_ctx_token), value=float("-inf"))
        src_mask = F.pad(src_mask, (0, nb_ctx_token), value=0)

        # TODO check if this is needed
        # expand along heads
        if src_mask.ndim == 3:
            src_mask = repeat(src_mask, "b t1 t2 -> (b h) t1 t2", h=self.heads)
        return src_mask

    def encode(
        self,
        encoder_input: torch.Tensor,
        ctx_emb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        time = batch["time_idx"]
        space = batch["space_idx"]
        pad_mask = self.get_temporal_padding_mask(encoder_input, batch)
        return self(encoder_input, ctx_emb, time, space, pad_mask)

    def get_temporal_padding_mask(
        self, ref: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if "shuffle" in batch:
            token_position = batch["shuffle"]
            token_position = token_position[: batch["encoder_frac"]]
        else:
            token_position = torch.arange(ref.shape[1], device=ref.device)
        token_position = rearrange(token_position, "t -> () t")
        return token_position >= rearrange(batch["token_length"], "b -> b ()")


class NDT2_Decoder(nn.Module):
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

        self.decoder = NDT2_Transformer(
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
            in_encoder=False,
        )

        self.mask_token = nn.Parameter(torch.randn(dim))
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
        decoder_mask_tokens = repeat(self.mask_token, "h -> b t h", b=b, t=t)
        decoder_input = torch.cat([encoder_output, decoder_mask_tokens], dim=1)

        # get time, space, and context
        time = torch.cat([batch["time_idx"], batch["time_idx_target"]], 1)
        space = torch.cat([batch["space_idx"], batch["space_idx_target"]], 1)

        # get temporal padding mask
        token_position = rearrange(batch["shuffle"], "t -> () t")
        pad_mask = token_position >= rearrange(batch["token_length"], "b -> b ()")

        # decoder forward
        decoder_out: torch.Tensor
        decoder_out = self.decoder(decoder_input, ctx_emb, time, space, pad_mask)

        target = batch["spike_tokens_target"].squeeze(-1)

        # compute rates
        reps = reps[:, -target.size(1) :]
        rates = self.out(decoder_out)

        # compute loss
        loss: torch.Tensor = self.loss(rates, target)  # b t' c
        loss_mask = self.get_loss_mask(batch, loss)
        loss = loss[loss_mask]
        return loss.mean()

    def get_loss_mask(self, batch: Dict[str, torch.Tensor], loss: torch.Tensor):
        loss_mask = torch.ones(loss.size(), device=loss.device, dtype=torch.bool)
        token_position = batch["shuffle"][batch["encoder_frac"] :]
        token_position = rearrange(token_position, "t -> () t")
        length_mask = token_position < rearrange(batch["token_length"], "b -> b ()")

        return loss_mask & length_mask.unsqueeze(-1)


import numpy as np

# from sklearn.metrics import r2_score
from torchmetrics import R2Score


class NDT2_bhvr_Decoder(nn.Module):
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
        behavior_lag,
        causal=True,
        activation="gelu",
        pre_norm=False,
        behavior_lag_lookahead=True,
    ):
        super().__init__()
        self.dim = dim
        self.causal = causal
        self.bin_time = bin_time
        self.behavior_lag = behavior_lag
        self.bhvr_lag_bins = round(behavior_lag / bin_time)
        self.decode_time_pool = decode_time_pool
        self.behavior_dim = behavior_dim
        self.behavior_lag_lookahead = behavior_lag_lookahead

        self.cls_token = nn.Parameter(torch.randn(dim))
        self.decoder = NDT2_Transformer(
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
            in_encoder=False,
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
        bhvr_tgt = batch["bhvr_vel"]
        time = batch["time_idx"]

        pad_mask = self.temporal_pad_mask(encoder_out, batch["token_length"])
        encoder_out, pad_mask = self.temporal_pool(time, encoder_out, pad_mask)
        decoder_in, pad_mask = self.prepare_decoder_input(
            bhvr_tgt, encoder_out, pad_mask, batch["bhvr_length"]
        )

        # get time, space
        time, space = self.get_time_space(bhvr_tgt)

        # decoder forward
        decoder_out: torch.Tensor
        decoder_out = self.decoder(decoder_in, ctx_emb, time, space, pad_mask)

        # compute behavior
        nb_injected_tokens = bhvr_tgt.shape[1]
        decoder_out = decoder_out[:, -nb_injected_tokens:]
        bhvr = self.get_bhvr(decoder_out)

        # Compute loss & r2
        length_mask = self.get_length_mask(decoder_out, bhvr_tgt, batch["token_length"])
        loss = self.loss(bhvr, bhvr_tgt, length_mask)
        r2 = self.r2(bhvr, bhvr_tgt, length_mask)
        return loss, r2

    def temporal_pad_mask(
        self, ref: torch.Tensor, max_lenght: torch.Tensor
    ) -> torch.Tensor:
        token_position = torch.arange(ref.shape[1], device=ref.device)
        token_position = rearrange(token_position, "t -> () t")
        return token_position >= rearrange(max_lenght, "b -> b ()")

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
        bhvr_vel: torch.Tensor,
        encoder_out: torch.Tensor,
        pad_mask: torch.Tensor,
        max_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t = bhvr_vel.size()[:2]
        extra_tokens = repeat(self.cls_token, "h -> b t h", b=b, t=t)
        if encoder_out.shape[1] < t:
            to_add = t - encoder_out.shape[1]
            encoder_out = F.pad(encoder_out, (0, 0, 0, to_add), value=0)
        decoder_in = torch.cat([encoder_out, extra_tokens], dim=1)

        if encoder_out.shape[1] < t:
            to_add = t - pad_mask.shape[1]
            pad_mask = F.pad(pad_mask, (0, to_add), value=True)
        extra_pad_mask = self.temporal_pad_mask(extra_tokens, max_length)
        pad_mask = torch.cat([pad_mask, extra_pad_mask], dim=1)

        return decoder_in, pad_mask

    def get_time_space(
        self,
        bhvr_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t = bhvr_vel.size()[:2]
        dev = bhvr_vel.device
        injected_time = repeat(torch.arange(t, device=dev), "t -> b t", b=b)
        decode_time = injected_time
        if self.causal and self.behavior_lag_lookahead:
            # allow looking N-bins of neural data into the "future"; we back-shift during the actual decode comparison.
            decode_time = injected_time + self.bhvr_lag_bins
        time = torch.cat([injected_time, decode_time], dim=1)
        # Do use space for this decoder
        injected_space = torch.zeros((b, t), device=dev, dtype=torch.long)
        space = torch.cat([injected_space, injected_space], dim=1)

        return time, space

    def get_bhvr(self, decoder_out: torch.Tensor) -> torch.Tensor:
        bhvr = self.out(decoder_out)

        if self.bhvr_lag_bins:
            bhvr = bhvr[:, : -self.bhvr_lag_bins]
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
        length_mask[:, : self.bhvr_lag_bins] = False
        return length_mask

    def loss(
        self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor, length_mask: torch.Tensor
    ) -> torch.Tensor:
        loss = F.mse_loss(bhvr, bhvr_tgt, reduction="none")
        return loss[length_mask].mean()

    def r2(
        self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor, length_mask: torch.Tensor
    ) -> np.ndarray:
        tgt = bhvr_tgt[length_mask].float().detach().cpu()
        bhvr = bhvr[length_mask].float().detach().cpu()
        # o.g. r2 is computed with sklearn.metrics.r2_score
        r2_score = R2Score(multioutput="raw_values")
        r2 = r2_score(bhvr, tgt)
        if r2.mean() < -10:
            r2 = np.zeros_like(r2)
        return r2


# params = list(self.named_parameters())
# # As of 2/24/23 all my parameters are named, this better stay the case
# accel_flag = lambda name: name in self.novel_params or (
#     "session_embed" in name
#     or "subject_embed" in name
#     or "task_embed" in name
#     or "array_embed" in name
# )
# grouped_params = [
#     {
#         "params": [p for n, p in params if accel_flag(n)],
#         "lr": self.cfg.lr_init * self.cfg.accelerate_new_params,
#     },
#     {
#         "params": [p for n, p in params if not accel_flag(n)],
#         "lr": self.cfg.lr_init,
#     },
# ]
