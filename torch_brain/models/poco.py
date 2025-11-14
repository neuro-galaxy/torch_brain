# Standalone implementation of POCO (Population-Conditioned Forecaster)
# Needs torch, numpy, einops, and xformers
import torch
import torch.nn as nn
import numpy as np
import itertools
import torch.nn.functional as F
import logging
from torchtyping import TensorType
from einops import repeat, rearrange
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import xformers.ops as xops
    except ImportError:
        xops = None
from typing import Union, Optional
from torch_brain.nn import (
    Embedding,
    RotaryCrossAttention,
    RotarySelfAttention,
)

from poyo import POYO
from torch_brain.registry import ModalitySpec


# RotaryCrossAttention poco ROTARARY ATTENTION is different with torch brain by adding the memory efficient module to accelerate xformer. maybe we modify torch brain later
class RotaryEmbedding(nn.Module):
    r"""Custom rotary positional embedding layer. This function generates sinusoids of
    different frequencies, which are then used to modulate the input data. Half of the
    dimensions are not rotated.

    The frequencies are computed as follows:

    .. math::
        f(i) = \frac{2\pi}{t_{\min}} \cdot \frac{t_{\max}}{t_\{min}}^{2i/dim}}

    To rotate the input data, use :func:`apply_rotary_pos_emb`.

    Args:
        dim (int): Dimensionality of the input data.
        t_min (float, optional): Minimum period of the sinusoids.
        t_max (float, optional): Maximum period of the sinusoids.
    """

    def __init__(self, dim, t_min=1e-4, t_max=4.0):
        super().__init__()
        inv_freq = torch.zeros(dim // 2)
        inv_freq[: dim // 4] = (
            2
            * torch.pi
            / (
                t_min
                * (
                    (t_max / t_min)
                    ** (torch.arange(0, dim // 2, 2).float() / (dim // 2))
                )
            )
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, timestamps):
        r"""Computes the rotation matrices for given timestamps.

        Args:
            timestamps (torch.Tensor): timestamps tensor.
        """
        freqs = torch.einsum("..., f -> ... f", timestamps, self.inv_freq)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs


class NeuralPredictionConfig:

    def __init__(self):
        self.seq_length = 64  # total length
        self.pred_length = 16  # prediction length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.compression_factor = 16  # 16 steps per token
        self.decoder_type = "POYO"
        self.conditioning = "mlp"
        self.conditioning_dim = 1024
        self.decoder_context_length = None

        self.poyo_num_latents = 8
        self.latent_session_embedding = False
        self.unit_embedding_components = [
            "session",
        ]  # embeddings that will in added on top of unit embedding
        self.decoder_num_layers = 1
        self.decoder_num_heads = 16
        self.poyo_unit_dropout = 0
        self.rotary_attention_tmax = 100
        self.decoder_hidden_size = 128

        self.freeze_backbone = False
        self.freeze_conditioned_net = False


from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class PerceiverRotary(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        context_dim=None,
        dim_head=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        use_memory_efficient_attn=True,
        t_max=120.0,
    ):
        super().__init__()

        self.rotary_emb = RotaryEmbedding(dim_head, t_min=t_max / 1000, t_max=t_max)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            # use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing transfomers (qkv-latent)
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                            # use_memory_efficient_attn=use_memory_efficient_attn,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
            # use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim
        # self.using_memory_efficient_attn = self.enc_atn.using_memory_efficient_attn

    def forward(
        self,
        *,  # (   padded   ) or (   chained   )
        inputs,  # (B, N_in, dim) or (N_all_in, dim)
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries,  # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps,  # (B, N_in) or (N_all_in,)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        input_mask=None,  # (B, N_in) or None
        input_seqlen=None,  # None or (B,)
        latent_seqlen=None,  # None or (B,)
        output_query_seqlen=None,  # None or (B,)
        output_latent=False,
    ) -> Union[
        TensorType["batch", "*nqueries", "dim"],  # if padded
        TensorType["ntotal_queries", "dim"],  # if chained
    ]:

        # Make sure the arguments make sense
        padded_input = input_mask is not None
        chained_input = (
            input_seqlen is not None
            or latent_seqlen is not None
            or output_query_seqlen is not None
        )

        if padded_input and chained_input:
            raise ValueError(
                f"Cannot specify both input_mask and "
                f"input_seqlen/latent_seqlen/output_query_seqlen."
            )

        if chained_input:
            if (
                input_seqlen is None
                or latent_seqlen is None
                or output_query_seqlen is None
            ):
                raise ValueError(
                    f"Must specify all of input_seqlen, latent_seqlen, "
                    f"output_query_seqlen."
                )

        if padded_input:
            assert inputs.dim() == 3
            assert latents.dim() == 3
            assert output_queries.dim() == 3
            assert input_timestamps.dim() == 2
            assert latent_timestamps.dim() == 2
            assert output_query_timestamps.dim() == 2
            assert input_mask.dim() == 2

        if chained_input:
            assert inputs.dim() == 2
            assert latents.dim() == 2
            assert output_queries.dim() == 2
            assert input_timestamps.dim() == 1
            assert latent_timestamps.dim() == 1
            assert output_query_timestamps.dim() == 1
            assert input_seqlen.dim() == 1
            assert latent_seqlen.dim() == 1
            assert output_query_seqlen.dim() == 1

        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            context_mask=input_mask,  # used if default attention
            query_seqlen=latent_seqlen,  # used if memory efficient attention
            context_seqlen=input_seqlen,  # used if memory efficient attention
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(latents, latent_timestamp_emb, x_seqlen=latent_seqlen)
            )
            latents = latents + self.dropout(self_ff(latents))

        if output_latent:
            if latents.dim() == 2:
                latents = latents.reshape(latent_seqlen.shape[0], -1, latents.shape[1])
            return latents

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            context_mask=None,
            query_seqlen=output_query_seqlen,
            context_seqlen=latent_seqlen,
        )
        output_queries = output_queries + self.dec_ffn(output_queries)

        return output_queries


class POYO(POYO):

    def __init__(
        self,
        *,
        input_dim=1,
        sequence_length=64,
        readout_spec: ModalitySpec,
        latent_step=1,
        dim: int = 512,
        depth: int = 2,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        dim_head=64,
        num_latents=64,
        emb_init_scale=0.02,
        use_memory_efficient_attn=True,
        input_size=None,
        query_length=1,
        T_step=1,
        unit_dropout=0.0,
        output_latent=False,  # if True, return the latent representation, else return the query representation
        t_max=100.0,
        num_datasets=1,
        unit_embedding_components=[
            "session",
        ],
        latent_session_embedding=False,
    ):
        # Pass required arguments to parent POYO class
        super().__init__(
            sequence_length=sequence_length,
            readout_spec=readout_spec,
            latent_step=latent_step,
            num_latents_per_step=num_latents,
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            emb_init_scale=emb_init_scale,
            t_max=t_max,
        )

        self.input_proj = nn.Linear(input_dim, dim)
        self.unit_emb = Embedding(sum(input_size), dim, init_scale=emb_init_scale)
        self.session_emb = Embedding(len(input_size), dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.dataset_emb = Embedding(num_datasets, dim, init_scale=emb_init_scale)

        self.num_latents = num_latents
        self.query_length = query_length
        self.unit_dropout = unit_dropout

        self.perceiver_io = PerceiverRotary(
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            # use_memory_efficient_attn=use_memory_efficient_attn,
            t_max=t_max,
        )

        self.dim = dim
        self.T_step = T_step
        # self.using_memory_efficient_attn = self.perceiver_io.using_memory_efficient_attn
        self.output_latent = output_latent
        self.unit_embedding_components = unit_embedding_components
        self.latent_session_embedding = latent_session_embedding

    def forward(
        self,
        # input sequence
        x: torch.Tensor,  # sum(B * D), L, input_dim
        unit_indices,  # sum(B * D)
        unit_timestamps,  # sum(B * D), L
        input_seqlen,  # (B, )
        # output sequence
        session_index,  # (B, )
        dataset_index,  # (B, )
    ):

        # input
        L = x.shape[1]
        B = input_seqlen.shape[0]
        T = L * self.T_step
        unit_embedding = self.unit_emb(unit_indices)

        session_indices = torch.concatenate(
            [
                torch.full((input_seqlen[i],), session_index[i], device=x.device)
                for i in range(B)
            ]
        )  # sum(B * D)
        dataset_indices = torch.concatenate(
            [
                torch.full((input_seqlen[i],), dataset_index[i], device=x.device)
                for i in range(B)
            ]
        )  # sum(B * D)

        if "session" in self.unit_embedding_components:
            unit_embedding = (
                unit_embedding
                + self.session_emb(session_indices)
                + self.dataset_emb(dataset_indices)
            )  # sum(B * D), dim

        inputs = unit_embedding.unsqueeze(1) + self.input_proj(x)  # sum(B * D), L, dim
        unit_timestamps = unit_timestamps.reshape(-1)
        inputs = inputs.reshape(-1, inputs.shape[2])

        # latents
        latent_index = torch.arange(self.num_latents, device=x.device)
        latents = self.latent_emb(latent_index)
        latents = latents.repeat(B, 1, 1)  # B, N_latent, dim
        if self.latent_session_embedding:
            latents = (
                latents
                + self.session_emb(session_index).unsqueeze(1)
                + self.dataset_emb(dataset_index).unsqueeze(1)
            )  # B, N_latent, dim
        latents = latents.reshape(-1, latents.shape[2])
        latent_seqlen = torch.full((B,), self.num_latents, device=x.device)
        latent_timestamps = torch.arange(
            0, T, step=T / self.num_latents, device=x.device
        )
        latent_timestamps = latent_timestamps.repeat(B)  # B * N_latent

        # outputs
        output_queries = unit_embedding
        sumD = output_queries.shape[0]
        output_queries = output_queries.repeat_interleave(
            self.query_length, dim=0
        )  # sum(B * D * q_len), dim
        output_timestamps = (
            torch.arange(self.query_length, device=x.device).repeat(sumD) + T
        )  # sum(B * D * q_len)

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=unit_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=None,
            input_seqlen=input_seqlen * L,
            latent_seqlen=latent_seqlen,
            output_query_seqlen=input_seqlen,
            output_latent=self.output_latent,
        )

        return output_latents

    def reset_for_finetuning(self):
        self.unit_emb.reset_parameters()
        self.session_emb.reset_parameters()

    def embedding_requires_grad(self, requires_grad=True):
        self.unit_emb.requires_grad_(requires_grad)
        self.session_emb.requires_grad_(requires_grad)
        # self.latent_emb.requires_grad_(requires_grad)


class POCO(nn.Module):

    def __init__(self, config: NeuralPredictionConfig, input_size):
        super().__init__()

        self.Tin = config.seq_length - config.pred_length
        self.dataset_idx = []  # the dataset index for each session
        for i_dataset, size in enumerate(input_size):
            self.dataset_idx += [i_dataset] * len(size)
        self.num_datasets = len(input_size)
        input_size = list(itertools.chain(*input_size))
        self.tokenizer = None
        self.tokenizer_type = "none"
        self.token_dim = config.compression_factor
        self.T_step = config.compression_factor

        self.input_size = input_size
        self.pred_step = config.pred_length

        from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY
        from torch_brain.nn.loss import MSELoss

        # Register the modality (returns an ID, but we need the ModalitySpec)
        # If it's already registered, just get it from the registry
        modality_name = "poco_zapbench"
        if modality_name not in MODALITY_REGISTRY:
            register_modality(
                modality_name,
                dim=1,
                type=DataType.CONTINUOUS,
                timestamp_key="poco.timestamps",
                value_key="poco.location",
                loss_fn=MSELoss(),
            )
        # Get the ModalitySpec from the registry
        readout_spec = MODALITY_REGISTRY[modality_name]
        assert config.decoder_type == "POYO"
        self.decoder = POYO(
            sequence_length=1.0,  # maybe the self.Tin not sure
            latent_step=1.0 / 8,
            readout_spec=readout_spec,
            input_dim=self.token_dim,
            dim=config.decoder_hidden_size,
            depth=config.decoder_num_layers,
            self_heads=config.decoder_num_heads,
            input_size=input_size,
            num_latents=config.poyo_num_latents,
            T_step=self.T_step,
            unit_dropout=config.poyo_unit_dropout,
            output_latent=False,
            t_max=float(config.rotary_attention_tmax),
            num_datasets=self.num_datasets,
            unit_embedding_components=config.unit_embedding_components,
            latent_session_embedding=config.latent_session_embedding,
        )
        self.embedding_dim = config.decoder_hidden_size
        self.linear_out_size = config.pred_length

        self.conditioning = config.conditioning
        self.conditioning_dim = config.conditioning_dim

        assert config.conditioning == "mlp"
        self.in_proj = nn.Sequential(
            nn.Linear(self.Tin, config.conditioning_dim), nn.ReLU()
        )

        self.conditioning_alpha = nn.Linear(self.embedding_dim, config.conditioning_dim)
        self.conditioning_beta = nn.Linear(self.embedding_dim, config.conditioning_dim)

        # init as zeros
        self.conditioning_alpha.weight.data.zero_()
        self.conditioning_alpha.bias.data.zero_()
        self.conditioning_beta.weight.data.zero_()
        self.conditioning_beta.bias.data.zero_()

        self.out_proj = nn.Linear(config.conditioning_dim, self.linear_out_size)

        if config.decoder_context_length is not None:
            self.Tin = config.decoder_context_length
        self.n_tokens = self.Tin // config.compression_factor

        # freeze parts of the model for finetuning
        if config.freeze_backbone:
            for param in self.decoder.parameters():
                param.requires_grad = False
            if config.decoder_type == "POYO":
                # allow all embedding layers to be trained
                self.decoder.embedding_requires_grad(True)

        if config.freeze_conditioned_net:
            assert (
                config.conditioning == "mlp"
            ), "Only support freezing conditioned net for MLP conditioning"
            self.in_proj.requires_grad_(False)
            self.conditioning_alpha.requires_grad_(False)
            self.conditioning_beta.requires_grad_(False)
            self.out_proj.requires_grad_(False)

    def forward(self, x_list, unit_indices=None, unit_timestamps=None):
        """
        x: list of tensors, each of shape (L, B, D)
        return: list of tensors, each of shape (pred_length, B, D)
        """

        bsz = [x.size(1) for x in x_list]
        L = x_list[0].size(0)
        pred_step = self.pred_step
        x = torch.concatenate(
            [x.permute(1, 2, 0).reshape(-1, L) for x in x_list], dim=0
        )  # sum(B * D), L

        # only use the last Tin steps
        if L != self.Tin:
            x = x[:, -self.Tin :]

        # Tokenize the input sequence
        if self.tokenizer_type == "vqvae":
            with torch.no_grad():
                out = self.tokenizer.encode(x)  # out: sum(B * D), TC, E
        elif self.tokenizer_type == "cnn":
            out = x.unsqueeze(1)
            out = self.tokenizer(out)  # out: sum(B * D), C, TC
            out = out.permute(0, 2, 1)  # out: sum(B * D), TC, C
        elif self.tokenizer_type == "none":
            out = x.reshape(
                x.shape[0], self.Tin // self.T_step, self.T_step
            )  # out: sum(B * D), TC, E
        else:
            raise ValueError(f"Unknown tokenizer type {self.tokenizer_type}")
        d_list = self.input_size

        if unit_indices is None:
            sum_channels = 0
            unit_indices = []
            for b, d in zip(bsz, self.input_size):
                indices = (
                    torch.arange(d, device=x.device)
                    .unsqueeze(0)
                    .repeat(b, 1)
                    .reshape(-1)
                )  # B * D
                unit_indices.append(indices + sum_channels)
                sum_channels += d
            unit_indices = torch.cat(unit_indices, dim=0)  # sum(B * D)
        if unit_timestamps is None:
            unit_timestamps = torch.zeros_like(unit_indices).unsqueeze(
                1
            ) + torch.arange(
                0, self.Tin, self.T_step, device=x.device
            )  # sum(B * D), TC

        input_seqlen = torch.cat(
            [
                torch.full((b,), d, device=x.device)
                for b, d in zip(bsz, self.input_size)
            ],
            dim=0,
        )
        session_index = torch.cat(
            [torch.full((b,), i, device=x.device) for i, b in enumerate(bsz)], dim=0
        )
        dataset_index = torch.cat(
            [
                torch.full((b,), self.dataset_idx[i], device=x.device)
                for i, b in enumerate(bsz)
            ],
            dim=0,
        )

        embed = self.decoder(
            out,
            unit_indices=unit_indices,
            unit_timestamps=unit_timestamps,
            input_seqlen=input_seqlen,
            session_index=session_index,
            dataset_index=dataset_index,
        )  # sum(B * D), embedding_dim; or sum(B * D), pred_length, embedding_dim

        # partition embed to a list of tensors, each of shape (B, D, embedding_dim)
        split_size = [b * d for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [
            xx.reshape(b, d, self.embedding_dim) for xx, b, d in zip(embed, bsz, d_list)
        ]  # (B, D, E)

        preds = []
        for i, (e, d, input) in enumerate(zip(embed, self.input_size, x_list)):
            alpha = self.conditioning_alpha(e)  # B, D, cond_dim
            beta = self.conditioning_beta(e)  # B, D, cond_dim
            input = input.permute(1, 2, 0)  # B, D, L
            weights = self.in_proj(input) * alpha + beta  # B, D, cond_dim
            pred = self.out_proj(weights)  # B, D, pred_length
            preds.append(pred.permute(2, 0, 1))

        return preds

    def load_pretrained(self, state_dict):
        own_state = self.state_dict()

        # copy the pretrained weights to the model
        for name, param in state_dict.items():
            if name in own_state and param.shape == own_state[name].shape:
                own_state[name].copy_(param)

        if hasattr(self.decoder, "reset_for_finetuning"):
            self.decoder.reset_for_finetuning()


if __name__ == "__main__":

    # Example usage
    config = NeuralPredictionConfig()
    device = config.device
    config.conditioning_dim = 128  # feel free to change parameters
    config.decoder_hidden_size = 64

    config.seq_length = 64
    config.pred_length = 16  # context length is 64 - 16 = 48

    input_size = [[5, 10]]  # 2 sessions, 5 and 10 units
    model = POCO(config, input_size).to(device)

    x_list = [
        torch.randn(48, 2, 5).to(device),
        torch.randn(48, 2, 10).to(device),
    ]  # list of tensors, each of shape (L, batch, n_neurons)
    out = model(x_list)  # forward pass

    # The output will be a list of tensors, each of shape (pred_length, batch, n_neurons)
    assert out[0].shape == (16, 2, 5)
    assert out[1].shape == (16, 2, 10)
