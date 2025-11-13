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

class NeuralPredictionConfig:

    def __init__(self):
        self.seq_length = 64 # total length
        self.pred_length = 16 # prediction length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.compression_factor = 16 # 16 steps per token
        self.decoder_type = 'POYO'
        self.conditioning = 'mlp'
        self.conditioning_dim = 1024
        self.decoder_context_length = None

        self.poyo_num_latents = 8
        self.latent_session_embedding = False
        self.unit_embedding_components = ['session', ] # embeddings that will in added on top of unit embedding
        self.decoder_num_layers = 1
        self.decoder_num_heads = 16
        self.poyo_unit_dropout = 0
        self.rotary_attention_tmax = 100
        self.decoder_hidden_size = 128

        self.freeze_backbone = False
        self.freeze_conditioned_net = False

# Adapted from POYO: https://poyo-brain.github.io/
class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool =True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value
        self.using_memory_efficient_attn = use_memory_efficient_attn

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        rotary_time_emb_query,
        rotary_time_emb_context,
        *,
        context_mask=None,
        query_seqlen=None,
        context_seqlen=None,
    ):

        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        if self.using_memory_efficient_attn:
            if context_mask is not None:
                raise NotImplementedError(
                    f"Got non-None `attn_mask`. "
                    f"This implementation with memory efficient attention only works "
                    f"with `x_seqlen` for handling unequal sample lengths. Traditional "
                    f"padding approach is supported with normal non-memory efficient "
                    f"attention."
                )

            if query_seqlen is None or context_seqlen is None:
                raise ValueError(
                    f"Both `query_seqlen` and `context_seqlen` must be valid "
                    f"sequence lengths."
                )

            out = rotary_memory_efficient_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=query_seqlen,
                kv_seqlen=context_seqlen,
            )

        else:
            if query_seqlen is not None or context_seqlen is not None:
                raise NotImplementedError(
                    f"Got non-None `*_seqlen`. "
                    f"You are using torch's attention implementation, which only "
                    f"accepts `attn_mask`."
                    f"If you wish to use seqlen, please use memory efficient "
                    f"attention. "
                )

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb_query, 
                rotary_time_emb_kv=rotary_time_emb_context,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=context_mask,
            )
        
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        use_memory_efficient_attn: bool = True,
    ):
        super().__init__()

        if use_memory_efficient_attn and xops is None:
            logging.warning(
                "xformers is not installed, falling back to default attention"
            )
            use_memory_efficient_attn = False

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.using_memory_efficient_attn = use_memory_efficient_attn
        self.rotate_value = rotate_value

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self, 
        x, 
        rotary_time_emb, 
        *,
        x_mask=None,
        x_seqlen=None,
    ):

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        if self.using_memory_efficient_attn:
            if x_mask is not None:
                raise NotImplementedError(
                    f"Got non-None `attn_mask`. "
                    f"This implementation with memory efficient attention only works "
                    f"with `x_seqlen` for handling unequal sample lengths. Traditional "
                    f"padding approach is supported with normal non-memory efficient "
                    f"attention."
                )

            if x_seqlen is None:
                raise ValueError(
                    f"`x_seqlen` must be a valid sequence length."
                )

            out = rotary_memory_efficient_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb,
                rotary_time_emb_kv=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                q_seqlen=x_seqlen,
                kv_seqlen=None, # self-attention has the same seqlen for q, k, v
            )

        else:
            if x_seqlen is not None:
                raise NotImplementedError(
                    f"Got non-None `x_seqlen`. "
                    f"You are using torch's attention implementation, which only "
                    f"accepts `attn_mask`."
                    f"If you wish to use `x_seqlen`, please use memory efficient "
                    f"attention. "
                )

            out = rotary_default_attention(
                q=q, k=k, v=v,
                rotary_time_emb_q=rotary_time_emb,
                rotary_time_emb_kv=rotary_time_emb,
                num_heads=self.heads,
                dropout_p=self.dropout if self.training else 0,
                rotate_value=self.rotate_value,
                kv_mask=x_mask,
            )
        
        out = self.to_out(out)
        return out


def rotary_default_attention(
    *,
    q, # (b, n_q, (h d), )
    k, # (b, n_kv, (h d), )
    v, # (b, n_kv, (h d), )
    rotary_time_emb_q, # (b, n_q, d)
    rotary_time_emb_kv, # (b, n_kv, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    kv_mask=None, # (b, n_kv)
): # Output: (b, n, (h d), )
    r"""Wraps the default attention implementation with rotary embedding application.
    """

    # default attention expects shape b h n d
    q = rearrange(q, "b n (h d) -> b h n d", h=num_heads)
    k = rearrange(k, "b n (h d) -> b h n d", h=num_heads)
    v = rearrange(v, "b n (h d) -> b h n d", h=num_heads)

    # apply rotary embeddings
    q = apply_rotary_pos_emb(rotary_time_emb_q, q, dim=1)
    k = apply_rotary_pos_emb(rotary_time_emb_kv, k, dim=1)
    if rotate_value:
        v = apply_rotary_pos_emb(rotary_time_emb_kv, v, dim=1)

    # attention mask
    if kv_mask is not None:
        kv_mask = rearrange(kv_mask, "b n -> b () () n") 

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=kv_mask, dropout_p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q, out, dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    return out


def rotary_memory_efficient_attention(
    *,
    q, # (n, (h d), )
    k, # (n, (h d), )
    v, # (n, (h d), )
    rotary_time_emb_q, # (n, d)
    rotary_time_emb_kv, # (n, d)
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
    q_seqlen=None,
    kv_seqlen=None,
): # Output: (n, (h d), )
    r"""Wraps the memory efficient attention implementation with rotary embedding 
    application.
    """

    # xformers attention expects shape (1, n, h, d, ) 
    q = rearrange(q, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    k = rearrange(k, "n (h d) -> n h d", h=num_heads).unsqueeze(0)
    v = rearrange(v, "n (h d) -> n h d", h=num_heads).unsqueeze(0)

    q = apply_rotary_pos_emb(rotary_time_emb_q.unsqueeze(0), q)
    k = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), k)
    if rotate_value:
        v = apply_rotary_pos_emb(rotary_time_emb_kv.unsqueeze(0), v)

    # Fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        # xformers expects 'list' of seqlens
        if q_seqlen is None:
            raise ValueError(
                f"`q_seqlen` must be a valid sequence length."
            )
        elif isinstance(q_seqlen, torch.Tensor):
            q_seqlen = q_seqlen.tolist()
        elif not isinstance(q_seqlen, list):
            raise ValueError(
                f"`q_seqlen` must be a list or a torch.Tensor, "
                f"got {type(q_seqlen)}"
            )

        if kv_seqlen is not None:
            # xformers expects 'list' of seqlens
            if isinstance(kv_seqlen, torch.Tensor):
                kv_seqlen = kv_seqlen.tolist()
            elif not isinstance(kv_seqlen, list):
                raise ValueError(
                    f"`kv_seqlen` must be a list or a torch.Tensor, "
                    f"got {type(kv_seqlen)}"
                )
            
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    # perform attention, by default will use the optimal attention implementation
    out = xops.memory_efficient_attention(
        q, k, v, attn_bias=attn_bias, p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-rotary_time_emb_q.unsqueeze(0), out)

    # return (n, (h d), ), b = 1 is removed
    out = rearrange(out, "b n h d -> b n (h d)").squeeze(0)
    return out

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


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(freqs, x, dim=2):
    r"""Apply the rotary positional embedding to the input data.
    
    Args:
        freqs (torch.Tensor): Frequencies of the sinusoids.
        x (torch.Tensor): Input data.
        dim (int, optional): Dimension along which to rotate.
    """
    dtype = x.dtype
    if dim == 1:
        freqs = rearrange(freqs, "n ... -> n () ...")
    elif dim == 2:
        freqs = rearrange(freqs, "n m ... -> n m () ...")
    x = (x * freqs.cos().to(dtype)) + (rotate_half(x) * freqs.sin().to(dtype))
    return x


class Embedding(nn.Embedding):
    r"""A simple extension of :class:`torch.nn.Embedding` to allow more control over
    the weights initializer. The learnable weights of the module of shape 
    `(num_embeddings, embedding_dim)` are initialized from 
    :math:`\mathcal{N}(0, \text{init_scale})`.
    
    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        init_scale (float, optional): standard deviation of the normal distribution used
            for the initialization. Defaults to 0.02, which is the default value used in
            most transformer models.
        **kwargs: Additional arguments. Refer to the documentation of 
            :class:`torch.nn.Embedding` for details.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_scale: float=0.02,
        **kwargs,
    ):
        self.init_scale = init_scale
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        torch.nn.init.normal_(self.weight, mean=0, std=self.init_scale)
        self._fill_padding_idx_with_zero()

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
        t_max=120
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
            use_memory_efficient_attn=use_memory_efficient_attn,
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
                            use_memory_efficient_attn=use_memory_efficient_attn,
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
            use_memory_efficient_attn=use_memory_efficient_attn,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim
        self.using_memory_efficient_attn = self.enc_atn.using_memory_efficient_attn

    def forward(
        self,
        *,      # (   padded   ) or (   chained   )
        inputs, # (B, N_in, dim) or (N_all_in, dim)
        latents, # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries, # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps, # (B, N_in) or (N_all_in,)
        latent_timestamps, # (B, N_latent) or (N_all_latent,)
        output_query_timestamps, # (B, N_out) or (N_all_out,)
        input_mask=None, # (B, N_in) or None
        input_seqlen=None, # None or (B,)
        latent_seqlen=None, # None or (B,)
        output_query_seqlen=None, # None or (B,)
        output_latent=False
    ) -> Union[
        TensorType["batch", "*nqueries", "dim"], # if padded
        TensorType["ntotal_queries", "dim"], # if chained
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
            context_mask=input_mask, # used if default attention
            query_seqlen=latent_seqlen, # used if memory efficient attention
            context_seqlen=input_seqlen, # used if memory efficient attention
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

class POYO(nn.Module):

    def __init__(
        self,
        *,
        input_dim=1,
        dim=512,
        dim_head=64,
        num_latents=64,
        depth=2,
        cross_heads=1,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        emb_init_scale=0.02,
        use_memory_efficient_attn=True,
        input_size=None,
        query_length=1,
        T_step=1,
        unit_dropout=0.0,
        output_latent=False, # if True, return the latent representation, else return the query representation
        t_max=100,
        num_datasets=1,
        unit_embedding_components=['session', ],
        latent_session_embedding=False,
    ):
        super().__init__()

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
            use_memory_efficient_attn=use_memory_efficient_attn,
            t_max=t_max,
        )

        self.dim = dim
        self.T_step = T_step
        self.using_memory_efficient_attn = self.perceiver_io.using_memory_efficient_attn
        self.output_latent = output_latent
        self.unit_embedding_components = unit_embedding_components
        self.latent_session_embedding = latent_session_embedding

    def forward(
        self,
        # input sequence
        x: torch.Tensor, # sum(B * D), L, input_dim
        unit_indices, # sum(B * D)
        unit_timestamps, # sum(B * D), L
        input_seqlen, # (B, )
        # output sequence
        session_index,  # (B, )
        dataset_index, # (B, )
    ):

        # input
        L = x.shape[1]
        B = input_seqlen.shape[0]
        T = L * self.T_step
        unit_embedding = self.unit_emb(unit_indices)

        session_indices = torch.concatenate([torch.full((input_seqlen[i], ), session_index[i], device=x.device) for i in range(B)]) # sum(B * D)
        dataset_indices = torch.concatenate([torch.full((input_seqlen[i], ), dataset_index[i], device=x.device) for i in range(B)]) # sum(B * D)

        if 'session' in self.unit_embedding_components:
            unit_embedding = unit_embedding + self.session_emb(session_indices) + self.dataset_emb(dataset_indices) # sum(B * D), dim

        inputs = unit_embedding.unsqueeze(1) + self.input_proj(x) # sum(B * D), L, dim
        unit_timestamps = unit_timestamps.reshape(-1)
        inputs = inputs.reshape(-1, inputs.shape[2])

        # latents
        latent_index = torch.arange(self.num_latents, device=x.device)
        latents = self.latent_emb(latent_index)
        latents = latents.repeat(B, 1, 1) # B, N_latent, dim
        if self.latent_session_embedding:
            latents = latents + self.session_emb(session_index).unsqueeze(1) + self.dataset_emb(dataset_index).unsqueeze(1) # B, N_latent, dim
        latents = latents.reshape(-1, latents.shape[2])
        latent_seqlen = torch.full((B, ), self.num_latents, device=x.device)
        latent_timestamps = torch.arange(0, T, step=T / self.num_latents, device=x.device)
        latent_timestamps = latent_timestamps.repeat(B) # B * N_latent

        # outputs
        output_queries = unit_embedding
        sumD = output_queries.shape[0]
        output_queries = output_queries.repeat_interleave(self.query_length, dim=0) # sum(B * D * q_len), dim
        output_timestamps = torch.arange(self.query_length, device=x.device).repeat(sumD) + T # sum(B * D * q_len)

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
        self.dataset_idx = [] # the dataset index for each session
        for i_dataset, size in enumerate(input_size):
            self.dataset_idx += [i_dataset] * len(size)
        self.num_datasets = len(input_size)
        input_size = list(itertools.chain(*input_size))
        self.tokenizer = None
        self.tokenizer_type = 'none'
        self.token_dim = config.compression_factor
        self.T_step = config.compression_factor

        self.input_size = input_size
        self.pred_step = config.pred_length

        assert config.decoder_type == 'POYO'
        self.decoder = POYO(
            input_dim=self.token_dim, 
            dim=config.decoder_hidden_size,
            depth=config.decoder_num_layers, 
            self_heads=config.decoder_num_heads,
            input_size=input_size,
            num_latents=config.poyo_num_latents,
            T_step=self.T_step,
            unit_dropout=config.poyo_unit_dropout,
            output_latent=False,
            t_max=config.rotary_attention_tmax,
            num_datasets=self.num_datasets,
            unit_embedding_components=config.unit_embedding_components,
            latent_session_embedding=config.latent_session_embedding,
        )
        self.embedding_dim = config.decoder_hidden_size
        self.linear_out_size = config.pred_length

        self.conditioning = config.conditioning
        self.conditioning_dim = config.conditioning_dim

        assert config.conditioning == 'mlp'
        self.in_proj = nn.Sequential(nn.Linear(self.Tin, config.conditioning_dim), nn.ReLU())
        
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
            if config.decoder_type == 'POYO':
                # allow all embedding layers to be trained
                self.decoder.embedding_requires_grad(True)

        if config.freeze_conditioned_net:
            assert config.conditioning == 'mlp', "Only support freezing conditioned net for MLP conditioning"
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
        x = torch.concatenate([x.permute(1, 2, 0).reshape(-1, L) for x in x_list], dim=0) # sum(B * D), L

        # only use the last Tin steps
        if L != self.Tin:
            x = x[:, -self.Tin: ]

        # Tokenize the input sequence
        if self.tokenizer_type == 'vqvae':
            with torch.no_grad():
                out = self.tokenizer.encode(x) # out: sum(B * D), TC, E
        elif self.tokenizer_type == 'cnn':
            out = x.unsqueeze(1)
            out = self.tokenizer(out) # out: sum(B * D), C, TC
            out = out.permute(0, 2, 1) # out: sum(B * D), TC, C
        elif self.tokenizer_type == 'none':
            out = x.reshape(x.shape[0], self.Tin // self.T_step, self.T_step) # out: sum(B * D), TC, E
        else:
            raise ValueError(f"Unknown tokenizer type {self.tokenizer_type}")
        d_list = self.input_size

        if unit_indices is None:
            sum_channels = 0
            unit_indices = []
            for b, d in zip(bsz, self.input_size):
                indices = torch.arange(d, device=x.device).unsqueeze(0).repeat(b, 1).reshape(-1) # B * D
                unit_indices.append(indices + sum_channels)
                sum_channels += d
            unit_indices = torch.cat(unit_indices, dim=0) # sum(B * D)
        if unit_timestamps is None:
            unit_timestamps = torch.zeros_like(unit_indices).unsqueeze(1) + torch.arange(0, self.Tin, self.T_step, device=x.device) # sum(B * D), TC

        input_seqlen = torch.cat([torch.full((b, ), d, device=x.device) 
                                        for b, d in zip(bsz, self.input_size)], dim=0)
        session_index = torch.cat([torch.full((b, ), i, device=x.device) 
                                        for i, b in enumerate(bsz)], dim=0)
        dataset_index = torch.cat([torch.full((b, ), self.dataset_idx[i], device=x.device)
                                        for i, b in enumerate(bsz)], dim=0)

        embed = self.decoder(
            out,
            unit_indices=unit_indices,
            unit_timestamps=unit_timestamps,
            input_seqlen=input_seqlen,
            session_index=session_index,
            dataset_index=dataset_index,
        ) # sum(B * D), embedding_dim; or sum(B * D), pred_length, embedding_dim

        # partition embed to a list of tensors, each of shape (B, D, embedding_dim)
        split_size = [b * d for b, d in zip(bsz, d_list)]
        embed = torch.split(embed, split_size, dim=0)
        embed = [xx.reshape(b, d, self.embedding_dim) for xx, b, d in zip(embed, bsz, d_list)] # (B, D, E)

        preds = []
        for i, (e, d, input) in enumerate(zip(embed, self.input_size, x_list)):
            alpha = self.conditioning_alpha(e) # B, D, cond_dim
            beta = self.conditioning_beta(e) # B, D, cond_dim
            input = input.permute(1, 2, 0) # B, D, L
            weights = self.in_proj(input) * alpha + beta # B, D, cond_dim
            pred = self.out_proj(weights) # B, D, pred_length
            preds.append(pred.permute(2, 0, 1))

        return preds
    
    def load_pretrained(self, state_dict):
        own_state = self.state_dict()

        # copy the pretrained weights to the model
        for name, param in state_dict.items():
            if name in own_state and param.shape == own_state[name].shape:
                own_state[name].copy_(param)

        if hasattr(self.decoder, 'reset_for_finetuning'):
            self.decoder.reset_for_finetuning()

if __name__ == "__main__":

    # Example usage
    config = NeuralPredictionConfig()
    device = config.device
    config.conditioning_dim = 128 # feel free to change parameters
    config.decoder_hidden_size = 64

    config.seq_length = 64
    config.pred_length = 16 # context length is 64 - 16 = 48

    input_size = [[5, 10]] # 2 sessions, 5 and 10 units
    model = POCO(config, input_size).to(device)

    x_list = [torch.randn(48, 2, 5).to(device), torch.randn(48, 2, 10).to(device)] # list of tensors, each of shape (L, batch, n_neurons)
    out = model(x_list) # forward pass

    # The output will be a list of tensors, each of shape (pred_length, batch, n_neurons)
    assert out[0].shape == (16, 2, 5)
    assert out[1].shape == (16, 2, 10)