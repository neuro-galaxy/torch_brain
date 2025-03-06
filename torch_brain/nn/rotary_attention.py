from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

try:
    import xformers.ops as xops
except ImportError:
    xops = None


from torch_brain.nn.rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    """Cross-attention layer with rotary positional embeddings.

    This layer performs cross-attention between a query sequence and a context sequence,
    with rotary positional embeddings applied to the queries and keys (and optionally values).
    It first normalizes the inputs, projects them to query/key/value space, applies rotary
    embeddings and attention, then projects back to the original dimension.

    The layer provides two forward methods:

    - forward(): This is the default, and is used for sequences in a batch that are of
      the same length, or are padded to the same length. When padding is used, attention
      masks need to be provided.
    - forward_varlen(): Uses sequence lengths instead of masks for sequences that are chained
      together in a single batch dimension. This can be more memory efficient since it avoids
      padding, but requires the sequences to be concatenated rather than stacked.

    Args:
        dim (int): Dimension of input query embeddings
        context_dim (Optional[int]): Dimension of input context embeddings. If None, uses same as dim
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        dropout (float): Dropout probability
        rotate_value (bool): Whether to apply rotary embeddings to values as well as queries/keys
    """

    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        query_pos_emb,
        context_pos_emb,
        context_mask=None,
    ):
        """Forward pass for regular or padded sequences.

        Shape:
            - x_query: (B, N_q, D_q)
            - x_context: (B, N_c, D_c)
            - query_pos_emb: (B, N_q, D_h)
            - context_pos_emb: (B, N_c, D_h)
            - context_mask: Optional[Tensor[B, N_c]]
            - Output: (B, N_q, D)

            where B is batch size, N_q is query sequence length, N_c is context sequence
            length, D_q is input dimension, D_c is context dimension, H is number of heads,
            and D_h is head dimension.
        """
        # normalize
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        # project to q, k, v
        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        # select attention kernel
        if xops is not None and x_query.device.type == "cuda":
            # if xformers is available, use it for attention.
            # xformers supports attention masks when using the memory efficient attention
            # kernel, but pytorch does not.
            rotary_attn_func = rotary_attn_xformers_func
        else:
            # otherwise use pytorch's default attention which will determine the best
            # attention kernel (math, mem_efficient or flash) based on the hardware and
            # other factors.
            rotary_attn_func = rotary_attn_pytorch_func

        # apply attention
        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=query_pos_emb,
            kv_pos_emb=context_pos_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            attn_mask=context_mask,
        )

        # project back to dim
        out = self.to_out(out)
        return out

    def forward_varlen(
        self,
        x_query,
        x_context,
        query_pos_emb,
        context_pos_emb,
        query_seqlen,
        context_seqlen,
    ):
        """Forward pass for variable length sequences.

        Similar to forward() but handles variable length sequences that have been chained
        together in the batch dimension rather than being stacked and padded. This approach
        can be more memory efficient since it avoids padding, but requires the sequences
        to be concatenated rather than stacked.

        Shape:
            - x_query: (N_q_total, D)
            - x_context: (N_c_total, D_c)
            - query_pos_emb: (N_q_total, D_h)
            - context_pos_emb: (N_c_total, D_h)
            - query_seqlen: (B,)
            - context_seqlen: (B,)
            - Output: (N_q_total, D)

            where N_q_total and N_c_total are the total sequence lengths across the batch,
            B is batch size, D is input dimension, D_c is context dimension, H is number of
            heads, and D_h is head dimension.
        """
        # normalize
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        # project to q, k, v
        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        # select attention kernel
        if xops is not None and x_query.device.type == "cuda":
            rotary_attn_func = rotary_attn_xformers_varlen_func
        else:
            if x_query.device.type == "cuda":
                raise RuntimeError(
                    "No varlen attention kernel available, please install xformers."
                )
            else:
                # forward_varlen is not implemented for CPU, forward should be used instead
                raise NotImplementedError(
                    "No varlen attention kernel available for CPU."
                )

        # apply attention
        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=query_pos_emb,
            kv_pos_emb=context_pos_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            q_seqlen=query_seqlen,
            kv_seqlen=context_seqlen,
        )

        # project back to dim
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    """Self-attention layer with rotary positional embeddings.

    This layer performs self-attention within a sequence, with rotary positional embeddings
    applied to the queries and keys (and optionally values). It first normalizes the input,
    projects it to query/key/value space, applies rotary embeddings and attention, then
    projects back to the original dimension.

    The layer provides two forward methods:
    - forward(): This is the default, and is used for sequences in a batch that are of
      the same length, or are padded to the same length. When padding is used, attention
      masks need to be provided.
    - forward_varlen(): Uses sequence lengths instead of masks for sequences that are chained
      together in a single batch dimension. This can be more memory efficient since it avoids
      padding, but requires the sequences to be concatenated rather than stacked.

    Args:
        dim (int): Dimension of input embeddings
        heads (int): Number of attention heads
        dim_head (int): Dimension of each attention head
        dropout (float): Dropout probability
        rotate_value (bool): Whether to apply rotary embeddings to values as well as queries/keys
    """

    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        rotary_time_emb,
        x_mask=None,
    ):
        """Forward pass for fixed-length sequences.

        Shape:
            - x: (B, N, D)
            - rotary_time_emb: (B, N, D_h)
            - x_mask: (B, N, N)
            - Output: (B, N, D)

            where B is batch size, N is sequence length, D is input dimension,
            and D_h is head dimension.
        """
        # normalize
        x = self.norm(x)

        # project to q, k, v
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # select attention kernel
        if xops is not None and x.device.type == "cuda":
            rotary_attn_func = rotary_attn_xformers_func
        else:
            rotary_attn_func = rotary_attn_pytorch_func

        # apply attention
        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=rotary_time_emb,
            kv_pos_emb=rotary_time_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            attn_mask=x_mask,
        )

        # project back to dim
        out = self.to_out(out)
        return out

    def forward_varlen(
        self,
        x,
        rotary_time_emb,
        x_seqlen,
    ):
        """Forward pass for variable-length sequences.

        Shape:
            - x: (N_total, D)
            - rotary_time_emb: (N_total, D_h)
            - x_seqlen: (B,)
            - Output: (N_total, D)

            where N_total is the total sequence length across the batch,
            B is batch size, D is input dimension, and D_h is head dimension.
        """
        # normalize
        x = self.norm(x)

        # project to q, k, v
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # select attention kernel
        if xops is not None and x.device.type == "cuda":
            rotary_attn_func = rotary_attn_xformers_varlen_func
        else:
            if x.device.type == "cuda":
                raise RuntimeError(
                    "No varlen attention kernel available, please install xformers."
                )
            else:
                # forward_varlen is not implemented for CPU, forward should be used instead
                raise NotImplementedError(
                    "No varlen attention kernel available for CPU."
                )

        # apply attention
        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=rotary_time_emb,
            kv_pos_emb=rotary_time_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            q_seqlen=x_seqlen,
            kv_seqlen=None,  # self-attention has the same seqlen for q, k, v
        )

        # project back to dim
        out = self.to_out(out)
        return out


def rotary_attn_pytorch_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    # uses the default scaled dot product attention from pytorch
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # this implements basic versions of memory efficient attention and flash attention
    # but more advanced versions are available in xformers and flash_attn (varlen)
    # which allow us to perform complex masking operations
    r"""Wraps the default attention implementation with rotary embedding application.

    Args:
        query: The query tensor, with shape (b, n_q, (h d))
        key: The key tensor, with shape (b, n_kv, (h d))
        value: The value tensor, with shape (b, n_kv, (h d))
        q_pos_emb: The query rotary position embedding, with shape (b, n_q, d)
        kv_pos_emb: The key rotary position embedding, with shape (b, n_kv, d)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key
        attn_mask: The attention mask, with shape (b, n_kv)

    Returns:
        The output tensor, with shape (b, n_q, (h d))
    """

    # default attention expects shape b h n d
    query = rearrange(query, "b n (h d) -> b h n d", h=num_heads)
    key = rearrange(key, "b n (h d) -> b h n d", h=num_heads)
    value = rearrange(value, "b n (h d) -> b h n d", h=num_heads)

    # apply rotary embeddings
    query = apply_rotary_pos_emb(q_pos_emb, query, head_dim=1)
    key = apply_rotary_pos_emb(kv_pos_emb, key, head_dim=1)
    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb, value, head_dim=1)

    # attention mask
    if attn_mask is not None:
        attn_mask = rearrange(attn_mask, "b n -> b () () n")

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb, out, head_dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    return out


def rotary_attn_xformers_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    r"""Wraps the memory efficient attention implementation with rotary embedding
    application.

    Args:
        query: The query tensor, with shape (b n (h d))
        key: The key tensor, with shape (b n (h d))
        value: The value tensor, with shape (b n (h d))
        q_pos_emb: The query rotary position embedding, with shape (b n d)
        kv_pos_emb: The key rotary position embedding, with shape (b n d)
        attn_mask: The attention mask, with shape (b, n_kv). A value of True indicates
            that the element should take part in attention.
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key

    Returns:
        The output tensor, with shape (b n (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    query = rearrange(query, "b n (h d) -> b n h d", h=num_heads)
    key = rearrange(key, "b n (h d) -> b n h d", h=num_heads)
    value = rearrange(value, "b n (h d) -> b n h d", h=num_heads)

    query = apply_rotary_pos_emb(q_pos_emb, query, head_dim=2)
    key = apply_rotary_pos_emb(kv_pos_emb, key, head_dim=2)

    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb, value, head_dim=2)

    # WARNING: this is very slow, avoid using attn_mask if possible, refer to xformers
    # documentation
    attn_mask = (
        repeat(attn_mask, "b m -> b h n m", h=num_heads, n=query.size(1))
        if attn_mask is not None
        else None
    )
    attn_bias = (
        attn_mask.to(query.dtype).masked_fill(attn_mask.logical_not(), float("-inf"))
        if attn_mask is not None
        else None
    )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb, out, head_dim=2)

    out = rearrange(out, "b n h d -> b n (h d)")
    return out


def rotary_attn_xformers_varlen_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    q_seqlen,
    kv_seqlen,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):
    r"""Wraps the memory efficient attention implementation with rotary embedding
    application.

    Args:
        query: The query tensor, with shape (n, (h d))
        key: The key tensor, with shape (n, (h d))
        value: The value tensor, with shape (n, (h d))
        query_pos_emb: The query rotary position embedding, with shape (n, d)
        key_pos_emb: The key rotary position embedding, with shape (n, d)
        num_heads: The number of attention heads
        dropout_p: The dropout probability
        rotate_value: Whether to rotate the value in addition to the query and key
        q_seqlen: The sequence length of the query tensor
        kv_seqlen: The sequence length of the key and value tensors

    Returns:
        The output tensor, with shape (n, (h d))
    """
    # xformers attention expects shape (1, n, h, d)
    query = rearrange(query, "n (h d) -> () n h d", h=num_heads)
    key = rearrange(key, "n (h d) -> () n h d", h=num_heads)
    value = rearrange(value, "n (h d) -> () n h d", h=num_heads)

    # TODO check rotation works
    query = apply_rotary_pos_emb(q_pos_emb.unsqueeze(0), query)
    key = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), key)

    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb.unsqueeze(0), value)

    if isinstance(q_seqlen, torch.Tensor):
        q_seqlen = q_seqlen.tolist()
    if isinstance(kv_seqlen, torch.Tensor):
        kv_seqlen = kv_seqlen.tolist()

    # fill attention_bias with BlockDiagonalMask
    with torch.no_grad():
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
        )

    out = xops.memory_efficient_attention(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb.unsqueeze(0), out)

    out = rearrange(out, "() n h d -> n (h d)")
    return out
