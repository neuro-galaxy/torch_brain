import functools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_brain.nn import RotaryTimeEmbedding

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None


class Attention(nn.Module):
    r"""Pre-norm self-attention supporting fixed and variable-length sequences.

    Args:
        dim: Input and output dimension.
        heads: Number of attention heads.
        dim_head: Dimension of each attention head.
        bias: Whether the query/key/value projection has a bias. Defaults to ``True``.
        rotate_value: Whether rotary embeddings are also applied to the values.
            Defaults to ``False``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        bias: bool = True,
        rotate_value: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.rotate_value = rotate_value
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=bias)
        self.to_out = nn.Linear(heads * dim_head, dim)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor | None = None,
        seqlen: Tensor | None = None,
    ) -> Tensor:
        r"""
        Args:
            x: Tokens of shape :math:`(B, N, D)`, or :math:`(N_{total}, D)` when
                ``seqlen`` is given.
            rotary_emb: Rotary embeddings from
                :class:`~torch_brain.nn.RotaryTimeEmbedding`, broadcastable to the
                token dimensions of ``x``.
            seqlen: Lengths of the sequences chained in ``x``, of shape :math:`(S,)`.
                Tokens only attend within their own sequence.

        Returns:
            Tensor of the same shape as ``x``.
        """
        q, k, v = (
            self.to_qkv(self.norm(x)).unflatten(-1, (3, self.heads, -1)).unbind(-3)
        )

        if rotary_emb is not None:
            q = RotaryTimeEmbedding.rotate(q, rotary_emb, unsqueeze_dim=-2)
            k = RotaryTimeEmbedding.rotate(k, rotary_emb, unsqueeze_dim=-2)
            if self.rotate_value:
                v = RotaryTimeEmbedding.rotate(v, rotary_emb, unsqueeze_dim=-2)

        if seqlen is None:
            out = _attention(q, k, v)
        else:
            out = _varlen_attention(q, k, v, seqlen.to(x.device))

        if rotary_emb is not None and self.rotate_value:
            inv_rotary_emb = RotaryTimeEmbedding.invert(rotary_emb)
            out = RotaryTimeEmbedding.rotate(out, inv_rotary_emb, unsqueeze_dim=-2)

        return self.to_out(out.flatten(-2))


def _attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    q, k, v = (t.transpose(-3, -2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(-3, -2)


def _varlen_attention(q: Tensor, k: Tensor, v: Tensor, seqlen: Tensor) -> Tensor:
    max_seqlen = int(seqlen.max())

    can_use_varlen_attn = (
        varlen_attn is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    if can_use_varlen_attn:
        cu_seqlen = F.pad(seqlen.cumsum(0, dtype=torch.int32), (1, 0))
        return varlen_attn(q, k, v, cu_seqlen, cu_seqlen, max_seqlen, max_seqlen)

    # Use normal SDPA if we cannot use torch's varlen_attn.
    _warn_no_varlen_attn()
    mask = torch.arange(max_seqlen, device=q.device) < seqlen[:, None]
    q, k, v = (_pad(t, mask) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
    return out.transpose(1, 2)[mask]


def _pad(x: Tensor, mask: Tensor) -> Tensor:
    padded = x.new_zeros(*mask.shape, *x.shape[1:])
    padded[mask] = x
    return padded.transpose(1, 2)


@functools.cache
def _warn_no_varlen_attn():
    warnings.warn(
        "torch's varlen_attn is unavailable (requires torch>=2.9, CUDA, and fp16/bf16 "
        "inputs). Falling back to padded SDPA, which is slower.",
        stacklevel=2,
    )
