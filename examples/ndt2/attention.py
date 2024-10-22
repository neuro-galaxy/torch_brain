import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from einops import rearrange


class FFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, bias=True):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            inter_size (Int): Dimension of the intermediate MLP layers
            dropout (Float): Dropout rate in MLP layers
        """
        super().__init__()

        inter_size = dim * mult
        self.ln = nn.LayerNorm(dim)
        self.up_proj = nn.Linear(dim, inter_size, bias=bias)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(inter_size, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Patchified spikes (over batch) (BxNxT, D)
        """
        x = self.ln(x)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, norm_kv=False, bias=True):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            n_heads (Int): Number of heads for MHA
            dropout (Float): Dropout rate in Attention layers
        """
        super().__init__()

        # Architecture config
        self.dim = dim
        self.heads = heads
        assert self.dim % self.heads == 0, f"Hidden dim is not multiple of head size"
        self.dim_head = self.dim // self.heads

        # Attention parameters
        self.ln_q = nn.LayerNorm(self.dim)
        self.ln_kv = nn.LayerNorm(self.dim) if norm_kv else nn.Identity()
        self.to_q = nn.Linear(self.dim, self.dim, bias=bias)
        self.to_kv = nn.Linear(self.dim, 2 * self.dim, bias=bias)
        self.attn_dropout = dropout

        # Final projection
        self.dropout = nn.Dropout(self.attn_dropout)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        if self.to_q.bias is not None:
            nn.init.constant_(self.to_q.bias, 0.0)
        if self.to_kv.bias is not None:
            nn.init.constant_(self.to_kv.bias, 0.0)

    def forward(
        self,
        x_q: torch.FloatTensor,
        x_kv: torch.FloatTensor,
        q_seqlen: torch.IntTensor,
        kv_seqlen: torch.IntTensor,
    ) -> torch.FloatTensor:  # (n_token, D)
        """
        Args:
            x (torch.Tensor): Patchified spikes (over batch) (n_token, D); n_token = B x NxT
            seqlen (torch.Tensor): seq lengths over batches (B)
        """
        assert x_q.size(0) == q_seqlen.sum(), f"Input size mismatch"
        assert x_kv.size(0) == kv_seqlen.sum(), f"Input size mismatch"

        q = self.to_q(self.ln_q(x_q))
        k, v = self.to_kv(self.ln_kv(x_kv)).chunk(2, dim=-1)

        q = rearrange(q, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)
        k = rearrange(k, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)
        v = rearrange(v, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)

        # fill attention_bias with BlockDiagonalMask
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=q_seqlen.tolist(),
            kv_seqlen=kv_seqlen.tolist(),
        )

        out = xops.memory_efficient_attention(q, k, v, attn_bias, self.attn_dropout)
        out = rearrange(out, "1 N H D_H -> N (H D_H)")
        out = self.dropout(self.out_proj(out))
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, bias=True):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            n_heads (Int): Number of heads for MHA
            dropout (Float): Dropout rate in Attention layers
        """
        super().__init__()

        # Architecture config
        self.dim = dim
        self.heads = heads
        assert self.dim % self.heads == 0, f"Hidden dim is not multiple of head size"
        self.dim_head = self.dim // self.heads

        # Attention parameters
        self.ln = nn.LayerNorm(self.dim)
        self.to_qkv = nn.Linear(self.dim, 3 * self.dim, bias=bias)
        self.attn_dropout = dropout

        # Final projection
        self.dropout = nn.Dropout(self.attn_dropout)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qkv.weight)
        if self.to_qkv.bias is not None:
            nn.init.constant_(self.to_qkv.bias, 0.0)

    def forward(
        self,
        x: torch.FloatTensor,
        seqlen: torch.IntTensor,
    ) -> torch.FloatTensor:  # (n_token, D)
        """
        Args:
            x (torch.Tensor): Patchified spikes (over batch) (n_token, D); n_token = B x NxT
            seqlen (torch.Tensor): seq lengths over batches (B)
        """
        assert x.size(0) == seqlen.sum(), f"Input size mismatch"

        q, k, v = self.to_qkv(self.ln(x)).chunk(3, dim=-1)

        q = rearrange(q, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)
        k = rearrange(k, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)
        v = rearrange(v, "N (H D_H) -> 1 N H D_H", H=self.heads, D_H=self.dim_head)

        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=seqlen.tolist(),
            kv_seqlen=seqlen.tolist(),
        )

        out = xops.memory_efficient_attention(q, k, v, attn_bias, self.attn_dropout)
        out = rearrange(out, "1 N H D_H -> N (H D_H)")
        out = self.dropout(self.out_proj(out))
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads, ffn_mult=4, dropout=0.0):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            n_heads (Int): Number of heads for MHA
            inter_dim (Int): Dimension of the intermediate MLP layers
            dropout (Float): Dropout rate in Attention layers
        """
        super().__init__()

        # Encoder block
        self.attn = SelfAttention(dim=dim, heads=heads, dropout=dropout)
        self.ffn = FFN(dim=dim, mult=ffn_mult, dropout=dropout)

    def forward(
        self, x: torch.FloatTensor, seqlen: torch.IntTensor
    ) -> torch.FloatTensor:  # (n_token, D)
        """
        Args:
            x (torch.Tensor): Patchified spikes (over batch) (n_token, D); n_token = B x NxT
            seqlen (torch.Tensor): seq lengths over batches (B)
        """
        x = x + self.attn(x, seqlen)
        x = x + self.ffn(x)
        return x
