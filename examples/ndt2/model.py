import math
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from attention import FFN, CrossAttention, SelfAttention, SelfAttentionBlock
from einops import einsum, rearrange, repeat
from torch import Tensor

from torch_brain.nn import InfiniteVocabEmbedding


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def sincos_position_embeddings(num_pos, dim, max_num_pos=10000):
    position = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
    div_term = (1.0 / float(max_num_pos)) ** (torch.arange(0, dim, 2).float() / dim)
    pe = torch.empty(num_pos, dim)
    pe[:, ::2] = torch.sin(torch.pi * position * div_term)
    pe[:, 1::2] = torch.cos(torch.pi * position * div_term)
    return pe


class NDT2_MaskManager(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.inc_mask = True

    def forward(self, batch: Dict[str, torch.Tensor]):
        # -- Mask
        if self.mask_ratio is not None and self.inc_mask:
            spikes = batch["spike_tokens"]
            shuffle = torch.randperm(spikes.size(1), device=spikes.device)
            encoder_frac = int((1 - self.mask_ratio) * spikes.size(1))
            for key in ["spike_tokens", "time_idx", "space_idx"]:
                t = batch[key].transpose(1, 0)[shuffle].transpose(1, 0)
                batch[key] = t[:, :encoder_frac]
                batch[f"target_{key}"] = t[:, encoder_frac:]

        return batch


class NDT2_Patchifier(nn.Module):
    def __init__(
        self,
        dim,
        patch_size,
        max_time_patches,
        max_space_patches,
    ):
        """
        Args:
            dim (Int): Dimension of the output patchified tensor
            patch_size (Tuple[Int, Int]): (num_neurons, num_time_bins)
            max_time_patches (Int): Maximum number of time patches
            max_space_patches (Int): Maximum number of space patches
        """
        super().__init__()
        spike_embed_dim = round(dim / patch_size[0])
        max_neuron_count = 21
        pad = 5
        self.readin = nn.Embedding(max_neuron_count, spike_embed_dim, padding_idx=pad)
        self.net = nn.Linear(patch_size[0] * patch_size[1], dim)

        self.time_emb = nn.Embedding(max_time_patches, dim)
        self.space_emb = nn.Embedding(max_space_patches, dim)

        self.ses_emb = InfiniteVocabEmbedding(dim, init_scale=1.0)
        self.ses_flag = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.subj_emb = InfiniteVocabEmbedding(dim, init_scale=1.0)
        self.subj_flag = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

    def session_emb(self, session: torch.Tensor) -> torch.Tensor:
        return self.ses_emb(session) + self.ses_flag

    def subject_emb(self, subject: torch.Tensor) -> torch.Tensor:
        return self.subj_emb(subject) + self.subj_flag

    def forward(self, spikes, time_idx, space_idx, session_idx=None, subject_idx=None):
        """
        Args:
            x (torch.Tensor): Binned spikes (bs, T, patch_size[0], patch_size[1])
            session_idx (torch.LongTensor):  session index (bs,)
            subject_idx (torch.LongTensor):  subject index (bs,)
        Returns: (NxT, D)
        """
        x = rearrange(spikes, "bs T Pn Pt -> bs T (Pn Pt)")
        # x = self.net(x.float())
        x = self.readin(x).flatten(-2, -1)
        x = x + self.time_emb(time_idx) + self.space_emb(space_idx)

        context_embs = []
        if session_idx is not None:
            context_embs.append(self.session_emb(session_idx).to(x.dtype))

        if subject_idx is not None:
            context_embs.append(self.subject_emb(subject_idx).to(x.dtype))

        if len(context_embs) > 0:
            context_embs = torch.stack(context_embs, dim=1)
            x = torch.cat([x, context_embs], dim=1)

        return x


class NDT2_TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, ffn_mult=4, dropout=0.0, out_norm=True):
        """
        Args:
            dim (Int): Dimension of the input/output tensor
            n_attn_layers (Int): Number of Attention layers (depth)
            n_heads (Int): Number of heads for MHA
            inter_dim (Int): Dimension of the intermediate MLP layers
            dropout (Float): Dropout rate in Attention layers
        """
        super().__init__()

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    heads=heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim) if out_norm else nn.Identity()

    def forward(self, x, seqlen):
        """
        Args:
            x (torch.Tensor): Patchified spikes (over batch) (n_token, D); n_token = B x NxT
            seqlen (torch.Tensor): seq lengths over batches (B)
        """
        for layer in self.layers:
            x = layer(x, seqlen)
        x = self.out_norm(x)
        return x


class NDT2_Predictor(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        patch_size,
        ffn_mult=4,
        dropout=0.0,
        max_time_patches=256,
        max_space_patches=1024,
    ):
        super().__init__()

        # -- Embeddings
        self.mask_emb = nn.Parameter(torch.randn(dim))
        self.time_emb = nn.Embedding(max_time_patches, dim)
        self.space_emb = nn.Embedding(max_space_patches, dim)

        # -- Transformer
        self.layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim,
                    heads=heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # -- Out proj
        self.patch_size = patch_size
        self.out_net = nn.Linear(dim, patch_size[0] * patch_size[1])

    def forward(self, x, time_idx, space_idx, seqlen, mask):

        # positional embeddings
        x[mask] = self.mask_emb
        x += self.time_emb(time_idx)
        x += self.space_emb(space_idx)

        # transformer
        for layer in self.layers:
            x = layer(x, seqlen)

        # output
        x = self.out_net(x).view(-1, self.patch_size[0], self.patch_size[1])
        return x


def bhv_loss_fn(pred, target, weights, type):
    if type == "regression":
        # for multi-GPUs
        weighted_loss = (
            F.mse_loss(pred, target, reduction="none").mean(dim=-1) * weights
        )
        weighted_loss_sum = AllReduceSum.apply(weighted_loss.sum())
        weight_sum = AllReduceSum.apply(weights.sum())
        loss = weighted_loss_sum / weight_sum

        # for single GPU
        # loss = weighted_loss.sum() / weights.sum()
    elif type == "class":
        loss = F.cross_entropy(pred, target.long())
    return loss


class NDT2_TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        depth,
        dim_out,
        ffn_mult=4,
        dropout=0.0,
        max_time_patches=256,
        max_space_patches=1024,
        loss_type="regression",
        ctx_pos_emb=False,
    ):
        super().__init__()

        self.dim = dim
        self.loss_type = loss_type

        self.query_emb = nn.Parameter(torch.randn(dim))
        self.query_time_emb = nn.Embedding(max_time_patches, dim)

        self.ctx_pos_emb = ctx_pos_emb
        if ctx_pos_emb:
            self.x_time_emb = nn.Embedding(max_time_patches, dim)
            self.x_space_emb = nn.Embedding(max_space_patches, dim)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttention(dim=dim, heads=heads, dropout=dropout),
                        SelfAttention(dim=dim, heads=heads, dropout=dropout),
                        FFN(dim=dim, mult=ffn_mult, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(dim, dim_out)

    def forward(
        self,
        x,
        spike_seqlen,
        time_idx,
        space_idx,
        output_time_idx,
        output_weights,
        output_seqlen,
        targets=None,
    ):

        y = self.query_emb + self.query_time_emb(output_time_idx)
        if self.ctx_pos_emb:
            x = x + self.x_time_emb(time_idx) + self.x_space_emb(space_idx)

        for ca, sa, ffn in self.layers:
            y = y + ca(x_q=y, x_kv=x, q_seqlen=output_seqlen, kv_seqlen=spike_seqlen)
            y = y + sa(x=y, seqlen=output_seqlen)
            y = y + ffn(y)

        y = self.out_proj(y)

        loss = None
        if targets is not None:
            loss = bhv_loss_fn(y, targets, output_weights, self.loss_type)

        return y, loss
