from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from torch_brain.nn import RotaryEmbedding, RotaryCrossAttention, RotarySelfAttention


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
        batch_type: Union[str, Tuple[str, str, str]] = "stacked",
        backend: Union[str, Tuple[str, str, str]] = "math",
        t_min=1e-4,
        t_max=4.0,
    ):
        super().__init__()

        if isinstance(batch_type, str):
            batch_type = (batch_type, batch_type, batch_type)
        assert len(batch_type) == 3
        self.batch_type = batch_type

        if isinstance(backend, str):
            backend = (backend, backend, backend)
        assert len(backend) == 3
        self.backend = backend

        self.rotary_emb = RotaryEmbedding(dim_head, t_min, t_max)

        self.dropout = nn.Dropout(p=lin_dropout)

        # Encoding transformer (q-latent, kv-input spikes)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=context_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
            batch_type=batch_type[0],
            backend=backend[0],
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
                            batch_type=batch_type[1],
                            backend=backend[1],
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
            batch_type=batch_type[2],
            backend=backend[2],
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dim = dim

    def forward(
        self,
        *,  # (   stacked  ) or (   chained   )
        inputs,  # (B, N_in, dim) or (N_all_in, dim)
        latents,  # (B, N_latent, dim) or (N_all_latent, dim)
        output_queries,  # (B, N_out, dim) or (N_all_out, dim)
        input_timestamps,  # (B, N_in) or (N_all_in,)
        latent_timestamps,  # (B, N_latent) or (N_all_latent,)
        output_query_timestamps,  # (B, N_out) or (N_all_out,)
        input_mask=None,  # (B, N_in) or None
        input_seqlen=None,  # None or (B,)
        latent_mask=None,  # (B, N_latent) or None
        latent_seqlen=None,  # None or (B,)
        output_query_seqlen=None,  # None or (B,)
    ) -> Union[
        TensorType["batch", "*nqueries", "dim"],  # if padded
        TensorType["ntotal_queries", "dim"],  # if chained
    ]:
        if latent_mask is not None:
            raise NotImplementedError("latent_mask is not supported yet.")

        # compute timestamp embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_query_timestamps)

        # make sure arguments make sense
        if self.batch_type[0] == "stacked":
            assert inputs.dim() == 3, (
                f"Expected stacked inputs with 3 dimensions (batch, num_tokens, dim), "
                f"got ({inputs.shape})."
            )
            assert latents.dim() == 3, (
                f"Expected stacked latents with 3 dimensions (batch, num_tokens, dim), "
                f"got ({latents.shape})."
            )
            assert (
                input_seqlen is None
            ), f"input_seqlen should be None as it will not be used."
        elif self.batch_type[0] == "chained":
            assert inputs.dim() == 2, (
                f"Expected chained inputs with 2 dimensions (num_tokens, dim), "
                f"got ({inputs.shape})."
            )
            assert latents.dim() == 2, (
                f"Expected chained latents with 2 dimensions (num_tokens, dim), "
                f"got ({latents.shape})."
            )
            assert (
                input_mask is None
            ), f"input_mask should be None as it will not be used."
            assert input_seqlen is not None, f"input_seqlen should be provided."
            assert latent_seqlen is not None, f"latent_seqlen should be provided."

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

        # reshape latents if needed
        if self.batch_type[0] == "stacked" and self.batch_type[1] == "chained":
            # (b n d) -> ((b n) d)
            latents = latents.view(-1, self.dim)
            latent_timestamp_emb = latent_timestamp_emb.view(-1, self.dim)
        elif self.batch_type[0] == "chained" and self.batch_type[1] == "stacked":
            # ((b n) d) -> (b n d)
            # assert all elements in latent_seqlen are the same
            assert latent_seqlen is not None
            if len(set(latent_seqlen.tolist())) != 1:
                raise NotImplementedError(
                    "Expected all latent sequences in the batch to have the same "
                    "length. Moving from chained to stacked is not supported yet."
                    f"Got {latent_seqlen}."
                )
            latents = latents.view(len(latent_seqlen), latent_seqlen[0], self.dim)
            latent_timestamp_emb = latent_timestamp_emb.view(
                len(latent_seqlen), latent_seqlen[0], self.dim
            )

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(
                self_attn(latents, latent_timestamp_emb, x_seqlen=latent_seqlen)
            )
            latents = latents + self.dropout(self_ff(latents))

        if output_queries is None:
            return latents

        if self.batch_type[1] == "stacked" and self.batch_type[2] == "chained":
            # (b n d) -> ((b n) d)
            latents = latents.view(-1, self.dim)
            latent_timestamp_emb = latent_timestamp_emb.view(-1, self.dim)
        elif self.batch_type[1] == "chained" and self.batch_type[2] == "stacked":
            # ((b n) d) -> (b n d)
            # assert all elements in output_query_seqlen are the same
            assert latent_seqlen is not None
            if len(set(latent_seqlen)) != 1:
                raise NotImplementedError(
                    "Expected all latent sequences in the batch to have the same "
                    "length. Moving from chained to stacked is not supported yet."
                )
            latents = latents.view(len(latent_seqlen), latent_seqlen[0], self.dim)
            latent_timestamp_emb = latent_timestamp_emb.view(
                len(latent_seqlen), latent_seqlen[0], self.dim
            )

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
