from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

try:
    import xformers.ops as xops
except ImportError:
    xops = None


from brainsets.taxonomy import DecoderSpec, Decoder
from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    FeedForward,
    MultitaskReadout,
    prepare_for_multitask_readout,
)
from torch_brain.data import chain, track_batch
from torch_brain.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
)


class POYOPlusE(nn.Module):
    def __init__(
        self,
        *,
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
        task_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        if xops is None:
            raise ImportError(
                "xformers not installed, please install `xformers` to use the efficient "
                "version of POYO+, otherwise use the default version."
            )

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(
            Decoder.max_value() + 1, dim, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

        # encoder layer
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # process layers
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.Sequential(
                    RotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=True,
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                )
            )

        # decoder layer
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Output projections + loss
        self.readout = MultitaskReadout(
            latent_dim=dim,
            decoder_specs=task_specs,
            batch_type=self.batch_type[2],
        )

        self.dim = dim

    def forward(
        self,
        *,
        # input sequence
        input_unit_index,  # (total_N_in,)
        input_timestamps,  # (total_N_in,)
        input_token_type,  # (total_N_in,)
        input_seqlen,  # (B,)
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_decoder_index,  # (B, N_out)
        output_seqlen,
        output_batch_index,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        # input
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # latents
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # outputs
        output_queries = self.session_emb(session_index) + self.task_emb(
            output_decoder_index
        )
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # encode
        latents = latents + self.enc_atn.forward_varlen(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            query_seqlen=latent_seqlen,
            context_seqlen=input_seqlen,
        )
        latents = latents + self.enc_ffn(latents)

        # reshape latents and latent timestamp embeddings
        latents = latents.view(len(latent_seqlen), latent_seqlen[0], self.dim)
        latent_timestamp_emb = latent_timestamp_emb.view(
            len(latent_seqlen), latent_seqlen[0], self.dim
        )

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # reshape latents again
        latents = latents.view(-1, self.dim)
        latent_timestamp_emb = latent_timestamp_emb.view(-1, self.dim)

        # decode
        output_queries = output_queries + self.dec_atn.forward_varlen(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
            query_seqlen=output_seqlen,
            context_seqlen=latent_seqlen,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)

        # multitask readout layer, each task has a seperate linear readout layer
        output, loss, losses_taskwise = self.readout(
            output_latents=output_latents,
            output_decoder_index=output_decoder_index,
            output_batch_index=output_batch_index,
            output_values=output_values,
            output_weights=output_weights,
        )

        return output, loss, losses_taskwise


class POYOPlusETokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        decoder_registry (Dict): Registry of the decoders.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        decoder_registry,
        latent_step,
        num_latents_per_step,
        batch_type,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.decoder_registry = decoder_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.batch_type = batch_type
        self.eval = eval

    def __call__(self, data):
        # context window
        start, end = 0, 1.0  # data.domain, data.end

        ### prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # create start and end tokens for each unit
        (
            se_token_type_index,
            se_unit_index,
            se_timestamps,
        ) = create_start_end_unit_tokens(unit_ids, start, end)

        # append start and end tokens to the spike sequence
        spike_token_type_index = np.concatenate(
            [se_token_type_index, np.zeros_like(spike_unit_index)]
        )
        spike_unit_index = np.concatenate([se_unit_index, spike_unit_index])
        spike_timestamps = np.concatenate([se_timestamps, spike_timestamps])

        # unit_index is relative to the recording, so we want it to map it to
        # the global unit index
        local_to_global_map = np.array(self.unit_tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        ### prepare outputs
        session_index = self.session_tokenizer(data.session)

        (
            output_timestamps,
            output_task_index,
            output_values,
            output_weights,
            output_subtask_index,
        ) = prepare_for_multitask_readout(
            data,
            self.decoder_registry,
        )

        session_index = np.repeat(session_index, len(output_timestamps))

        batch = {
            # input sequence
            "spike_unit_index": chain(spike_unit_index),
            "spike_timestamps": chain(spike_timestamps),
            "spike_type": chain(spike_token_type_index),
            "input_seqlen": len(spike_unit_index),
            # latent sequence
            "latent_index": chain(latent_index),
            "latent_timestamps": chain(latent_timestamps),
            "latent_seqlen": len(latent_index),
            # output sequence
            "session_index": chain(session_index),
            "output_timestamps": chain(output_timestamps),
            "output_decoder_index": chain(output_task_index),
            "output_seqlen": len(output_timestamps),
            "output_batch_index": track_batch(output_timestamps),
            "output_values": chain(output_values, allow_missing_keys=True),
            "output_weights": chain(output_weights, allow_missing_keys=True),
        }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(
                output_subtask_index, allow_missing_keys=True
            )

        return batch
