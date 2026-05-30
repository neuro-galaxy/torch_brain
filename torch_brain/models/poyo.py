from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    PerceiverRotary,
    compute_loss_or_metric,
)
from torch_brain.data import pad, chain, track_mask, track_batch
from torch_brain.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
)
from brainsets.taxonomy import Task, OutputType


class POYO(nn.Module):
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
        use_memory_efficient_attn=True,
    ):
        super().__init__()

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

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
        )

        # Output projections + loss
        self.readout = nn.Linear(dim, 2)

        self.dim = dim
        self.using_memory_efficient_attn = self.perceiver_io.using_memory_efficient_attn

    def forward(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, N_in)
        spike_timestamps,  # (B, N_in)
        spike_type,  # (B, N_in)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
        output_mask=None,
        output_values: Optional[Dict[str, torch.Tensor]] = None,
        output_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        Dict[str, TensorType["batch", "*nqueries", "*nchannelsout"]],
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:

        # input
        inputs = self.unit_emb(spike_unit_index) + self.spike_type_emb(spike_type)

        # latents
        latents = self.latent_emb(latent_index)

        # outputs
        output_queries = self.session_emb(session_index)

        # feed into perceiver
        output_latents = self.perceiver_io(
            inputs=inputs,
            latents=latents,
            output_queries=output_queries,
            input_timestamps=spike_timestamps,
            latent_timestamps=latent_timestamps,
            output_query_timestamps=output_timestamps,
            input_mask=input_mask,
            input_seqlen=input_seqlen,
            latent_seqlen=latent_seqlen,
            output_query_seqlen=output_seqlen,
        )

        # readout layer
        output_pred = self.readout(output_latents)

        if self.using_memory_efficient_attn:
            loss = compute_loss_or_metric(
                "mse", OutputType.CONTINUOUS, output_pred, output_values, output_weights
            )
        else:
            assert output_mask is not None
            loss = compute_loss_or_metric(
                "mse",
                OutputType.CONTINUOUS,
                output_pred[output_mask],
                output_values,
                output_weights,
            )

        output = []
        if self.using_memory_efficient_attn:
            batch_size = output_batch_index.max().item() + 1
            for i in range(batch_size):
                output.append(output[output_batch_index == i])
        else:
            batch_size = output_latents.shape[0]
            for i in range(batch_size):
                output.append(output[i, output_mask[i]])

        return output, loss


class POYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO1 model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
    """

    def __init__(
        self,
        unit_tokenizer,
        session_tokenizer,
        latent_step,
        num_latents_per_step,
        using_memory_efficient_attn: bool = True,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step

        self.using_memory_efficient_attn = using_memory_efficient_attn
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

        output_timestamps = data.cursor.timestamps
        output_values = data.cursor.vel
        output_subtask_index = data.cursor.subtask_index

        # compute weights
        weight = data.config["reach_decoder"].get("weight", 1.0)
        subtask_weights = data.config["reach_decoder"].get("subtask_weights", {})
        num_subtasks = Task.REACHING.max_value()
        subtask_weight_map = np.ones(num_subtasks, dtype=np.float32)
        for subtask, subtask_weight in subtask_weights.items():
            subtask_weight_map[Task.from_string(subtask).value] = subtask_weight
        subtask_weight_map *= weight
        output_weights = subtask_weight_map[output_subtask_index]

        if not self.using_memory_efficient_attn:
            # Padding
            batch = {
                # input sequence
                "spike_unit_index": pad(spike_unit_index),
                "spike_timestamps": pad(spike_timestamps),
                "spike_type": pad(spike_token_type_index),
                "input_mask": track_mask(spike_unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # output sequence
                "session_index": pad(np.repeat(session_index, len(output_timestamps))),
                "output_timestamps": pad(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }
        else:
            # Chaining
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
                "session_index": chain(
                    np.repeat(session_index, len(output_timestamps))
                ),
                "output_timestamps": chain(output_timestamps),
                "output_seqlen": len(output_timestamps),
                "output_batch_index": track_batch(output_timestamps),
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
            }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(output_subtask_index)

        return batch
