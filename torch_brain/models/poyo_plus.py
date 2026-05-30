from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from brainsets.taxonomy import DecoderSpec, Decoder
from torch_brain.nn import (
    Embedding,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    PerceiverRotary,
    prepare_for_multitask_readout,
)
from torch_brain.data import pad, chain, track_mask, track_batch
from torch_brain.utils import (
    create_start_end_unit_tokens,
    create_linspace_latent_tokens,
)


BACKEND_CONFIGS = {
    "cpu": (("stacked", "stacked", "stacked"), ("math", "math", "math")),
    "gpu_fp32": (
        ("stacked", "stacked", "stacked"),
        ("math", "mem_efficient", "math"),
    ),
    "gpu_fp32_var": (  # assumes that attn_dropout is 0.0 for cross attends
        ("chained", "stacked", "chained"),
        ("mem_efficient", "mem_efficient", "mem_efficient"),
    ),
    "gpu_fp16": (("chained", "chained", "chained"), ("flash", "flash", "flash")),
}


class POYOPlus(nn.Module):
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
        backend_config="gpu_fp32",
        task_specs: Dict[str, DecoderSpec],
    ):
        super().__init__()

        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.spike_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(
            Decoder.max_value() + 1, dim, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)

        # determine backend
        if backend_config not in BACKEND_CONFIGS.keys():
            raise ValueError(
                f"Invalid backend config: {backend_config}, must be one of"
                f" {list(BACKEND_CONFIGS.keys())}"
            )

        self.batch_type = BACKEND_CONFIGS[backend_config][0]

        # TODO try and catch error to provide more helpful error message
        self.perceiver_io = PerceiverRotary(
            dim=dim,
            dim_head=dim_head,
            depth=depth,
            cross_heads=cross_heads,
            self_heads=self_heads,
            ffn_dropout=ffn_dropout,
            lin_dropout=lin_dropout,
            atn_dropout=atn_dropout,
            batch_type=self.batch_type,
            backend=BACKEND_CONFIGS[backend_config][1],
        )

        # Output projections + loss
        self.readout = MultitaskReadout(
            latent_dim=dim,
            decoder_specs=task_specs,
            batch_type=self.batch_type[2],
        )

        self.dim = dim

    def freeze_middle(self) -> List[nn.Module]:
        # Freeze everything except the readout, unit embedding, and session embedding
        # layers.
        frozen_modules = []
        banned_modules = [
            self.unit_emb,
            self.session_emb,
        ]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)

        return frozen_modules

    def unfreeze_middle(self) -> None:
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True

    def forward(
        self,
        *,
        # input sequence
        spike_unit_index,  # (B, N_in) or (total_N_in,)
        spike_timestamps,  # (B, N_in) or (total_N_in,)
        spike_type,  # (B, N_in) or (total_N_in,)
        input_mask=None,  # (B, N_in)
        input_seqlen=None,  # (B,)
        # latent sequence
        latent_index,  # (B, N_latent)
        latent_timestamps,  # (B, N_latent)
        latent_seqlen=None,
        # output sequence
        session_index,  # (B,)
        output_timestamps,  # (B, N_out)
        output_decoder_index,  # (B, N_out)
        output_seqlen=None,
        output_batch_index=None,
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
        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            session_index
        )

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

        # Readout layer
        output, loss, losses_taskwise = self.readout(
            output_latents=output_latents,
            output_decoder_index=output_decoder_index,
            output_batch_index=output_batch_index,
            output_values=output_values,
            output_weights=output_weights,
        )

        return output, loss, losses_taskwise


class POYOPlusTokenizer:
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

        batch = {}
        if self.batch_type[0] == "stacked":
            batch = {
                **batch,
                # input sequence
                "spike_unit_index": pad(spike_unit_index),
                "spike_timestamps": pad(spike_timestamps),
                "spike_type": pad(spike_token_type_index),
                "input_mask": track_mask(spike_unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
            }
        else:
            batch = {
                **batch,
                # input sequence
                "spike_unit_index": chain(spike_unit_index),
                "spike_timestamps": chain(spike_timestamps),
                "spike_type": chain(spike_token_type_index),
                "input_seqlen": len(spike_unit_index),
                # latent sequence
                "latent_index": chain(latent_index),
                "latent_timestamps": chain(latent_timestamps),
                "latent_seqlen": len(latent_index),
            }
        if self.batch_type[1] == "chained":
            batch["latent_seqlen"] = len(latent_index)

        if self.batch_type[2] == "stacked":
            batch = {
                **batch,
                # output sequence
                "session_index": pad(session_index),
                "output_timestamps": pad(output_timestamps),
                "output_decoder_index": pad(output_task_index),
                "output_values": chain(output_values, allow_missing_keys=True),
                "output_weights": chain(output_weights, allow_missing_keys=True),
            }
        else:
            batch = {
                **batch,
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
