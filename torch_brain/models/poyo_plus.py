from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from torchtyping import TensorType

from torch_brain.data import chain, pad, track_mask
from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryEmbedding,
    prepare_for_multitask_readout,
)
from torch_brain.registry import ModalitySpec

from torch_brain.utils import (
    create_linspace_latent_tokens,
    create_start_end_unit_tokens,
)


class POYOPlus(nn.Module):
    """POYO+ model from `"Multi-session, multi-task neural decoding from distinct
    cell-types and brain regions" <https://arxiv.org/abs/2409.15666>`_.

    POYO+ is a transformer-based model for neural decoding from population recordings.
    It extends the POYO architecture with multiple task-specific decoders.

    The model processes neural spike sequences through the following steps:

    1. Input tokens are constructed by combining unit embeddings, token type embeddings,
        and time embeddings for each spike in the sequence.
    2. The input sequence is compressed using cross-attention, where learnable latent
        tokens (each with an associated timestamp) attend to the input tokens.
    3. The compressed latent token representations undergo further refinement through
        multiple self-attention processing layers.
    4. Query tokens are constructed for the desired outputs by combining task embeddings,
        session embeddings, and output timestamps.
    5. These query tokens attend to the processed latent representations through
        cross-attention, producing outputs in the model's dimensional space (dim).
    6. Finally, task-specific linear layers map the outputs from the model dimension
        to the appropriate output dimension required by each task.

    Args:
        dim: Dimension of all embeddings
        dim_head: Dimension of each attention head
        num_latents: Number of unique latent tokens
        depth: Number of processing layers
        cross_heads: Number of attention heads used in a cross-attention layer
        self_heads: Number of attention heads used in a self-attention layer
        ffn_dropout: Dropout rate for feed-forward networks
        lin_dropout: Dropout rate for linear layers
        atn_dropout: Dropout rate for attention
        emb_init_scale: Scale for embedding initialization
        t_min: Minimum timestamp resolution for rotary embeddings
        t_max: Maximum timestamp resolution for rotary embeddings
        readout_specs: Specifications for each prediction task
    """

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
        t_min=1e-4,
        t_max=4.0,
        readout_specs: Dict[str, ModalitySpec],
    ):
        super().__init__()

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(len(readout_specs), dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryEmbedding(dim_head, t_min, t_max)

        self.dropout = nn.Dropout(p=lin_dropout)

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
            dim=dim,
            readout_specs=readout_specs,
        )

        self.dim = dim

    def forward(
        self,
        *,
        # input sequence
        input_unit_index: TensorType["batch", "n_in", int],
        input_timestamps: TensorType["batch", "n_in", float],
        input_token_type: TensorType["batch", "n_in", int],
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        # latent sequence
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        # output sequence
        output_session_index: TensorType["batch", "n_out", int],
        output_timestamps: TensorType["batch", "n_out", float],
        output_decoder_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ) -> Tuple[List[Dict[str, TensorType["*nqueries", "*nchannelsout"]]]]:
        """Forward pass of the POYO+ model.

        The model processes input spike sequences through its encoder-processor-decoder
        architecture to generate task-specific predictions.

        Args:
            input_unit_index: Indices of input units
            input_timestamps: Timestamps of input spikes
            input_token_type: Type of input tokens
            input_mask: Mask for input sequence
            latent_index: Indices for latent tokens
            latent_timestamps: Timestamps for latent tokens
            session_index: Index of the recording session
            output_timestamps: Timestamps for output predictions
            output_decoder_index: Indices indicating which decoder to use
            output_batch_index: Optional batch indices for outputs
            output_values: Ground truth values for supervised training
            output_weights: Optional weights for loss computation

        Returns:
            Tuple containing:
            - A list of dictionaries, each containing the predicted outputs for a
                given task
            - Total loss value
            - Dictionary of per-task losses
        """

        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        # input
        inputs = self.unit_emb(input_unit_index) + self.token_type_emb(input_token_type)
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # latents
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # outputs
        output_queries = self.session_emb(output_session_index) + self.task_emb(
            output_decoder_index
        )
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)

        # multitask readout layer, each task has a seperate linear readout layer
        output = self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output


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
        sequence_length=1.0,
        eval=False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.decoder_registry = decoder_registry

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.sequence_length = sequence_length

        self.eval = eval

    def __call__(self, data):
        # context window
        start, end = 0, self.sequence_length

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
            "input_unit_index": pad(spike_unit_index),
            "input_timestamps": pad(spike_timestamps),
            "input_token_type": pad(spike_token_type_index),
            "input_mask": track_mask(spike_unit_index),
            # latent sequence
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            # output sequence
            "output_session_index": pad(session_index),
            "output_timestamps": pad(output_timestamps),
            "output_decoder_index": pad(output_task_index),
            # ground truth targets
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
        }

        if self.eval:
            # we will add a few more fields needed for evaluation
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start
            batch["output_subtask_index"] = chain(
                output_subtask_index, allow_missing_keys=True
            )

        return batch
