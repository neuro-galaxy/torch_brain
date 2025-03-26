from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging

import numpy as np
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import chain, pad8, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryEmbedding,
)
from torch_brain.registry import ModalitySpec

from torch_brain.utils import (
    create_linspace_latent_tokens,
    create_start_end_unit_tokens,
    prepare_for_readout,
)


class POYO(nn.Module):
    """POYO model from `Azabou et al. 2023, A Unified, Scalable Framework for Neural Population Decoding
    <https://arxiv.org/abs/2310.16046>`_.

    POYO is a transformer-based model for neural decoding from electrophysiological
    recordings.

    1. Input tokens are constructed by combining unit embeddings, token type embeddings,
        and time embeddings for each spike in the sequence.
    2. The input sequence is compressed using cross-attention, where learnable latent
        tokens (each with an associated timestamp) attend to the input tokens.
    3. The compressed latent token representations undergo further refinement through
        multiple self-attention processing layers.
    4. Query tokens are constructed for the desired outputs by combining session
        embeddings, and output timestamps.
    5. These query tokens attend to the processed latent representations through
        cross-attention, producing outputs in the model's dimensional space (dim).
    6. Finally, a task-specific linear layer maps the outputs from the model dimension
        to the appropriate output dimension.

    Args:
        sequence_length: Maximum duration of the input spike sequence (in seconds)
        readout_spec: A :class:`torch_brain.registry.ModalitySpec` specifying readout properties
        latent_step: Timestep of the latent grid (in seconds)
        num_latents_per_step: Number of unique latent tokens (repeated at every latent step)
        dim: Hidden dimension of the model
        depth: Number of processing layers (self-attentions in the latent space)
        dim_head: Dimension of each attention head
        cross_heads: Number of attention heads used in a cross-attention layer
        self_heads: Number of attention heads used in a self-attention layer
        ffn_dropout: Dropout rate for feed-forward networks
        lin_dropout: Dropout rate for linear layers
        atn_dropout: Dropout rate for attention
        emb_init_scale: Scale for embedding initialization
        t_min: Minimum timestamp resolution for rotary embeddings
        t_max: Maximum timestamp resolution for rotary embeddings
    """

    def __init__(
        self,
        *,
        sequence_length: float,
        readout_spec: ModalitySpec,
        latent_step: float,
        num_latents_per_step: int = 64,
        dim: int = 512,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 4.0,
    ):
        super().__init__()

        self._validate_params(sequence_length, latent_step)

        self.sequence_length = sequence_length
        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.readout_spec = readout_spec

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(
            num_latents_per_step, dim, init_scale=emb_init_scale
        )
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
        self.readout = nn.Linear(dim, readout_spec.dim)

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
        output_mask: Optional[TensorType["batch", "n_out", bool]] = None,
        unpack_output: bool = False,
    ) -> Union[
        TensorType["batch", "n_out", "dim_out", float],
        List[TensorType[..., "dim_out", float]],
    ]:
        """Forward pass of the POYO model.

        The model processes input spike sequences through its encoder-processor-decoder
        architecture to generate task-specific predictions.

        Args:
            input_unit_index: Indices of input units
            input_timestamps: Timestamps of input spikes
            input_token_type: Type of input tokens
            input_mask: Mask for input sequence
            latent_index: Indices for latent tokens
            latent_timestamps: Timestamps for latent tokens
            output_session_index: Index of the recording session
            output_timestamps: Timestamps for output predictions
            output_mask: A mask of the same size as output_timestamps. True implies
                that particular timestamp is a valid query for POYO. This is required
                iff `unpack_output` is set to True.
            unpack_output: If False, this function will return a padded tensor of
                shape (batch size, num of max output queries in batch, `dim_out`).
                In this case you have to use `output_mask` externally to only look
                at valid outputs. If True, this will return a list of Tensors:
                the length of the list is equal to batch size, the shape of
                i^th Tensor is (num of valid output queries for i^th sample, `d_out`).

        Returns:
            A :class:`torch.Tensor` of shape `(batch, n_out, dim_out)`
            containing the predicted outputs corresponding to `output_timestamps`.
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
        output_queries = self.session_emb(output_session_index)
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
        output = self.readout(output_latents)

        if unpack_output:
            output = [output[b][output_mask[b]] for b in range(output.size(0))]

        return output

    def tokenize(self, data: Data) -> Dict:
        r"""Tokenizer used to tokenize Data for the POYO model.

        This tokenizer can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.
        """

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
        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        ### prepare latents
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start,
            end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )

        output_timestamps, output_values, output_weights, eval_mask = (
            prepare_for_readout(data, self.readout_spec)
        )

        # create session index for output
        output_session_index = self.session_emb.tokenizer(data.session.id)
        output_session_index = np.repeat(output_session_index, len(output_timestamps))

        data_dict = {
            "model_inputs": {
                # input sequence (keys/values for the encoder)
                "input_unit_index": pad8(spike_unit_index),
                "input_timestamps": pad8(spike_timestamps),
                "input_token_type": pad8(spike_token_type_index),
                "input_mask": track_mask8(spike_unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # output query sequence (queries for the decoder)
                "output_session_index": pad8(output_session_index),
                "output_timestamps": pad8(output_timestamps),
                "output_mask": track_mask8(output_session_index),
            },
            # ground truth targets
            "target_values": pad8(output_values),
            "target_weights": pad8(output_weights),
            # extra data needed for evaluation
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": pad8(eval_mask),
        }

        return data_dict

    def _validate_params(self, sequence_length, latent_step):
        r"""Ensure: sequence_length, and latent_step are floating point numbers greater
        than zero. And sequence_length is a multiple of latent_step.
        """

        if not isinstance(sequence_length, float):
            raise ValueError("sequence_length must be a float")
        if not sequence_length > 0:
            raise ValueError("sequence_length must be greater than 0")
        self.sequence_length = sequence_length

        if not isinstance(latent_step, float):
            raise ValueError("latent_step must be a float")
        if not latent_step > 0:
            raise ValueError("latent_step must be greater than 0")
        self.latent_step = latent_step

        # check if sequence_length is a multiple of latent_step
        if abs(sequence_length % latent_step) > 1e-10:
            logging.warning(
                f"sequence_length ({sequence_length}) is not a multiple of latent_step "
                f"({latent_step}). This is a simple warning, and this behavior is allowed."
            )


def poyo_mp(readout_spec: ModalitySpec, ckpt_path=None):
    if ckpt_path is not None:
        raise NotImplementedError("Loading from checkpoint is not supported yet.")

    return POYO(
        sequence_length=1.0,
        latent_step=1.0 / 8,
        dim=64,
        readout_spec=readout_spec,
        dim_head=64,
        num_latents_per_step=16,
        depth=6,
        cross_heads=2,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.2,
        emb_init_scale=0.02,
        t_min=1e-4,
        t_max=4.0,
    )
