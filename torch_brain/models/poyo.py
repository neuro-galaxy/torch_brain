from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data, Interval

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
    resolve_weights_based_on_interval_membership,
)


class POYO(nn.Module):
    """POYO model from `"A Unified, Scalable Framework for Neural Population Decoding"
    <https://arxiv.org/abs/2310.16046>`.

    POYO is a transformer-based model for neural decoding from electrophysiological
    recordings.

    TODO: Document the model architecture

    Args:
        dim: Hidden dimension of the model
        dim_out: Dimension of the decoding targets
        dim_head: Dimension of each attention head
        num_latents: Number of unique latent tokens (repeated at every latent step)
        depth: Number of processing layers (self-attentions in the latent space)
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
        dim=512,
        dim_out=2,
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
    ):
        super().__init__()

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
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
        self.readout = nn.Linear(dim, dim_out)

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
    ) -> TensorType["batch", "n_out", "dim_out", float]:
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
            output_session_index: Index of the recording session
            output_timestamps: Timestamps for output predictions

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

        return output


def poyo_mp(dim_out, ckpt_path=None):
    if ckpt_path is not None:
        raise NotImplementedError("Loading from checkpoint is not supported yet.")

    return POYO(
        dim=64,
        dim_out=dim_out,
        dim_head=64,
        num_latents=16,
        depth=6,
        cross_heads=2,
        self_heads=8,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.2,
    )


class POYOTokenizer:
    r"""Tokenizer used to tokenize Data for the POYO model.

    This tokenizer can be called as a transform. If you are applying multiple
    transforms, make sure to apply this one last.

    Args:
        unit_tokenizer (Callable): Tokenizer for the units.
        session_tokenizer (Callable): Tokenizer for the sessions.
        decoder_registry (Dict): Registry of the decoders.
        weight_registry (Dict): Registry of the weights.
        latent_step (float): Step size for generating latent tokens.
        num_latents_per_step (int): Number of latents per step.
        subtask_weights (List[float]): Loss-weights for different subtasks.
    """

    def __init__(
        self,
        unit_tokenizer: Callable,
        session_tokenizer: Callable,
        latent_step: float,
        num_latents_per_step: int,
        modality_spec: ModalitySpec,
        sequence_length: float = 1.0,
        eval: bool = False,
    ):
        self.unit_tokenizer = unit_tokenizer
        self.session_tokenizer = session_tokenizer

        self.modality_spec = modality_spec

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.sequence_length = sequence_length
        self.eval = eval

    def __call__(self, data: Data) -> Dict:
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

        ### prepare output queries and targets
        output_timestamps = data.get_nested_attribute(self.modality_spec.timestamp_key)
        output_values = data.get_nested_attribute(self.modality_spec.value_key)
        if output_values.dtype == np.float64:
            output_values = output_values.astype(np.float32)

        output_session_index = self.session_tokenizer(data.session)
        output_session_index = np.repeat(output_session_index, len(output_timestamps))

        # resolve weights
        output_weights = resolve_weights_based_on_interval_membership(
            output_timestamps, data, config=data.config.get("weights", None)
        )

        batch = {
            # input sequence
            "input_unit_index": pad8(spike_unit_index),
            "input_timestamps": pad8(spike_timestamps),
            "input_token_type": pad8(spike_token_type_index),
            "input_mask": track_mask8(spike_unit_index),
            # latent sequence
            "latent_index": latent_index,
            "latent_timestamps": latent_timestamps,
            # output sequence
            "output_session_index": pad8(output_session_index),
            "output_timestamps": pad8(output_timestamps),
            "output_mask": track_mask8(output_session_index),
            # ground truth targets
            "target_values": pad8(output_values),
            "target_weights": pad8(output_weights),
        }

        if self.eval:
            batch["session_id"] = data.session
            batch["absolute_start"] = data.absolute_start

        return batch
