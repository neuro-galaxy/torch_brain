from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import chain, pad8, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
    prepare_for_multitask_readout,
)
from torch_brain.registry import ModalitySpec, MODALITY_REGISTRY

from torch_brain.utils import (
    create_linspace_latent_tokens,
    create_start_end_unit_tokens,
)

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model."""

    def __init__(
        self, 
        *,
        input_size, 
        hidden_sizes,
        sequence_length: float,
        readout_specs: Dict[str, ModalitySpec] = MODALITY_REGISTRY,
        activation=nn.ReLU,
    ):

        super(MLP, self).__init__()

        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation())
            in_size = hidden_size
        
        
        self.net = nn.Sequential(*layers)
        self.readout = MultitaskReadout(
            dim=hidden_sizes[-1],
            readout_specs=readout_specs,
        )
        self.readout_specs = readout_specs
        self.sequence_length = sequence_length
        
    def forward(
        self,
        *,
        input_bins: TensorType["batch", "n_in"],
        output_decoder_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ):
        # import pdb; pdb.set_trace()
        output_embs = self.net(input_bins)

        output = self.readout(
            output_embs=output_embs,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output
    
    def tokenize(self, data: Data) -> Dict:
        start, end = 0, self.sequence_length
        # compute number of bins and bin indices
        bin_size = 0.02
        t_start = start
        t_end = end
        num_neurons = 96
        n_bins = int(np.ceil((t_end - t_start) / bin_size))
        timestamps = data.spikes.timestamps
        neuron_ids = data.spikes.unit_index
        dtype = np.float32
        # map timestamps to bin indices (0..n_bins-1). Clamp boundary cases.
        bin_idx = np.floor((timestamps - t_start) / bin_size).astype(int)
        # Handle timestamps exactly == t_end -> put in last bin
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        # validate neuron ids (clip or raise depending on preference)
        if not np.issubdtype(neuron_ids.dtype, np.integer):
            # try convert
            neuron_ids = neuron_ids.astype(int)

        if np.any((neuron_ids < 0) | (neuron_ids >= num_neurons)):
            raise ValueError(f"neuron_ids must be in [0, {num_neurons-1}]")

        # vectorized accumulation using bincount on flattened indices
        flat_idx = neuron_ids.astype(np.int64) * n_bins + bin_idx.astype(np.int64)
        counts_flat = np.bincount(flat_idx, minlength=num_neurons * n_bins).astype(np.float32)
        # counts = counts_flat.reshape((num_neurons, n_bins)).astype(dtype, copy=False)
        
        # Avery: Don't love this either
        readout_id = self.readout_specs[data.config["multitask_readout"][0]["readout_id"]]["id"]

        assert len(data.finger.vel) == 50
        data_dict = {
            "model_inputs": {
                "input_bins": pad8(counts_flat),
                # create a per-output scalar readout index (one id per timestep),
                # not one per-channel (data.finger.vel may be 2D: timesteps x channels)
                # "output_decoder_index": pad8(np.full((len(data.finger.vel),), 4, dtype=int)),
                "output_decoder_index": np.full((len(data.finger.vel),), 4, dtype=int),
            },
            "target_values": chain(data.finger.vel),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
        }

        return data_dict