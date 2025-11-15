from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from einops import rearrange
from temporaldata import Data
from torch_brain.nn import (
    MultitaskReadout,
    prepare_for_multitask_readout,
)
from torch_brain.data import chain, pad8


class RNN(nn.Module):
    """A simple Recurrent Neural Network (RNN) model."""

    def __init__(
        self,
        *,
        input_size,
        hidden_size,
        readout_specs,
        num_layers=1,
        rnn_type="simple",
        nonlinearity="tanh",
        sequence_length=1.0
    ):
        super(RNN, self).__init__()
        if rnn_type.lower() == "simple":
            self.net = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity=nonlinearity)
        elif rnn_type.lower() == "lstm":
            self.net = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type.lower() == "gru":
            self.net = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.readout = MultitaskReadout(
            dim=hidden_size,
            readout_specs=readout_specs,
        )
        self.sequence_length = sequence_length
        self.readout_specs = readout_specs

    def forward(
        self,
        *,
        x,
        h=None,
        output_decoder_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ):
        output_embs, h = self.net(x, h)
        if output_embs.dim() == 2:
            output_embs = output_embs.unsqueeze(1)  # Add sequence dimension if missing
        out = self.readout(
            output_embs=output_embs,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )
        return out, h

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
        counts = counts_flat.reshape((num_neurons, n_bins)).astype(dtype, copy=False)
        counts = rearrange(counts, 'n_neurons n_bins -> n_bins n_neurons')

        # Avery: Don't love this either
        readout_id = self.readout_specs[data.config["multitask_readout"][0]["readout_id"]]["id"]
        # TODO this needs to be reconstructed from evaluation mask intervals or timeseries.
        weights = np.ones(len(data.finger.vel), dtype=np.float32)
        # assert len(data.finger.vel) == 50
        # if data.finger.vel.shape[0] > 50:
            # breakpoint()
            # data.finger = data.finger[:50]
            # raise ValueError(f"data.finger.vel.shape[0] < 50: {data.finger.vel.shape[0]}")
            # Pad to 50, unclear why this is happening
            # data.finger.vel = np.pad(
                # data.finger.vel, ((0, 50 - data.finger.vel.shape[0]), (0, 0)), mode='constant'
            # )

        data_dict = {
            "model_inputs": {
                "x": counts[:50],
                # create a per-output scalar readout index (one id per timestep),
                # not one per-channel (data.finger.vel may be 2D: timesteps x channels)
                # "output_decoder_index": pad8(np.full((len(data.finger.vel),), 4, dtype=int)),
                # "output_decoder_index": readout_id,
                # "output_decoder_index": np.array([readout_id], dtype=int), # 1D crashes multitask forward, 0D crashes loss masking in train loop, prefer latter.
                "output_decoder_index": np.full((len(data.finger.vel[:50]),), readout_id, dtype=int),
            },
            "target_timestamps": data.finger.timestamps[:50],
            # "target_weights": weights,
            # "target_weights": chain(weights),
            "target_values": data.finger.vel[:50], # JY: Chain works opposite as advertised, stacking is by default?
            # "target_values": chain(data.finger.vel),
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
        }

        return data_dict