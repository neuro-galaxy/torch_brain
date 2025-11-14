import torch 
import torch.nn as nn
from torchtyping import TensorType

from torch_brain.nn import (
    MultitaskReadout,
    prepare_for_multitask_readout,
)

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
        nonlinearity="tanh"
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