import torch 
import torch.nn as nn
from torchtyping import TensorType

from torch_brain.nn import (
    MultitaskReadout,
    prepare_for_multitask_readout,
)

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model."""

    def __init__(
        self, 
        *,
        input_size, 
        hidden_sizes,
        readout_specs, 
        activation=nn.ReLU
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
        
    def forward(
        self,
        *,
        x,
        output_decoder_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ):
        output_embs = self.net(x)
        if output_embs.dim() == 2:
            output_embs = output_embs.unsqueeze(1)  # Add sequence dimension if missing
        output = self.readout(
            output_embs=output_embs,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output