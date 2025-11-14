import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) model."""

    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(activation())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
