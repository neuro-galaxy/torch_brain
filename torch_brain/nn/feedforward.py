import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    """Gated Gaussian Error Linear Unit (GEGLU) activation function, as introduced in
    the paper "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """A feed-forward network with GEGLU activation.

    Args:
        dim (int): Input and output dimension
        mult (int, optional): Multiplier for hidden dimension. Defaults to 4
        dropout (float, optional): Dropout probability. Defaults to 0.2
    """

    def __init__(self, dim, mult=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)
