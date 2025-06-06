from typing import Union
import torch
from torch import nn, Tensor
from einops import repeat, rearrange


class SinusoidalTimeEmbedding(nn.Module):
    r"""Sinusoidal time/position embedding layer.
    These embeddings are generally added/concatenated to tokens to give
    them a sense of time/position.
    The timeperiods are logarithmically spaced between ``t_min`` and ``t_max``
    (both inclusive).

    Args:
        dim (int): The dimension of the embedding needed (must be a multiple of 2)
        t_min (float): Minimum period of the sinusoids. Set this to the smallest
            timescale you care about.
        t_max (float): Maximum period of the sinusoids. Set this to the largest
            timescale you care about.
    """

    omega: Tensor

    def __init__(self, dim: int, t_min: float, t_max: float):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("`dim` must be a multiple of 2")

        periods = generate_logspace_timeperiods(dim // 2, t_min, t_max)
        omega = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Convert raw timestamps to sinusoidal embeddings

        Args:
            timestamps (torch.Tensor): timestamps tensor
        """
        angles = timestamps[..., None] * self.omega
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class RotaryTimeEmbedding(nn.Module):
    r"""Rotary time/positional embedding layer. This module is designed to be used with
    :class:`torch_brain.nn.RotarySelfAttention` and :class:`torch_brain.nn.RotaryCrossAttention` to
    modulate the attention weights in accordance with relative timing/positions of the tokens.
    Original paper: `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_

    The timeperiods are computed using :func:`generate_logspace_timeperiods`.

    Args:
        head_dim (int): Dimension of the attention head.
        rotate_dim (int): Number of dimensions to rotate. You can choose to rotate only a
            small portion of the head dimension using this parameter.
            E.g. `PerceiverIO <https://arxiv.org/abs/2107.14795>`_ found rotating only half
            dimensions to be effective.
        t_min (float): Minimum period of the sinusoids. Set this to the smallest
            timescale the attention layer should care about.
        t_max (float): Maximum period of the sinusoids. Set this to the largest
            timescale the attention layer should care about.
    """

    omega: Tensor

    def __init__(self, head_dim: int, rotate_dim: int, t_min: float, t_max: float):
        super().__init__()

        if rotate_dim % 2 != 0:
            raise ValueError("rotate_dim must be a multiple of 2")

        if not head_dim >= rotate_dim:
            raise ValueError("head_dim must be equal to or larger than rotate_dim")

        periods = generate_logspace_timeperiods(rotate_dim // 2, t_min, t_max)
        omega = torch.zeros(head_dim // 2)
        omega[: rotate_dim // 2] = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Computes the rotary embeddings for given timestamps,
        which can then be used by :meth:`RotaryTimeEmbedding.rotate`.

        Args:
            timestamps (torch.Tensor): timestamps tensor.
        """
        angles = torch.einsum("..., f -> ... f", timestamps, self.omega)
        angles = repeat(angles, "... n -> ... (n r)", r=2)
        rotary_emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return rotary_emb

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    @staticmethod
    def rotate(
        x: Tensor,
        rotary_emb: Tensor,
        unsqueeze_dim: int = 2,
    ) -> Tensor:
        r"""Apply the rotary positional embedding to the input data.

        Args:
            x (torch.Tensor): Input data.
            rotary_emb (torch.Tensor): The rotary embedding produced by a forward
                call of :class:`RotaryTimeEmbedding`.
            unsqueeze_dim (int, optional): Dimension where heads are located in the input tensor.
                E.g. For input shape (batch, heads, seq_len, dim) use 1.
                For input shape (batch, seq_len, heads, dim) use 2.
                Defaults to 2.
        """
        rotary_emb = rotary_emb.unsqueeze(unsqueeze_dim).to(x.dtype)
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)
        return (x * cos) + (RotaryTimeEmbedding._rotate_half(x) * sin)

    @staticmethod
    def invert(rotary_emb: Tensor) -> Tensor:
        r"""Invert/Negate rotary embedding. If the input embeddings correspond to a time
        :math:`t`, then the output embeddings correspond to time :math:`-t`.

        Args:
            rotary_emb (torch.Tensor): Embeddings produced by a forward call of
                :class:`RotaryTimeEmbedding`.
        """
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)
        return torch.cat((cos, -sin), dim=-1)


def generate_logspace_timeperiods(
    num: int,
    t_min: Union[float, Tensor],
    t_max: Union[float, Tensor],
) -> Tensor:
    r"""Generates ``num`` timeperiods that are logarithmically spaced between
    ``t_min`` and ``t_max`` (both inclusive).

    Args:
        num (int): number of timestamps needed
        t_min (float): smallest timeperiod
        t_max (float): largest timeperiod
    """
    if not 0 < t_min < t_max:
        raise ValueError(
            f"Invalid t_min ({t_min}) and t_max ({t_max}). They should follow 0 < t_min < t_max."
        )
    exponents = torch.linspace(0, 1.0, num, dtype=torch.float32)
    t_min, t_max = torch.tensor(t_min), torch.tensor(t_max)
    periods = torch.exp(torch.lerp(t_min.log(), t_max.log(), exponents))
    assert torch.isclose(periods[0], t_min)
    assert torch.isclose(periods[-1], t_max)
    return periods
