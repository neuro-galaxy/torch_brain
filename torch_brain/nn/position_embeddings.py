import torch
from torch import nn, Tensor


class SinusoidalTimeEmbedding(nn.Module):
    r"""Sinusoidal time/position embedding layer.
    These embeddings are generally added/concatenated to tokens to give
    them a sense of time/position.
    The timeperiods are logarithmically spaced between ``t_min`` and ``t_max``
    (both inclusive).

    Args:
        dim: The dimension of the embedding needed (must be a multiple of 2)
        t_min: Minimum period of the sinusoids. Set this to the smallest
            timescale you care about.
        t_max: Maximum period of the sinusoids. Set this to the largest
            timescale you care about.
    """

    omega: Tensor

    def __init__(self, dim: int, t_min: float, t_max: float):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("`dim` must be a multiple of 2")

        F = dim // 2
        periods = generate_logspace_timeperiods(F, t_min, t_max)  # (F,)
        omega = 2 * torch.pi / periods
        self.register_buffer("omega", omega)  # (F,)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Convert raw timestamps to sinusoidal embeddings

        Args:
            timestamps (torch.Tensor): timestamps tensor
        """
        angles = timestamps[..., None] * self.omega  # (...,) x (F,)-> (..., F)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)  # (..., D)


class RotaryTimeEmbedding(nn.Module):
    r"""Rotary time/positional embedding layer. This module is designed to be used with
    :class:`torch_brain.nn.RotarySelfAttention` and :class:`torch_brain.nn.RotaryCrossAttention` to
    modulate the attention weights in accordance with relative timing/positions of the tokens.
    Original paper: `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/abs/2104.09864>`_

    The timeperiods are computed using :func:`generate_logspace_timeperiods`.

    Args:
        head_dim: Dimension of the attention head.
        rotate_dim: Number of dimensions to rotate. You can choose to rotate only a
            small portion of the head dimension using this parameter.
            E.g. `PerceiverIO <https://arxiv.org/abs/2107.14795>`_ found rotating only half
            dimensions to be effective.
        t_min: Minimum period of the sinusoids. Set this to the smallest
            timescale the attention layer should care about.
        t_max: Maximum period of the sinusoids. Set this to the largest
            timescale the attention layer should care about.
    """

    omega: Tensor

    def __init__(self, head_dim: int, rotate_dim: int, t_min: float, t_max: float):
        super().__init__()

        if rotate_dim % 2 != 0:
            raise ValueError("rotate_dim must be a multiple of 2")

        if not head_dim >= rotate_dim:
            raise ValueError("head_dim must be equal to or larger than rotate_dim")

        F = rotate_dim // 2
        D = head_dim // 2
        periods = generate_logspace_timeperiods(F, t_min, t_max)  # (F,)
        omega = torch.zeros(D)
        omega[:F] = 2 * torch.pi / periods
        self.register_buffer("omega", omega)  # (D,)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Computes the rotary embeddings for given timestamps,
        which can then be used by :meth:`RotaryTimeEmbedding.rotate`.

        Args:
            timestamps: timestamps tensor.
        """

        angles = timestamps[..., None] * self.omega  # (...,) x (D,)-> (..., D)
        angles = angles.repeat_interleave(2, dim=-1)  # (..., D) -> (..., H)
        rotary_emb = torch.cat((angles.cos(), angles.sin()), dim=-1)  # (..., H*2)
        return rotary_emb

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x = x.unflatten(-1, sizes=(-1, 2))  # (..., H) -> (..., D, 2)
        x1, x2 = x.unbind(dim=-1)  # (..., D, 2) -> (..., D), (..., D)
        x = torch.stack((-x2, x1), dim=-1)  # (..., D, 2)
        return x.flatten(start_dim=-2)  # (..., D, 2) -> (..., H)

    @staticmethod
    def rotate(
        x: Tensor,
        rotary_emb: Tensor,
        unsqueeze_dim: int = 2,
    ) -> Tensor:
        r"""Apply the rotary positional embedding to the input data.

        Args:
            x: Input data.
            rotary_emb: The rotary embedding produced by a forward
                call of :class:`RotaryTimeEmbedding`.
            unsqueeze_dim: Dimension where heads are located in the input tensor.
                E.g. For input shape (batch, heads, seq_len, dim) use 1.
                For input shape (batch, seq_len, heads, dim) use 2.
                Defaults to 2.
        """
        rotary_emb = rotary_emb.unsqueeze(unsqueeze_dim)  # (..., H*2) -> (..., 1, H*2)
        rotary_emb = rotary_emb.to(x.dtype)
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)  # (..., 1, H), (..., 1, H)
        return (x * cos) + (RotaryTimeEmbedding._rotate_half(x) * sin)  # (..., H)

    @staticmethod
    def invert(rotary_emb: Tensor) -> Tensor:
        r"""Invert/Negate rotary embedding. If the input embeddings correspond to a time
        :math:`t`, then the output embeddings correspond to time :math:`-t`.

        Args:
            rotary_emb: Embeddings produced by a forward call of
                :class:`RotaryTimeEmbedding`.
        """
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)  # (..., H), (..., H)
        return torch.cat((cos, -sin), dim=-1)  # (..., H*2)


def generate_logspace_timeperiods(
    num: int,
    t_min: float | Tensor,
    t_max: float | Tensor,
) -> Tensor:
    r"""Generates ``num`` timeperiods that are logarithmically spaced between
    ``t_min`` and ``t_max`` (both inclusive).

    Args:
        num: number of timestamps needed
        t_min: smallest timeperiod
        t_max: largest timeperiod
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
