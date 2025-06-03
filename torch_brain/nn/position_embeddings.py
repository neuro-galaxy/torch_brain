from typing import Union
import torch
from torch import nn, Tensor
from einops import repeat, rearrange


class SinusoidalEmbedding(nn.Module):
    r"""Sinusoidal time/position embedding layer.
    These embeddings are generally added/concatenated to tokens to give
    them a sense of time/position.

    Args:
        dim (int): The dimension of the embedding needed (must be a multiple of 2)
        t_min (float): Smallest timescale
        t_max (float): Largest timescale
    """

    omega: Tensor

    def __init__(self, dim: int, t_min: float, t_max: float):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("`dim` must be a multiple of 2")

        periods = self.get_periods(dim // 2, t_min, t_max)
        omega = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        r"""Convert raw timestamps to `dim`-dimensional sinusoidal embeddings

        Args:
            timestamps (torch.Tensor): timestamps tensor
        """
        angles = timestamps[..., None] * self.omega
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @staticmethod
    def get_periods(num: int, t_min: Union[float, Tensor], t_max: Union[float, Tensor]):
        r"""Generates `num` timeperiods that are logarithmically spaced between
        `t_min` and `t_max`. Both `t_min` and `t_max` are included in the returned
        periods

        Args:
            num (int): number of timestamps needed
            t_min (float): smallest timeperiod
            t_max (float): largest timeperiod
        """
        exponents = torch.linspace(0, 1.0, num, dtype=torch.float32)
        t_min, t_max = torch.tensor(t_min), torch.tensor(t_max)
        periods = torch.exp(torch.lerp(t_min.log(), t_max.log(), exponents))
        assert torch.isclose(periods[0], t_min)
        assert torch.isclose(periods[-1], t_max)
        return periods


class RotaryEmbedding(nn.Module):
    r"""Rotary time/positional embedding layer. This layer is used in conjunction with
    `torch_brain.nn.RotarySelfAttention` and `torch_brain.nn.RotaryCrossAttention` to
    module the attention in accordance with relative timing/positions of the tokens.

    `Original paper <https://arxiv.org/abs/2104.09864>`

    Args:
        head_dim (int): Dimension of the attention head.
        rotate_dim (int): Number of dimensions to rotate. You can choose to rotate only a
            small portion of the head dimension using this parameter.
            E.g. [PerceiverIO](https://arxiv.org/abs/2107.14795) found rotating only half
            dimensions to be effective.
        t_min (float, optional): Minimum period of the sinusoids. Set this to the smallest
            timescale the attention layer should care about.
        t_max (float, optional): Maximum period of the sinusoids. Set this to the largest
            timescale the attention layer should care about.
    """

    omega: Tensor

    def __init__(self, head_dim: int, rotate_dim: int, t_min: float, t_max: float):
        super().__init__()

        if rotate_dim % 2 != 0:
            raise ValueError("rotate_dim must be a multiple of 2")

        if not head_dim >= rotate_dim:
            raise ValueError("head_dim must be equal to or larger than rotate_dim")

        periods = SinusoidalEmbedding.get_periods(rotate_dim // 2, t_min, t_max)
        omega = torch.zeros(head_dim // 2)
        omega[: rotate_dim // 2] = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        r"""Computes the rotary embeddings for given timestamps,
        which can then be used by `RotaryEmbedding.apply_rotary_emb`.

        Args:
            timestamps (torch.Tensor): timestamps tensor.
        """
        angles = timestamps.unsqueeze(-1) * self.omega
        angles = repeat(angles, "... n -> ... (n r)", r=2)
        pos_emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return pos_emb

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")

    @classmethod
    def apply_rotary_emb(
        cls,
        rotary_emb: Tensor,
        x: Tensor,
        head_dim: int = 2,
    ) -> Tensor:
        r"""Apply the rotary positional embedding to the input data.

        Args:
            rotary_emb (torch.Tensor): The rotary embedding produced by a forward
                call of `RotaryEmbedding`.
            x (torch.Tensor): Input data.
            head_dim (int, optional): Dimension of the head. Defaults to 2.
        """
        rotary_emb = rotary_emb.unsqueeze(head_dim).to(x.dtype)
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)
        return (x * cos) + (cls._rotate_half(x) * sin)

    @staticmethod
    def invert_rotary_emb(rotary_emb: Tensor) -> Tensor:
        r"""Invert/Negate rotary embedding. If the input embeddings correspond to a time `t`,
        then the output embeddings correspond to time `-t`.

        Args:
            rotary_emb (torch.Tensor): Embeddings produced by a forward call of
                `RotaryEmbedding`.
        """
        cos, sin = rotary_emb.chunk(chunks=2, dim=-1)
        return torch.cat((cos, -sin), dim=-1)
