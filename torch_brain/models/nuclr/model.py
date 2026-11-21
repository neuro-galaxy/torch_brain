from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_brain.batching import chain, collate
from torch_brain.data import Data
from torch_brain.nn import RotaryTimeEmbedding
from torch_brain.samplers import SequentialFixedWindowSampler

from .attention import Attention


class NuCLR(nn.Module):
    r"""`NuCLR <https://arxiv.org/abs/2512.01199>`_ model.

    NuCLR produces an embedding for every unit from its *binned* spiking activity.
    Each unit's activity is split into ``latent_step``-long chunks and projected to
    a token. Temporal attention mixes the tokens of a unit across time, and is then
    interleaved with spatial attention, which mixes tokens across units at the same
    time step. The final embedding of a unit is the mean over its tokens across time.

    For NuCLR's forward pass, the transform returned by :meth:`get_input_fn` is the
    intended final transform to be added to the
    :class:`~torch_brain.datasets.Dataset`. It converts
    :class:`~torch_brain.data.Data` samples into the format NuCLR needs. See the
    documentation of :meth:`get_input_fn` for expected attributes.

    Args:
        ctx_duration: Duration of the input window (in seconds).
        latent_step: Duration covered by each token (in seconds). Defaults to ``1.0``.
        bin_size: Spike binning resolution (in seconds). Defaults to ``0.02``.
        dim: Hidden and embedding dimension. Defaults to ``256``.
        self_heads: Number of attention heads. Defaults to ``4``.
        dim_head: Dimension of each attention head. Defaults to ``64``.
        lin_dropout: Dropout rate. Defaults to ``0.3``.
        t_layers: Number of temporal-only blocks. Defaults to ``2``.
        st_layers: Number of spatial + temporal blocks. Defaults to ``4``.
        rot_ratio: Fraction of each head's dimensions that are rotated by the
            rotary time embedding. Defaults to ``0.5``.
        rotate_value: Whether temporal attention rotates the values as well.
            Defaults to ``True``.
        attn_biases: Whether attention query/key/value projections have a bias.
            Defaults to ``True``.

    **References**

    Arora, V., Lachi, D., Knight, I. J., Azabou, M., Richards, B., Hurwitz, C.,
    Siegle, J. H., & Dyer, E. L.
    *Know Thyself by Knowing Others: Learning Neuron Identity from Population Context.*
    `NeurIPS 2025 <https://arxiv.org/abs/2512.01199>`_.
    """

    def __init__(
        self,
        *,
        ctx_duration: float,
        latent_step: float = 1.0,
        bin_size: float = 0.02,
        dim: int = 256,
        self_heads: int = 4,
        dim_head: int = 64,
        lin_dropout: float = 0.3,
        t_layers: int = 2,
        st_layers: int = 4,
        rot_ratio: float = 0.5,
        rotate_value: bool = True,
        attn_biases: bool = True,
    ):
        super().__init__()
        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.num_latents = _ratio(ctx_duration, latent_step)
        self.bins_per_latent = _ratio(latent_step, bin_size)
        self.dim = dim

        self.read_in = nn.Linear(self.bins_per_latent, dim)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=latent_step,
            t_max=8.0 * self.num_latents * latent_step,
        )
        self.dropout = nn.Dropout(lin_dropout)

        def temporal_attention():
            return Attention(dim, self_heads, dim_head, attn_biases, rotate_value)

        def spatial_attention():
            return Attention(dim, self_heads, dim_head, attn_biases)

        def ffn():
            return FeedForward(dim, dropout=lin_dropout)

        self.t_blocks = nn.ModuleList(
            nn.ModuleList([temporal_attention(), ffn()]) for _ in range(t_layers)
        )
        self.st_blocks = nn.ModuleList(
            nn.ModuleList([spatial_attention(), ffn(), temporal_attention(), ffn()])
            for _ in range(st_layers)
        )

    def forward(self, bins: Tensor, group_lengths: Tensor) -> Tensor:
        r"""
        Args:
            bins: Binned spike counts of all units in the batch, of shape
                :math:`(U, T)`, where :math:`T = \text{ctx_duration} / \text{bin_size}`.
            group_lengths: Number of units in each spatial attention group, of shape
                :math:`(G,)`. Groups are contiguous in ``bins`` and sum to :math:`U`.

        Returns:
            Unit embeddings of shape :math:`(U, D)`.
        """
        num_units, num_bins = bins.shape
        if num_bins != self.num_latents * self.bins_per_latent:
            raise ValueError(
                f"Expected {self.num_latents * self.bins_per_latent} bins per unit "
                f"(ctx_duration / bin_size), got {num_bins}."
            )
        x = bins.unflatten(1, (self.num_latents, self.bins_per_latent))
        x = self.read_in(x.to(self.read_in.weight.dtype))

        t = torch.arange(self.num_latents, device=x.device) * self.latent_step
        rotary_emb = self.rotary_emb(t)
        spatial_seqlen = group_lengths.repeat(self.num_latents)

        for attn, ffn in self.t_blocks:
            x = x + self.dropout(attn(x, rotary_emb=rotary_emb))
            x = x + self.dropout(ffn(x))

        for s_attn, s_ffn, t_attn, t_ffn in self.st_blocks:
            x = x.transpose(0, 1).flatten(0, 1)
            x = x + self.dropout(s_attn(x, seqlen=spatial_seqlen))
            x = x + self.dropout(s_ffn(x))
            x = x.unflatten(0, (self.num_latents, num_units)).transpose(0, 1)

            x = x + self.dropout(t_attn(x, rotary_emb=rotary_emb))
            x = x + self.dropout(t_ffn(x))

        return x.mean(1)

    def get_input_fn(self, group_by: str | None = "probe_id") -> "NuCLRInputFn":
        r"""Returns the transform that converts a :class:`~torch_brain.data.Data`
        sample into NuCLR inputs.

        The transform bins the spikes of a sample, drops units that do not spike in
        it, and sorts the remaining units into spatial attention groups. It is
        intended to be used as the last transform of a
        :class:`~torch_brain.datasets.Dataset`, with batches assembled by
        :func:`~torch_brain.batching.collate`.

        Expected attributes of ``data``:

        - ``domain``: Time span of the sample, at most ``ctx_duration`` long.
        - ``spikes.timestamps``: Spike times in seconds, relative to the start of the
          window (as produced by :meth:`Data.slice <torch_brain.data.Data.slice>`).
        - ``spikes.unit_index``: Index of each spike's unit in ``units``.
        - ``units.id``: Unit identifiers.
        - ``units.<group_by>``: Spatial attention group of each unit. Only needed if
          ``group_by`` is not ``None``.

        The transform returns a dict with ``bins`` and ``group_lengths`` (the inputs
        to :meth:`forward`), ``num_units``, and per-unit ``unit_ids`` and
        ``group_ids`` in the same order as ``bins``. It raises ``ValueError`` if
        ``data`` spans more than ``ctx_duration``.

        Args:
            group_by: Name of a ``data.units`` attribute. Spatial attention is
                restricted to units sharing the same value. If ``None``, all units of
                a sample attend to each other. The original paper recommends
                restricting spatial attention to units on the same probe.
                Defaults to ``"probe_id"``.
        """
        return NuCLRInputFn(self.ctx_duration, self.bin_size, group_by)

    @torch.no_grad()
    def embed(
        self,
        data: Data,
        *,
        step: float | None = None,
        batch_size: int = 32,
        group_by: str | None = "probe_id",
    ) -> Tensor:
        r"""Embeds every unit of a recording.

        The recording is split into ``ctx_duration``-long windows, and the embedding
        of each unit is averaged over the windows in which it spikes. Call
        :meth:`eval` first to disable dropout.

        Args:
            data: Recording with the attributes expected by :meth:`get_input_fn`.
            step: Time between the starts of consecutive windows (in seconds).
                Defaults to ``ctx_duration / 2``.
            batch_size: Number of windows per forward pass. Defaults to ``32``.
            group_by: See :meth:`get_input_fn`. Defaults to ``"probe_id"``.

        Returns:
            Unit embeddings of shape :math:`(U, D)`, in the order of ``data.units``.
            Units that never spike are ``NaN``.
        """
        sampler = SequentialFixedWindowSampler(
            sampling_intervals={"": data.domain},
            window_length=self.ctx_duration,
            step=step or self.ctx_duration / 2,
            drop_short=True,
        )
        windows = list(sampler)
        input_fn = self.get_input_fn(group_by)
        device = self.read_in.weight.device
        unit_index = {uid: i for i, uid in enumerate(data.units.id.astype(str))}

        total = torch.zeros(
            len(unit_index), self.dim, device=device, dtype=torch.float64
        )
        count = torch.zeros(len(unit_index), device=device, dtype=torch.float64)
        for i in range(0, len(windows), batch_size):
            samples = [
                input_fn(data.slice(w.start, w.end))
                for w in windows[i : i + batch_size]
            ]
            batch = collate(samples)
            z = self(batch["bins"].to(device), batch["group_lengths"].to(device))
            idx = torch.tensor(
                [unit_index[uid] for uid in batch["unit_ids"]], device=device
            )
            total.index_add_(0, idx, z.double())
            count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))

        return (total / count[:, None]).to(self.read_in.weight.dtype)


@dataclass
class NuCLRInputFn:
    r"""Transform returned by :meth:`NuCLR.get_input_fn`."""

    ctx_duration: float
    bin_size: float
    group_by: str | None = "probe_id"

    def __post_init__(self):
        self.num_bins = _ratio(self.ctx_duration, self.bin_size)

    def __call__(self, data: Data) -> dict:
        duration = data.domain.end[-1] - data.domain.start[0]
        if duration > self.ctx_duration * (1 + 1e-9):  # Room for floating-point errors.
            raise ValueError(
                f"Sample duration ({duration}s) is longer than "
                f"ctx_duration ({self.ctx_duration}s)."
            )

        num_units = len(data.units.id)
        num_bins = self.num_bins

        # Binning:
        # Not using torch_brain.utils.bin_spikes: it anchors bins to spikes.domain,
        # which can be shorter than the window. This model needs exactly `num_bins`.
        bin_index = np.floor(data.spikes.timestamps / self.bin_size).astype(np.int64)
        unit_index = data.spikes.unit_index.astype(np.int64)
        valid = (bin_index >= 0) & (bin_index < num_bins)
        bins = np.bincount(
            unit_index[valid] * num_bins + bin_index[valid],
            minlength=num_units * num_bins,
        ).reshape(num_units, num_bins)

        # Compute spatial attention groups of units that spike in this window:
        if self.group_by is None:
            group_ids = np.zeros(num_units, dtype=np.int64)
        else:
            group_ids = np.asarray(getattr(data.units, self.group_by))
        active = np.flatnonzero(bins.any(1))
        order = active[np.argsort(group_ids[active], kind="stable")]
        _, group_lengths = np.unique(group_ids[order], return_counts=True)

        return {
            "bins": chain(bins[order].astype(np.int16)),
            "group_lengths": chain(group_lengths),
            "num_units": len(order),
            "unit_ids": chain(data.units.id.astype(str)[order]),
            "group_ids": chain(group_ids[order]),
        }


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * mult * dim)
        self.out_proj = nn.Linear(mult * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.gelu(gate) * x))


def _ratio(a: float, b: float) -> int:
    ratio = round(a / b)
    if abs(a / b - ratio) > 1e-6:
        raise ValueError(f"{a} is not a multiple of {b}")
    return ratio
