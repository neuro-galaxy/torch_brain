from collections.abc import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NuCLRLoss(nn.Module):
    r"""Contrastive loss used to train :class:`~torch_brain.models.NuCLR`.

    Within a sample, embeddings that share a unit id (e.g. the same unit seen in two
    views of a recording) are positives, and all other units in the same group are
    negatives. Samples are weighted by their number of positive pairs, jointly across
    ranks when running distributed.

    Args:
        dim: Embedding dimension.
        temperature: Softmax temperature. Defaults to ``0.2``.
        decoupled: Whether to exclude positives from the denominator
            (`DCL <https://arxiv.org/abs/2110.06848>`_). Defaults to ``True``.
        projector: Whether to pass the embeddings through an MLP projector.
            Defaults to ``True``.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 0.2,
        decoupled: bool = True,
        projector: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.decoupled = decoupled
        self.projector = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
            if projector
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        unit_ids: Sequence | np.ndarray | Tensor,
        sample_lengths: Sequence[int] | Tensor,
        group_ids: Sequence | np.ndarray | Tensor | None = None,
    ) -> Tensor:
        r"""
        Args:
            x: Unit embeddings of shape :math:`(N, D)`.
            unit_ids: Unit identifiers of shape :math:`(N,)`.
            sample_lengths: Number of embeddings in each sample, of shape :math:`(B,)`.
                Embeddings are only contrasted within their sample.
            group_ids: Group identifiers of shape :math:`(N,)`. If given, negatives are
                restricted to the same group.

        Returns:
            Scalar loss.
        """
        x = F.normalize(self.projector(x), dim=-1)
        unit_index = _as_index(unit_ids, x.device)
        group_index = (
            torch.zeros_like(unit_index)
            if group_ids is None
            else _as_index(group_ids, x.device)
        )
        lengths = torch.as_tensor(sample_lengths).tolist()

        losses, num_positives = [], []
        samples = zip(
            x.split(lengths),
            unit_index.split(lengths),
            group_index.split(lengths),
            strict=True,
        )
        for x_, u, g in samples:
            sim = x_ @ x_.T / self.temperature
            identity = torch.eye(len(x_), dtype=torch.bool, device=x.device)
            positives = (u[:, None] == u[None, :]) & ~identity
            excluded = identity | (g[:, None] != g[None, :])
            if self.decoupled:
                excluded |= positives

            lse = sim.masked_fill(excluded, float("-inf")).logsumexp(1)
            valid = lse.isfinite()
            n = positives.sum()

            numer = -(sim * positives).sum() / n.clamp(min=1)
            denom = lse[valid].sum() / valid.sum().clamp(min=1)
            losses.append(numer + denom)
            num_positives.append(n)

        losses, num_positives = torch.stack(losses), torch.stack(num_positives)
        weights = num_positives / _all_reduce(num_positives.sum()).clamp(min=1)
        return _all_reduce((losses * weights).sum())


def _as_index(ids: Sequence | np.ndarray | Tensor, device: torch.device) -> Tensor:
    if isinstance(ids, Tensor):
        return ids.to(device)
    inverse = np.unique(np.asarray(ids), return_inverse=True)[1].reshape(-1)
    return torch.as_tensor(inverse, device=device)


def _all_reduce(x: Tensor) -> Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(x)
