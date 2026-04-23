"""Metrics for ordinal frame indices (e.g. natural movie readouts)."""

import torch
from torch import Tensor
from torchmetrics import Metric


class FrameIndexWindowAccuracy(Metric):
    r"""Share of samples where the predicted frame index is within a band around the target.

    For each example, the predicted class is ``preds.argmax(dim=-1)``. A hit is counted
    when ``|pred - target| <= half_window`` (inclusive), i.e. predictions within that
    many **frames** of the label count as correct.

    Typical choice: set ``half_window`` to about the number of frames in one model
    context window (e.g. 1 s at 30 Hz → 30 frames).

    Args:
        half_window: Maximum absolute difference in **frame index** for a hit.
            ``0`` recovers exact top-1 accuracy.
    """

    full_state_update: bool = True

    def __init__(self, half_window: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.half_window = int(half_window)
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.dim() != 2:
            raise ValueError(
                f"FrameIndexWindowAccuracy expects preds of shape (N, C); got {preds.shape}"
            )
        if target.dim() != 1:
            raise ValueError(
                f"FrameIndexWindowAccuracy expects target of shape (N,); got {target.shape}"
            )
        if preds.shape[0] != target.shape[0]:
            raise ValueError("preds and target must have the same batch size")

        pred_idx = preds.argmax(dim=-1)
        tgt = target.long()
        hits = (pred_idx - tgt).abs() <= self.half_window
        self.correct += hits.sum()
        self.total += hits.numel()

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device, dtype=torch.float32)
        return self.correct.float() / self.total.float()
