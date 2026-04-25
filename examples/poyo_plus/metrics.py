"""Custom torchmetrics for the Calcium POYO+ example.

Lives under ``examples/poyo_plus`` so it can be referenced from the dataset
configs as ``_target_: metrics.WithinDeltaAccuracy``. The ``examples/poyo_plus``
directory is on ``sys.path`` whenever ``train.py`` / ``finetune.py`` is invoked
(the script's directory is auto-prepended to ``sys.path[0]``), which makes
``metrics`` importable as a sibling module without a package install.
"""

import torch
from torch import Tensor
from torchmetrics import Metric


class WithinDeltaAccuracy(Metric):
    r"""Share of samples whose predicted class index is within a band around the target.

    For each example, the predicted class is ``preds.argmax(dim=-1)``. A hit is counted
    when ``|pred - target| <= tolerance`` (inclusive), i.e. predictions within that
    many index steps of the label count as correct. ``tolerance=0`` recovers exact
    top-1 accuracy.

    Useful for ordinal classification problems where neighboring class indices encode
    nearby quantities (e.g. frame indices in a video readout, where a one-frame miss
    should not be penalized as harshly as a random one).

    Args:
        tolerance: Maximum absolute difference (in class-index steps) for a hit.
            Must be a non-negative integer. ``0`` recovers exact top-1 accuracy.
    """

    full_state_update: bool = False

    def __init__(self, tolerance: int = 0, **kwargs):
        super().__init__(**kwargs)
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0; got {tolerance}")
        self.tolerance = int(tolerance)
        self.add_state(
            "correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.dim() != 2:
            raise ValueError(
                f"WithinDeltaAccuracy expects preds of shape (N, C); got {preds.shape}"
            )
        if target.dim() != 1:
            raise ValueError(
                f"WithinDeltaAccuracy expects target of shape (N,); got {target.shape}"
            )
        if preds.shape[0] != target.shape[0]:
            raise ValueError("preds and target must have the same batch size")

        pred_idx = preds.argmax(dim=-1)
        tgt = target.long()
        hits = (pred_idx - tgt).abs() <= self.tolerance
        self.correct += hits.sum()
        self.total += hits.numel()

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.tensor(
                0.0, device=self.correct.device, dtype=torch.float32
            )
        return self.correct.float() / self.total.float()
