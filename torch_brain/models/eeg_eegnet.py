"""
EEGNet v2 — Torch Brain compatible
==================================

A Torch Brain–integrated reimplementation of **EEGNet v2**
(Lawhern et al., 2018), a compact convolutional neural network for
EEG window classification. This version follows the unified Torch Brain
design used by ShallowNet and DeepConvNet
 
References
----------
- Lawhern, V. J., et al. (2018). "EEGNet: a compact convolutional
  neural network for EEG-based brain–computer interfaces."
  *Journal of Neural Engineering.*

- Original Keras implementation:
  https://github.com/vlawhern/arl-eegmodels

License & Attribution
---------------------
This implementation follows the EEGNet architecture from Lawhern et al.
and the API conventions of the Torch Brain EEG model suite. Please cite
the original paper when using this model in academic work.

"""

from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from temporaldata import Data

from torch_brain.utils.preprocessing import z_score_normalize


class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d with MaxNorm constraint on the weights, similar to Keras' MaxNorm.
    Intended use: spatial depthwise conv in EEGNet.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply max-norm constraint on weights before conv
        with torch.no_grad():
            w = self.weight
            # For Conv2d: (out_channels, in_channels/groups, kh, kw)
            norms = w.norm(p=2, dim=(1, 2, 3), keepdim=True)
            desired = torch.clamp(norms, max=self.max_norm)
            self.weight.copy_(w * (desired / (1e-8 + norms)))
        return super().forward(x)


class EEGNet(nn.Module):
    """
    PyTorch reimplementation of EEGNet v2 (Lawhern et al., 2018),
    adapted to the Torch Brain ecosystem.

    Core interface (shared with other EEG models)
    ---------------------------------------------
    Args
    ----
    in_chans : int
        Number of EEG channels.
    in_times : int
        Number of time samples per window.
    n_classes : int
        Number of output classes.

    Architecture hyperparameters
    ----------------------------
    F1 : int, default=8
        Number of temporal filters in the first block.
    D : int, default=2
        Depth multiplier for spatial depthwise conv (F2 = F1 * D if F2 is None).
    F2 : int or None, default=None
        Number of feature maps after depthwise-separable conv. If None, F2 = F1 * D.
    kernel_time : int, default=64
        Temporal kernel size for the first convolution (over time).
    kernel_time_separable : int, default=16
        Temporal kernel size for the depthwise separable conv in block 2.
    pool_time_size_1 : int, default=4
        Temporal pooling size for the first pooling layer.
    pool_time_size_2 : int, default=8
        Temporal pooling size for the second pooling layer.
    dropout : float, default=0.25
        Dropout probability.
    spatial_max_norm : float, default=1.0
        MaxNorm constraint for the spatial depthwise conv.

    Tokenizer options
    -----------------
    normalize_mode : {"zscore", "none"}, default="zscore"
        How to normalize EEG channels over time in tokenize().

    Expected input to forward()
    ---------------------------
    x : FloatTensor [B, C, T]
        EEG batch with `B` samples, `C` channels, and `T` time points.

    Output
    ------
    logits : FloatTensor [B, n_classes]
        Raw class scores (logits) suitable for CrossEntropyLoss.
    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        *,
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = None,
        kernel_time: int = 64,
        kernel_time_separable: int = 16,
        pool_time_size_1: int = 4,
        pool_time_size_2: int = 8,
        dropout: float = 0.25,
        spatial_max_norm: float = 1.0,
        normalize_mode: str = "zscore",
    ):
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        self.in_chans = in_chans
        self.n_classes = n_classes

        # store kernel / pool hyperparams with unified naming
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_time = int(kernel_time)
        self.kernel_time_separable = int(kernel_time_separable)
        self.pool_time_size_1 = int(pool_time_size_1)
        self.pool_time_size_2 = int(pool_time_size_2)

        # minimal temporal length: product of pooling factors (as in original EEGNet)
        self.min_T = self.pool_time_size_1 * self.pool_time_size_2
        self.in_times = max(in_times, self.min_T)

        # tokenizer state
        self._tokenizer_eval = False
        self._normalize_mode = normalize_mode

        # ---------- Block 1: Temporal Convolution ----------
        # Input will be shaped [B, 1, C, T]; conv over time (last dim)
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_time),
            padding=(0, self.kernel_time // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.F1, momentum=0.01, eps=1e-3)

        # ---------- Block 1: Depthwise Spatial Convolution ----------
        self.conv_spatial = Conv2dWithConstraint(
            in_channels=self.F1,
            out_channels=self.F1 * self.D,
            kernel_size=(in_chans, 1),
            groups=self.F1,
            bias=False,
            max_norm=spatial_max_norm,
        )
        self.bn2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, eps=1e-3)
        self.elu = nn.ELU()
        self.pool1_layer = nn.AvgPool2d(kernel_size=(1, self.pool_time_size_1))
        self.drop1 = nn.Dropout(dropout)

        # ---------- Block 2: Depthwise-Separable Convolution ----------
        self.conv_depthwise = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F1 * self.D,
            kernel_size=(1, self.kernel_time_separable),
            groups=self.F1 * self.D,
            padding=(0, self.kernel_time_separable // 2),
            bias=False,
        )
        self.conv_pointwise = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.F2, momentum=0.01, eps=1e-3)
        self.pool2_layer = nn.AvgPool2d(kernel_size=(1, self.pool_time_size_2))
        self.drop2 = nn.Dropout(dropout)

        # ---------- Final Conv classifier (global conv over spatial+time) ----------
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, self.in_times)  # [B, C, T]
            feat = self._forward_features_from_C_T(dummy)
            _, _, h, w = feat.shape

        self.classifier = nn.Conv2d(
            in_channels=self.F2,
            out_channels=n_classes,
            kernel_size=(h, w),
            bias=True,
        )

        # Keras-like initialization (glorot_uniform + zero biases)
        self._init_keras_style()

    # ------------------------------------------------------------------
    # Tokenizer controls (match ShallowNet / DeepConvNet)
    # ------------------------------------------------------------------
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """Placeholder for future stochastic augs; kept for interface parity."""
        self._tokenizer_eval = flag

    def set_tokenizer_opts(self, *, normalize_mode: str = "zscore") -> None:
        """
        Configure tokenizer options.

        Args
        ----
        normalize_mode : {"zscore", "none"}
            How to normalize EEG channels over time.
        """
        if normalize_mode not in {"zscore", "none"}:
            raise ValueError(f"Unknown normalize_mode={normalize_mode}")
        self._normalize_mode = normalize_mode

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    def tokenize(self, data: Data) -> Dict[str, Any]:
        """
        Convert a temporaldata.Data sample into tensors for EEGNet.

        temporaldata stores EEG as [T, C] but EEGNet expects [C, T].

        Returns a dictionary compatible with Torch Brain pipelines.
        """
        # ---- 1) Extract EEG ----
        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("Sample missing EEG at data.eeg.signal")

        x = np.asarray(sig, dtype=np.float32)  # [T, C]
        if x.ndim != 2:
            raise ValueError(f"Expected EEG shape [T, C], got {x.shape}")

        # convert to [C, T]
        x = x.T
        x_t = torch.from_numpy(x)  # [C, T]

        # ---- 2) Normalization ----
        if self._normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self._normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self._normalize_mode}")

        # ---- 3) Label handling ----
        if hasattr(data, "trials") and len(getattr(data.trials, "label", [])) > 0:
            try:
                y = int(data.trials.label[0])
                w = 1.0
            except Exception:
                y, w = 0, 0.0
        else:
            # unlabeled → dummy label, zero weight (ignored in loss)
            y, w = 0, 0.0

        # ---- 4) Check min time requirement + pad if needed ----
        C, T = x_t.shape
        if T < self.min_T:
            pad_t = self.min_T - T
            input_mask = torch.cat(
                [
                    torch.ones(T, dtype=torch.bool),
                    torch.zeros(pad_t, dtype=torch.bool),
                ],
                dim=0,
            )
            x_t = F.pad(x_t, (0, pad_t))  # pad time axis → [C, min_T]
            T_out = self.min_T
        else:
            input_mask = torch.ones(T, dtype=torch.bool)
            T_out = T

        return {
            "input_values": x_t,  # [C, T_out]
            "input_mask": input_mask,  # [T_out]
            "target_values": torch.tensor(y, dtype=torch.long),
            "target_weights": torch.tensor(w, dtype=torch.float32),
            "model_hints": {
                "in_chans": C,
                "in_times": T_out,
                "F1": self.F1,
                "D": self.D,
                "F2": self.F2,
                "kernel_time": self.kernel_time,
                "kernel_time_separable": self.kernel_time_separable,
                "pool_time_size_1": self.pool_time_size_1,
                "pool_time_size_2": self.pool_time_size_2,
                "min_time_required": self.min_T,
            },
        }

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_keras_style(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Feature extractor from [B, 1, C, T] or [B, C, T]
    # ------------------------------------------------------------------
    def _forward_features_from_C_T(self, x: torch.Tensor) -> torch.Tensor:
        """
        Utility to run features starting from [B, C, T] input (no batchnorm etc. changed).
        Used for dummy shape probing and main forward.
        """
        # x: [B, C, T]
        if x.dim() != 3:
            raise ValueError(
                f"EEGNet expects x of shape [B, C, T] in this helper, got {tuple(x.shape)}"
            )
        x = x.unsqueeze(1)  # [B, 1, C, T]
        return self._forward_features(x)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, C, T]
        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1_layer(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2_layer(x)
        x = self.drop2(x)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args
        ----
        x : FloatTensor [B, C, T]

        Returns
        -------
        logits : FloatTensor [B, n_classes]
        """
        if x.dim() != 3:
            raise ValueError(
                f"EEGNet forward expects x of shape [B, C, T], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {x.shape[1]}")

        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self._forward_features(x)
        x = self.classifier(x)
        x = x.squeeze(-1).squeeze(-1)  # [B, n_classes]
        return x
