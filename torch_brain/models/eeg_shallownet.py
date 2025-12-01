"""
Shallow ConvNet (ShallowNet) for EEG — Torch Brain compatible
=============================================================

This module implements a Shallow ConvNet variant (Schirrmeister et al., 2017) adapted
for the Torch Brain codebase. It provides a compact CNN for **window classification**
(one label per EEG window) plus an **optional built-in tokenizer** that turns raw
`temporaldata.Data` samples into model-ready tensors. The design mirrors the common
Torch Brain pattern: Dataset → (transforms) → tokenizer → DataLoader → `forward`.

References
----------
- Schirrmeister, R. T., et al. (2017). *Deep learning with convolutional neural networks
  for EEG decoding and visualization.* Human Brain Mapping.
  arXiv: https://arxiv.org/abs/1703.05051
- Braindecode (BSD-3) reference implementation:
  https://github.com/braindecode/braindecode

License & attribution
---------------------
This implementation is inspired by Braindecode’s BSD-3 codebase
and follows Torch Brain conventions. Please attribute the original
paper when using this model in publications and respect third-party licenses if you
reuse external components.
"""

import torch
from torch import nn
from typing import Union, Tuple, Optional, Dict, Any
import torch.nn.functional as F
import numpy as np
from temporaldata import Data
from torch_brain.utils.preprocessing import z_score_normalize


class ShallowNet(nn.Module):
    """
    Shallow ConvNet for EEG window classification (Schirrmeister et al., 2017),
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

    Main architecture hyperparameters
    ---------------------------------
    kernel_time : int, default=25
        Temporal kernel size for the first convolution.
    n_filters_time : int, default=40
        Number of temporal filters.
    n_filters_spat : int, default=40
        Number of spatial filters.
    pool_time_size : int, default=75
        Temporal pooling window size.
    pool_time_stride : int, default=15
        Temporal pooling stride.
    final_conv_length : int or "auto", default="auto"
        Temporal length for the final classifier conv. If "auto", it is
        computed analytically from in_times and the conv/pool stack.
    dropout : float, default=0.5
        Dropout probability after pooling.
    use_logsoftmax : bool, default=True
        If True, apply log-softmax to output logits; otherwise return raw
        scores.

    Tokenizer controls
    ------------------
    auto_kernel : bool, default=False
        If True, suggest kernel_time / pool_time_* heuristically from in_times.
    verbose : bool, default=False
        If True and auto_kernel is enabled, print the chosen kernels.

    Input / output shapes
    ---------------------
    Forward:
        x : FloatTensor of shape [B, C, T] or [B, C, T, 1]
        returns: FloatTensor [B, n_classes]

    Tokenizer:
        tokenize(data: Data) -> dict with keys:
            - input_values   : FloatTensor [C, T_pad]
            - input_mask     : BoolTensor [T_pad]
            - target_values  : LongTensor scalar
            - target_weights : FloatTensor scalar
            - model_hints    : dict
    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        *,
        kernel_time: int = 25,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        pool_time_size: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: Union[int, str] = "auto",
        dropout: float = 0.5,
        use_logsoftmax: bool = True,
        auto_kernel: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.n_classes = n_classes

        # ---- optional heuristic kernel selection ----
        if auto_kernel:
            kernel_time, pool_time_size, pool_time_stride = self.suggest_kernels(
                in_times
            )
            if verbose:
                print(
                    "[ShallowNet] auto kernels: "
                    f"kernel_time={kernel_time}, "
                    f"pool_time_size={pool_time_size}, "
                    f"pool_time_stride={pool_time_stride}"
                )

        # ---- basic type checks ----
        for name, val in {
            "kernel_time": kernel_time,
            "pool_time_size": pool_time_size,
            "pool_time_stride": pool_time_stride,
        }.items():
            if not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")

        # store on self
        self.kernel_time = kernel_time
        self.pool_time_size = pool_time_size
        self.pool_time_stride = pool_time_stride

        # minimal temporal length for valid conv + pool
        self.min_T = self.kernel_time + self.pool_time_size - 1
        self.in_times = max(in_times, self.min_T)

        # ---- layers ----
        # Input will be [B, 1, T, C] before conv_time
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(self.kernel_time, 1),
            bias=True,
        )
        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(1, in_chans),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(n_filters_spat, momentum=0.1)

        self.pool = nn.AvgPool2d(
            kernel_size=(self.pool_time_size, 1),
            stride=(self.pool_time_stride, 1),
        )
        self.dropout = nn.Dropout(p=dropout)

        # analytic final conv length along time dimension
        if final_conv_length == "auto":
            # after conv_time (valid conv)
            L1 = self.in_times - self.kernel_time + 1
            # after pooling
            L2 = (L1 - self.pool_time_size) // self.pool_time_stride + 1
            self.final_conv_length = max(1, L2)
        else:
            self.final_conv_length = int(final_conv_length)

        self.conv_classifier = nn.Conv2d(
            in_channels=n_filters_spat,
            out_channels=n_classes,
            kernel_size=(self.final_conv_length, 1),
            bias=True,
        )

        self.activation = nn.LogSoftmax(dim=1) if use_logsoftmax else nn.Identity()

        # ---- init (Schirrmeister / Braindecode style) ----
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1.0)
        if self.conv_time.bias is not None:
            nn.init.constant_(self.conv_time.bias, 0.0)

        nn.init.xavier_uniform_(self.conv_spat.weight, gain=1.0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

        nn.init.xavier_uniform_(self.conv_classifier.weight, gain=1.0)
        nn.init.constant_(self.conv_classifier.bias, 0.0)

        # ---- tokenizer state (shared across EEG models) ----
        self._tokenizer_eval = False
        self._normalize_mode = "zscore"  # "zscore" | "none"

    # ------------------------------------------------------------------
    # Kernel suggestion heuristic
    # ------------------------------------------------------------------
    def suggest_kernels(self, in_times: int) -> Tuple[int, int, int]:
        """
        Heuristic kernel sizing for ShallowNet:

        - kernel_time     ~ 1% of in_times, clipped to [10, 100]
        - pool_time_size  ~ 10 * kernel_time (capped at in_times//2, min 25)
        - pool_time_stride ~ 20% of pool_time_size (min 5)
        """
        k_t = max(10, min(int(in_times * 0.01), 100))
        p = max(25, min(k_t * 10, max(25, in_times // 2)))
        s = max(5, p // 5)
        return int(k_t), int(p), int(s)

    # ------------------------------------------------------------------
    # Tokenizer controls
    # ------------------------------------------------------------------
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """Placeholder for future stochastic aug; kept for interface parity."""
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
        Convert a temporaldata.Data sample into a Torch Brain compatible dict.

        Expects:
            data.eeg.sig : array-like [T, C]

        Returns:
            dict with keys:
                - input_values   : FloatTensor [C, T_pad]
                - input_mask     : BoolTensor [T_pad]
                - target_values  : LongTensor scalar
                - target_weights : FloatTensor scalar
                - model_hints    : dict
        """
        # 1) get EEG
        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("Sample missing EEG at data.eeg.sig")

        x = np.asarray(sig, dtype=np.float32)  # [T, C]
        if x.ndim != 2:
            raise ValueError(f"EEG must be 2D [T, C], got {x.shape}")
        x = x.T  # [C, T]

        # 2) to torch + normalize
        x_t = torch.from_numpy(x)  # [C, T]
        if self._normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self._normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self._normalize_mode}")

        # 3) label + weight (same convention across EEG models)
        if hasattr(data, "trials") and len(getattr(data.trials, "label", [])) > 0:
            try:
                y = int(data.trials.label[0])
                w = 1.0
            except Exception:
                y, w = 0, 0.0
        else:
            # unlabeled → dummy label, zero weight (ignored in loss)
            y, w = 0, 0.0

        # 4) pad time to min_T if needed + mask
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
            x_t = F.pad(x_t, (0, pad_t))  # [C, min_T]
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
                "kernel_time": self.kernel_time,
                "pool_time_size": self.pool_time_size,
                "pool_time_stride": self.pool_time_stride,
                "min_time_required": self.min_T,
            },
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args
        ----
        x : FloatTensor
            [B, C, T] or [B, C, T, 1]

        Returns
        -------
        logits_or_log_probs : FloatTensor [B, n_classes]
        """
        # accept (B, C, T) or (B, C, T, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # -> (B, C, T, 1)
        elif x.dim() == 4:
            if x.shape[-1] != 1:
                raise ValueError(
                    f"ShallowNet 4D input must be [B, C, T, 1], got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"ShallowNet expects x of shape [B, C, T] or [B, C, T, 1], "
                f"got dim={x.dim()} and shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {x.shape[1]}")

        # reorder to (B, 1, T, C) for conv_time over time then conv_spat over channels
        x = x.permute(0, 3, 2, 1)

        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x * x  # square nonlinearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # safe log
        x = self.dropout(x)
        x = self.conv_classifier(x)
        x = self.activation(x)
        x = x.squeeze(3).squeeze(2)  # (B, n_classes)
        return x
