"""
EEGNet v2 baseline for Torch Brain
==================================

This module implements the EEGNet v2 architecture for EEG window
classification, adapted to the Torch Brain ecosystem.

Design:
- Single-label classification per EEG window.
- Torch Brain–friendly:
    temporaldata.Data -> tokenize -> DataLoader -> forward.
- Standardized constructor args:
    in_chans, in_times, n_classes, filter_time_length, n_filters_time,
    n_filters_spat, pool_time_length, pool_time_stride, ...
- Built-in tokenizer with normalization, padding, and mask.


References
----------
- Lawhern, V. J., et al. (2018). *EEGNet: a compact convolutional
  neural network for EEG-based brain–computer interfaces.*
  Journal of Neural Engineering.
- EEGNet reference implementation:
  https://github.com/vlawhern/arl-eegmodels


"""

from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from temporaldata import Data

from torch_brain.utils.preprocessing import z_score_normalize


class EEGNet(nn.Module):
    """
    EEGNet v2 for EEG window classification.

    Follows the canonical EEGNet v2 structure:
    - Block 1: temporal conv -> depthwise spatial conv -> pool.
    - Block 2: depthwise–separable temporal conv -> pool.
    Then a linear classifier on flattened features.

    Parameters
    ----------
    in_chans : int
        Number of EEG input channels.
    in_times : int
        Number of time samples per window.
    n_classes : int
        Number of output classes.
    filter_time_length : int, default 64
        Temporal kernel size of the first convolutional block.
        (EEGNet's `kernel_length`).
    n_filters_time : int, default 8
        Number of temporal filters in the first block (EEGNet's F1).
    n_filters_spat : Optional[int], default None
        Number of filters after the separable conv block (EEGNet's F2).
        If None, F2 = n_filters_time * depth_multiplier.
    depth_multiplier : int, default 2
        Depth multiplier for depthwise conv (EEGNet's D).
    sep_filter_time_length : int, default 16
        Temporal kernel size in the separable conv block.
    pool_time_length : int, default 4
        Pooling kernel length in the first block (time dimension).
    pool_time_stride : int, default 4
        Pooling stride in the first block (time dimension).
    sep_pool_time_length : int, default 8
        Pooling kernel length in the second block (time dimension).
    sep_pool_time_stride : int, default 8
        Pooling stride in the second block (time dimension).
    dropout_p : float, default 0.5
        Dropout probability after each pooling block.
    logsoftmax : bool, default True
        If True, apply LogSoftmax over the classifier output.
        Otherwise, return raw logits.
    auto_kernel : bool, default False
        If True, call `suggest_kernels(in_times)` to override temporal
        kernel/pool hyperparameters from the input length.
    verbose : bool, default True
        If True and `auto_kernel=True`, print the chosen kernel settings.
    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        filter_time_length: int = 64,
        n_filters_time: int = 8,
        n_filters_spat: Optional[int] = None,
        depth_multiplier: int = 2,
        sep_filter_time_length: int = 16,
        pool_time_length: int = 4,
        pool_time_stride: int = 4,
        sep_pool_time_length: int = 8,
        sep_pool_time_stride: int = 8,
        dropout_p: float = 0.5,
        logsoftmax: bool = True,
        auto_kernel: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.in_times = in_times
        self.n_classes = n_classes

        # ---- optional kernel suggestion ----
        if auto_kernel:
            (
                filter_time_length,
                pool_time_length,
                pool_time_stride,
                sep_filter_time_length,
                sep_pool_time_length,
                sep_pool_time_stride,
            ) = self.suggest_kernels(in_times)
            if verbose:
                print(
                    "[EEGNet] Auto kernel settings: "
                    f"filter_time_length={filter_time_length}, "
                    f"pool_time_length={pool_time_length}, "
                    f"pool_time_stride={pool_time_stride}, "
                    f"sep_filter_time_length={sep_filter_time_length}, "
                    f"sep_pool_time_length={sep_pool_time_length}, "
                    f"sep_pool_time_stride={sep_pool_time_stride}"
                )

        # ---- basic type checks for shared hyperparams ----
        for name, val in {
            "filter_time_length": filter_time_length,
            "pool_time_length": pool_time_length,
            "pool_time_stride": pool_time_stride,
            "sep_filter_time_length": sep_filter_time_length,
            "sep_pool_time_length": sep_pool_time_length,
            "sep_pool_time_stride": sep_pool_time_stride,
        }.items():
            if not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")

        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.sep_filter_time_length = sep_filter_time_length
        self.sep_pool_time_length = sep_pool_time_length
        self.sep_pool_time_stride = sep_pool_time_stride
        self.depth_multiplier = depth_multiplier

        self.n_filters_time = n_filters_time
        if n_filters_spat is None:
            n_filters_spat = n_filters_time * depth_multiplier
        self.n_filters_spat = n_filters_spat

        # ---- Block 1: temporal conv + depthwise spatial conv ----
        # Expect input as [B, 1, C, T]
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_filters_time,
            kernel_size=(1, self.filter_time_length),
            padding=(0, self.filter_time_length // 2),  # ~ "same" over time
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.n_filters_time)

        # Depthwise spatial conv over channels
        self.conv_depthwise = nn.Conv2d(
            in_channels=self.n_filters_time,
            out_channels=self.n_filters_time * self.depth_multiplier,
            kernel_size=(in_chans, 1),
            groups=self.n_filters_time,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.n_filters_time * self.depth_multiplier)

        self.pool1 = nn.AvgPool2d(
            kernel_size=(1, self.pool_time_length),
            stride=(1, self.pool_time_stride),
        )
        self.dropout = nn.Dropout(dropout_p)

        # ---- Block 2: separable conv (depthwise temporal + pointwise) ----
        self.conv_sep_depthwise = nn.Conv2d(
            in_channels=self.n_filters_time * self.depth_multiplier,
            out_channels=self.n_filters_time * self.depth_multiplier,
            kernel_size=(1, self.sep_filter_time_length),
            padding=(0, self.sep_filter_time_length // 2),
            groups=self.n_filters_time * self.depth_multiplier,
            bias=False,
        )
        self.conv_sep_pointwise = nn.Conv2d(
            in_channels=self.n_filters_time * self.depth_multiplier,
            out_channels=self.n_filters_spat,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.n_filters_spat)

        self.pool2 = nn.AvgPool2d(
            kernel_size=(1, self.sep_pool_time_length),
            stride=(1, self.sep_pool_time_stride),
        )

        # ---- Classifier head: infer feature dim with dummy forward ----
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_chans, in_times)
            feat = self._forward_features(dummy)
            feat_dim = feat.shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1) if logsoftmax else nn.Identity()

        # ---- tokenizer state ----
        self._normalize_mode = "zscore"
        self._tokenizer_eval = False  # placeholder for future stochastic opts

    # ------------------------------------------------------------------
    # Kernel suggestion (auto_kernel)
    # ------------------------------------------------------------------
    def suggest_kernels(self, in_times: int) -> Tuple[int, int, int, int, int, int]:
        """
        Suggest kernel and pooling hyperparameters given input length.

        Heuristic:
        - filter_time_length ~ 5–10% of in_times, clipped.
        - pool_time_length and sep_pool_time_length chosen so that
          min_T = pool_time_length * sep_pool_time_length is safely <= in_times.
        - Strides equal to their corresponding kernel lengths.

        Returns
        -------
        filter_time_length, pool_time_length, pool_time_stride,
        sep_filter_time_length, sep_pool_time_length, sep_pool_time_stride
        """
        # temporal conv kernel
        filter_t = max(16, min(int(in_times * 0.1), 128))

        # pooling; ensure that their product is not ridiculously large
        pool1 = max(2, min(8, in_times // 4))
        pool2 = max(2, min(8, in_times // (4 * pool1)))

        pool1_stride = pool1
        pool2_stride = pool2

        sep_filter_t = max(8, min(32, filter_t // 2))

        return (
            int(filter_t),
            int(pool1),
            int(pool1_stride),
            int(sep_filter_t),
            int(pool2),
            int(pool2_stride),
        )

    # ------------------------------------------------------------------
    # Tokenizer controls
    # ------------------------------------------------------------------
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """Toggle deterministic evaluation mode (placeholder for future use)."""
        self._tokenizer_eval = flag

    def set_tokenizer_opts(self, *, normalize_mode: str = "zscore") -> None:
        """
        Configure tokenizer options.

        Parameters
        ----------
        normalize_mode : {"zscore", "none"}
            Normalization to apply per-channel over time.
        """
        self._normalize_mode = normalize_mode

    # ------------------------------------------------------------------
    # Tokenizer: Data -> dict
    # ------------------------------------------------------------------
    def tokenize(self, data: Data) -> Dict[str, Any]:
        """
        Convert a `temporaldata.Data` sample into model-ready tensors.

        Expects `data.eeg.signal` with shape [T, C] (time × channels).

        Returns
        -------
        batch : dict
            {
              "input_values": FloatTensor [C, T’],
              "input_mask":   BoolTensor  [T’],
              "target_values": LongTensor [],
              "target_weights": FloatTensor [],
              "model_hints": dict,
            }
        """
        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("EEGNet.tokenize expects data.eeg.signal to be present.")

        x_np = np.asarray(sig, dtype=np.float32)
        if x_np.ndim != 2:
            raise ValueError(f"EEG signal must be 2D [T, C], got {x_np.shape}")

        # [T, C] -> [C, T]
        x_np = x_np.T
        x_t = torch.from_numpy(x_np)  # [C, T]

        # normalization
        if self._normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self._normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self._normalize_mode}")

        # target + weight (simple convention; can later be replaced by a helper)
        if (
            hasattr(data, "trials")
            and hasattr(data.trials, "label")
            and len(data.trials) > 0
        ):
            try:
                y, w = int(data.trials.label[0]), 1.0
            except Exception:
                y, w = 0, 0.0
        else:
            y, w = 0, 0.0

        C, T = x_t.shape

        # ---- padding logic & mask ----
        # For EEGNet with AvgPool2d and stride = kernel, we need:
        # T >= pool_time_length * sep_pool_time_length to have at least 1 output.
        min_T = self.pool_time_length * self.sep_pool_time_length

        if T < min_T:
            pad_t = min_T - T
            input_mask = torch.cat(
                [torch.ones(T, dtype=torch.bool), torch.zeros(pad_t, dtype=torch.bool)],
                dim=0,
            )
            x_t = F.pad(x_t, (0, pad_t))  # pad on time dimension
            T = min_T
        else:
            input_mask = torch.ones(T, dtype=torch.bool)

        return {
            "input_values": x_t,  # [C, T’]
            "input_mask": input_mask,  # [T’]
            "target_values": torch.tensor(y, dtype=torch.long),
            "target_weights": torch.tensor(w, dtype=torch.float32),
            "model_hints": {
                "in_chans": C,
                "in_times": T,
                "filter_time_length": self.filter_time_length,
                "pool_time_length": self.pool_time_length,
                "pool_time_stride": self.pool_time_stride,
                "sep_filter_time_length": self.sep_filter_time_length,
                "sep_pool_time_length": self.sep_pool_time_length,
                "sep_pool_time_stride": self.sep_pool_time_stride,
                "min_time_required": min_T,
            },
        }

    # ------------------------------------------------------------------
    # Padding helper for batched tensors
    # ------------------------------------------------------------------
    def pad_time_if_needed(self, x_bct: torch.Tensor) -> torch.Tensor:
        """
        Pad batched input on the time dimension if it is too short.

        Parameters
        ----------
        x_bct : FloatTensor
            [B, C, T] or [B, C, T, 1]

        Returns
        -------
        padded : FloatTensor
            Same shape as input, but with time padded to >= `min_T`.

        Notes
        -----
        This is a safety net; in most training setups, padding is handled in
        the tokenizer + DataLoader. Still, this prevents obscure conv/pool
        shape errors if someone feeds very short windows directly.
        """
        if x_bct.dim() == 4:
            B, C, T, _ = x_bct.shape
        elif x_bct.dim() == 3:
            B, C, T = x_bct.shape
        else:
            raise ValueError("Input must be (B, C, T) or (B, C, T, 1)")

        min_T = self.pool_time_length * self.sep_pool_time_length
        if T >= min_T:
            return x_bct

        pad_t = min_T - T
        if x_bct.dim() == 3:
            return F.pad(x_bct, (0, pad_t))  # pad time
        else:
            return F.pad(x_bct, (0, 0, 0, pad_t))  # pad time dim for (B,C,T,1)

    # ------------------------------------------------------------------
    # Internal feature extractor (conv blocks only)
    # ------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional blocks only.

        Expects `x` as [B, 1, C, T].
        Returns flattened features [B, F].
        """
        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv_depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv_sep_depthwise(x)
        x = self.conv_sep_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        return x.flatten(start_dim=1)  # [B, F]

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : FloatTensor
            [B, C, T] or [B, C, T, 1].

        Returns
        -------
        logits : FloatTensor [B, n_classes]
        """
        # (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            B, C, T = x.shape
            if C != self.in_chans:
                raise ValueError(
                    f"EEGNet expected {self.in_chans} channels, got {C} in x.shape={x.shape}"
                )
            x = x.unsqueeze(1)  # [B, 1, C, T]

        # (B, C, T, 1) -> (B, 1, C, T)
        elif x.dim() == 4:
            B, C, T, last = x.shape
            if last == 1:
                if C != self.in_chans:
                    raise ValueError(
                        f"EEGNet expected {self.in_chans} channels, got {C} in x.shape={x.shape}"
                    )
                x = x.permute(0, 3, 1, 2)  # [B, 1, C, T]
            else:
                raise ValueError(
                    "EEGNet.forward expected [B, C, T] or [B, C, T, 1], "
                    f"got shape {x.shape}."
                )
        else:
            raise ValueError(
                "EEGNet.forward expected [B, C, T] or [B, C, T, 1], "
                f"got tensor with shape {x.shape} and dim={x.dim()}."
            )

        # Safety net for too-short T
        x = self.pad_time_if_needed(x)

        feat = self._forward_features(x)
        logits = self.classifier(feat)
        return self.logsoftmax(logits)
