"""
Deep ConvNet (DeepConvNet) for EEG — Torch Brain compatible
===========================================================

A deep convolutional baseline for EEG window classification from:

    Schirrmeister, R. T., et al. (2017).
    "Deep learning with convolutional neural networks for EEG decoding
     and visualization." Human Brain Mapping.
     https://arxiv.org/abs/1703.05051

This implementation follows Torch Brain conventions:

- Forward accepts EEG windows as tensors of shape [B, C, T] (or [B, C, T, 1]).
- A built-in tokenizer takes `temporaldata.Data` and returns a Torch Brain style
  dict with input values, masks, targets, and model hints.
- `min_T` is computed analytically from the conv/pool stack: it is the true
  minimum temporal length required for the network to produce at least one
  output time step. Shorter windows are padded by the tokenizer.

Compared to ShallowNet, DeepConvNet uses more conv–pool blocks and a wider
filter hierarchy, but exposes the same tokenizer API so switching baselines in
tutorials or training scripts is frictionless.

This model is inspired by the Braindecode BSD-3 Deep4Net implementation.
"""

from typing import Dict, Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from temporaldata import Data
from torch_brain.utils.preprocessing import z_score_normalize


class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG window classification.

    Architecture (time × channels view)
    -----------------------------------
    The model applies four successive temporal convolution + pooling stages:

        Block 1:
            conv_time (over time)
            conv_spat (across channels)
            batch norm → ELU → max-pool → dropout

        Blocks 2–4:
            temporal conv → batch norm → ELU → max-pool → dropout

    A final temporal conv (`conv_classifier`) collapses the remaining time
    dimension and produces logits of shape [B, n_classes]. If `logsoftmax=True`,
    the outputs are log-probabilities suitable for `torch.nn.NLLLoss`.

    Input
    -----
    x : FloatTensor
        - [B, C, T]   (preferred)
        - [B, C, T, 1] (legacy / compatibility)

    Output
    ------
    logits : FloatTensor [B, n_classes]

    Tokenizer interface (Torch Brain standard)
    -----------------------------------------
    `tokenize(data: temporaldata.Data) -> dict` with keys:

        - input_values   : FloatTensor [C, T_pad]
        - input_mask     : BoolTensor [T_pad]
        - target_values  : LongTensor scalar
        - target_weights : FloatTensor scalar
        - model_hints    : dict (includes min_time_required)

    Normalization and options
    -------------------------
    - `set_tokenizer_eval(flag: bool)`      : reserved for future stochastic aug.
    - `set_tokenizer_opts(normalize_mode=)` :
        * "zscore" (default): per-channel z-score over time
        * "none"            : no normalization

    Minimum temporal length (min_T)
    --------------------------------
    `min_T` is derived analytically from the conv/pool configuration: it is the
    smallest number of time samples T such that all conv + pool layers can be
    applied and yield at least one output position.

    The tokenizer pads any window with T < min_T up to min_T along the time
    dimension, and builds an input mask that marks real vs. padded samples.
    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        #
        # Temporal kernels (optionally auto-adjusted)
        #
        filter_time_length: int = 10,
        conv2_kernel: int = 10,
        conv3_kernel: int = 10,
        conv4_kernel: int = 10,
        #
        pool_time_length: int = 3,
        pool_time_stride: int = 3,
        #
        n_filters_time: int = 25,
        n_filters_spat: int = 25,
        n_filters_2: int = 50,
        n_filters_3: int = 100,
        n_filters_4: int = 200,
        #
        dropout_p: float = 0.5,
        logsoftmax: bool = True,
        auto_kernel: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.in_times = in_times
        self.n_classes = n_classes

        # --------------------------------------------------------------
        # Kernel selection (optional auto-heuristics)
        # --------------------------------------------------------------
        if auto_kernel:
            (
                filter_time_length,
                conv2_kernel,
                conv3_kernel,
                conv4_kernel,
                pool_time_length,
                pool_time_stride,
            ) = self.suggest_kernels(in_times)
            if verbose:
                print(
                    "[DeepConvNet] auto kernels:"
                    f" conv1={filter_time_length}, conv2={conv2_kernel},"
                    f" conv3={conv3_kernel}, conv4={conv4_kernel},"
                    f" pool={pool_time_length}, stride={pool_time_stride}"
                )

        # store on self for tokenizer + min_T logic
        self.filter_time_length = int(filter_time_length)
        self.conv2_kernel = int(conv2_kernel)
        self.conv3_kernel = int(conv3_kernel)
        self.conv4_kernel = int(conv4_kernel)
        self.pool_time_length = int(pool_time_length)
        self.pool_time_stride = int(pool_time_stride)

        # --------------------------------------------------------------
        # Block 1: temporal + spatial conv
        # --------------------------------------------------------------
        self.conv_time = nn.Conv2d(
            1,
            n_filters_time,
            (self.filter_time_length, 1),
            bias=False,
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time,
            n_filters_spat,
            (1, in_chans),
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(n_filters_spat)
        self.pool1 = nn.MaxPool2d(
            (self.pool_time_length, 1),
            (self.pool_time_stride, 1),
        )
        self.drop1 = nn.Dropout(dropout_p)

        # --------------------------------------------------------------
        # Block 2
        # --------------------------------------------------------------
        self.conv2 = nn.Conv2d(
            n_filters_spat,
            n_filters_2,
            (self.conv2_kernel, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(n_filters_2)
        self.pool2 = nn.MaxPool2d(
            (self.pool_time_length, 1),
            (self.pool_time_stride, 1),
        )
        self.drop2 = nn.Dropout(dropout_p)

        # --------------------------------------------------------------
        # Block 3
        # --------------------------------------------------------------
        self.conv3 = nn.Conv2d(
            n_filters_2,
            n_filters_3,
            (self.conv3_kernel, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(n_filters_3)
        self.pool3 = nn.MaxPool2d(
            (self.pool_time_length, 1),
            (self.pool_time_stride, 1),
        )
        self.drop3 = nn.Dropout(dropout_p)

        # --------------------------------------------------------------
        # Block 4
        # --------------------------------------------------------------
        self.conv4 = nn.Conv2d(
            n_filters_3,
            n_filters_4,
            (self.conv4_kernel, 1),
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(n_filters_4)
        self.pool4 = nn.MaxPool2d(
            (self.pool_time_length, 1),
            (self.pool_time_stride, 1),
        )
        self.drop4 = nn.Dropout(dropout_p)

        # --------------------------------------------------------------
        # Analytic min_T + final conv length
        # --------------------------------------------------------------
        self.min_T = self._compute_min_T()

        # Use a single dummy forward at min_T to determine the temporal
        # dimension right before the classifier.
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_chans, self.min_T)  # [B, C, T]
            final_T = self._forward_features(dummy).shape[2]

        self.conv_classifier = nn.Conv2d(
            n_filters_4,
            n_classes,
            (final_T, 1),
            bias=True,
        )

        self.logsoftmax = nn.LogSoftmax(dim=1) if logsoftmax else nn.Identity()

        # --------------------------------------------------------------
        # Init
        # --------------------------------------------------------------
        for m in [
            self.conv_time,
            self.conv_spat,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv_classifier,
        ]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        for bn in [self.bn1, self.bn2, self.bn3, self.bn4]:
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)

        # tokenizer state (mirrors ShallowNet)
        self._tokenizer_eval = False
        self._normalize_mode = "zscore"  # "zscore" | "none"

    # ==================================================================
    # Kernel suggestion (DeepConvNet-aware)
    # ==================================================================
    def suggest_kernels(self, T: int):
        """
        Heuristic kernel sizing based on window length.

        - conv1 kernel ~1% of input length, clipped to [10, 40].
        - conv2–4 use the same temporal kernel as conv1 by default.
        - pooling uses small, stable (3, 3) settings.
        """
        conv1 = max(10, min(int(T * 0.01), 40))
        conv2 = conv1
        conv3 = conv1
        conv4 = conv1

        pool = 3
        stride = 3
        return conv1, conv2, conv3, conv4, pool, stride

    # ==================================================================
    # Analytic min_T: true architectural minimum temporal length
    # ==================================================================
    def _compute_min_T(self) -> int:
        """
        Compute the minimum temporal length required so that the full
        conv+pool pipeline produces at least one output time step.

        We invert the forward shapes analytically, starting from a final
        length of 1 and reversing each pool + conv pair.
        """
        conv_k = [
            self.filter_time_length,
            self.conv2_kernel,
            self.conv3_kernel,
            self.conv4_kernel,
        ]
        pool_k = [self.pool_time_length] * 4
        pool_s = [self.pool_time_stride] * 4

        L = 1  # require final temporal length >= 1

        # Reverse through blocks: (pool4, conv4), ..., (pool1, conv1)
        for ck, pk, ps in reversed(list(zip(conv_k, pool_k, pool_s))):
            # reverse pool: L_in >= (L_out - 1) * stride + kernel
            L = (L - 1) * ps + pk
            # reverse conv: L_in >= L_out + kernel - 1
            L = L + ck - 1

        return int(L)

    # ==================================================================
    # Tokenizer controls (ShallowNet-compatible)
    # ==================================================================
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """
        Toggle deterministic evaluation mode for the tokenizer.

        Currently a placeholder for future stochastic augmentations; kept
        for interface compatibility with other Torch Brain models.
        """
        self._tokenizer_eval = flag

    def set_tokenizer_opts(
        self,
        *,
        normalize_mode: str = "zscore",
    ) -> None:
        """
        Configure tokenizer options.

        Parameters
        ----------
        normalize_mode : {"zscore", "none"}
            How to normalize EEG channels over time before model input.
        """
        if normalize_mode not in {"zscore", "none"}:
            raise ValueError(f"Unknown normalize_mode={normalize_mode}")
        self._normalize_mode = normalize_mode

    # ==================================================================
    # Feature extractor (conv stack)
    # ==================================================================
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conv–pool backbone.

        Accepts:
            - [B, C, T]
            - [B, C, T, 1] (legacy / compatibility)

        Internally:
            - ensure [B, C, T, 1]
            - permute to [B, 1, T, C] so conv_time operates over time and
              conv_spat over channels.
        """
        if x.dim() == 3:
            # [B, C, T] -> [B, C, T, 1]
            x = x.unsqueeze(-1)
        elif x.dim() == 4:
            # expect [B, C, T, 1]
            if x.shape[-1] != 1:
                raise ValueError(
                    f"DeepConvNet 4D input must be [B, C, T, 1], got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                f"DeepConvNet expects x of shape [B, C, T] or [B, C, T, 1], "
                f"got dim={x.dim()} and shape={tuple(x.shape)}"
            )

        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {x.shape[1]}")

        # -> [B, 1, T, C] for conv_time (time) then conv_spat (channels)
        x = x.permute(0, 3, 2, 1)

        # Block 1
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool4(x)
        x = self.drop4(x)

        return x

    # ==================================================================
    # Tokenizer (ShallowNet-compatible output dict)
    # ==================================================================
    def tokenize(self, data: Data) -> Dict[str, Any]:
        """
        Convert a temporaldata.Data sample into tensors compatible with
        DeepConvNet and Torch Brain DataLoaders.

        Expects:
            data.eeg.signal : array-like [T, C]
        """
        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("Sample missing EEG at data.eeg.signal")

        # temporaldata uses [T, C]; convert to [C, T]
        x = np.asarray(sig, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"EEG must be 2D [T, C], got shape {x.shape}")
        x = x.T  # [C, T]

        x_t = torch.from_numpy(x)

        # normalization (per-channel over time)
        if self._normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self._normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self._normalize_mode}")

        # label + weight (mirrors ShallowNet behavior)
        if hasattr(data, "trials") and len(getattr(data.trials, "label", [])) > 0:
            try:
                y_val = int(data.trials.label[0])
                y_w = 1.0
            except Exception:
                y_val, y_w = 0, 0.0
        else:
            # Fallback for dummy or unlabeled data; keep loss computable
            y_val, y_w = 1, 1.0

        C, T = x_t.shape

        # pad to analytic min_T if needed
        if T < self.min_T:
            pad_len = self.min_T - T
            input_mask = torch.cat(
                [
                    torch.ones(T, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool),
                ],
                dim=0,
            )
            x_t = F.pad(x_t, (0, pad_len))  # [C, min_T]
            T_out = self.min_T
        else:
            input_mask = torch.ones(T, dtype=torch.bool)
            T_out = T

        model_hints = {
            "in_chans": C,
            "in_times": T_out,
            "filter_time_length": self.filter_time_length,
            "conv2_kernel": self.conv2_kernel,
            "conv3_kernel": self.conv3_kernel,
            "conv4_kernel": self.conv4_kernel,
            "pool_time_length": self.pool_time_length,
            "pool_time_stride": self.pool_time_stride,
            "min_time_required": self.min_T,
        }

        return {
            "input_values": x_t,  # [C, T_out]
            "input_mask": input_mask,  # [T_out]
            "target_values": torch.tensor(y_val, dtype=torch.long),
            "target_weights": torch.tensor(y_w, dtype=torch.float32),
            "model_hints": model_hints,
        }

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : FloatTensor
            - [B, C, T]
            - [B, C, T, 1]

        Returns
        -------
        logits : FloatTensor [B, n_classes]
        """
        feats = self._forward_features(x)
        out = self.conv_classifier(feats)
        out = self.logsoftmax(out)
        out = out.squeeze(-1).squeeze(-1)  # [B, n_classes]
        return out
