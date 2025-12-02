from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from temporaldata import Data
from torch_brain.utils.preprocessing import z_score_normalize


class DeepConvNet(nn.Module):
    """
    Deep ConvNet for EEG window classification.

    Core interface
    --------------
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
    kernel_time_1 : int, default=10
        Temporal kernel size for block 1 conv_time.
    kernel_time_2 : int, default=10
        Temporal kernel size for block 2 conv.
    kernel_time_3 : int, default=10
        Temporal kernel size for block 3 conv.
    kernel_time_4 : int, default=10
        Temporal kernel size for block 4 conv.
    pool_time_size : int, default=3
        Temporal pooling kernel size (same for all 4 blocks).
    pool_time_stride : int, default=3
        Temporal pooling stride (same for all 4 blocks).

    n_filters_time : int, default=25
        Number of temporal filters in block 1.
    n_filters_spat : int, default=25
        Number of spatial filters in block 1.
    n_filters_2 : int, default=50
    n_filters_3 : int, default=100
    n_filters_4 : int, default=200

    dropout : float, default=0.5
        Dropout probability after each pooling layer.
    use_logsoftmax : bool, default=True
        If True, apply log-softmax to outputs (for NLLLoss).
        If False, return raw logits (for CrossEntropyLoss).

    auto_kernel : bool, default=False
        If True, kernel sizes are suggested heuristically from in_times.
    verbose : bool, default=False
        If True and auto_kernel is enabled, print chosen kernels.

    Tokenizer
    ---------
    tokenize(data: Data) expects:

        data.eeg.sig : array-like [T, C]
        data.trials.label : array-like with at least one label (optional)

    It returns:

        {
            "input_values"   : FloatTensor [C, T_pad],
            "input_mask"     : BoolTensor [T_pad],
            "target_values"  : LongTensor scalar,
            "target_weights" : FloatTensor scalar,
            "model_hints"    : dict
        }
    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        *,
        # temporal kernels (optionally auto-adjusted)
        kernel_time_1: int = 10,
        kernel_time_2: int = 10,
        kernel_time_3: int = 10,
        kernel_time_4: int = 10,
        # pooling
        pool_time_size: int = 3,
        pool_time_stride: int = 3,
        # filters
        n_filters_time: int = 25,
        n_filters_spat: int = 25,
        n_filters_2: int = 50,
        n_filters_3: int = 100,
        n_filters_4: int = 200,
        # regularization / head
        dropout: float = 0.5,
        use_logsoftmax: bool = True,
        # kernel heuristics
        auto_kernel: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans
        self.n_classes = n_classes

        # --------------------------------------------------------------
        # Kernel selection (optional auto-heuristics)
        # --------------------------------------------------------------
        if auto_kernel:
            (
                kernel_time_1,
                kernel_time_2,
                kernel_time_3,
                kernel_time_4,
                pool_time_size,
                pool_time_stride,
            ) = self.suggest_kernels(in_times)
            if verbose:
                print(
                    "[DeepConvNet] auto kernels:"
                    f" k1={kernel_time_1}, k2={kernel_time_2},"
                    f" k3={kernel_time_3}, k4={kernel_time_4},"
                    f" pool={pool_time_size}, stride={pool_time_stride}"
                )

        # store hyperparameters
        self.kernel_time_1 = int(kernel_time_1)
        self.kernel_time_2 = int(kernel_time_2)
        self.kernel_time_3 = int(kernel_time_3)
        self.kernel_time_4 = int(kernel_time_4)
        self.pool_time_size = int(pool_time_size)
        self.pool_time_stride = int(pool_time_stride)

        # analytic minimum temporal length and effective in_times
        self.min_T = self._compute_min_T()
        self.in_times = max(in_times, self.min_T)

        # --------------------------------------------------------------
        # Block 1: temporal + spatial conv
        # --------------------------------------------------------------
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(self.kernel_time_1, 1),
            bias=False,
        )
        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(1, in_chans),
            bias=True,
        )
        self.bn1 = nn.BatchNorm2d(n_filters_spat)
        self.pool1 = nn.MaxPool2d(
            kernel_size=(self.pool_time_size, 1),
            stride=(self.pool_time_stride, 1),
        )
        self.drop1 = nn.Dropout(dropout)

        # --------------------------------------------------------------
        # Block 2
        # --------------------------------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=n_filters_spat,
            out_channels=n_filters_2,
            kernel_size=(self.kernel_time_2, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(n_filters_2)
        self.pool2 = nn.MaxPool2d(
            kernel_size=(self.pool_time_size, 1),
            stride=(self.pool_time_stride, 1),
        )
        self.drop2 = nn.Dropout(dropout)

        # --------------------------------------------------------------
        # Block 3
        # --------------------------------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=n_filters_2,
            out_channels=n_filters_3,
            kernel_size=(self.kernel_time_3, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(n_filters_3)
        self.pool3 = nn.MaxPool2d(
            kernel_size=(self.pool_time_size, 1),
            stride=(self.pool_time_stride, 1),
        )
        self.drop3 = nn.Dropout(dropout)

        # --------------------------------------------------------------
        # Block 4
        # --------------------------------------------------------------
        self.conv4 = nn.Conv2d(
            in_channels=n_filters_3,
            out_channels=n_filters_4,
            kernel_size=(self.kernel_time_4, 1),
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(n_filters_4)
        self.pool4 = nn.MaxPool2d(
            kernel_size=(self.pool_time_size, 1),
            stride=(self.pool_time_stride, 1),
        )
        self.drop4 = nn.Dropout(dropout)

        # --------------------------------------------------------------
        # Final conv length: probe with a dummy forward
        # --------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_chans, self.in_times)  # [B, C, T]
            final_T = self._forward_features(dummy).shape[2]

        self.conv_classifier = nn.Conv2d(
            in_channels=n_filters_4,
            out_channels=n_classes,
            kernel_size=(final_T, 1),
            bias=True,
        )

        self.activation = nn.LogSoftmax(dim=1) if use_logsoftmax else nn.Identity()

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

        # tokenizer state (ShallowNet/EEGNet-compatible)
        self._tokenizer_eval = False
        self._normalize_mode = "zscore"  # "zscore" | "none"

    # ==================================================================
    # Kernel suggestion (DeepConvNet-aware)
    # ==================================================================
    def suggest_kernels(self, T: int) -> Tuple[int, int, int, int, int, int]:
        """
        Heuristic kernel sizing based on window length.

        - conv kernels ~1% of input length, clipped to [10, 40].
        - pooling uses small, stable (3, 3) settings.
        """
        k = max(10, min(int(T * 0.01), 40))
        pool = 3
        stride = 3
        return k, k, k, k, pool, stride

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
            self.kernel_time_1,
            self.kernel_time_2,
            self.kernel_time_3,
            self.kernel_time_4,
        ]
        pool_k = [self.pool_time_size] * 4
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
    # Tokenizer controls
    # ==================================================================
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """
        Toggle deterministic evaluation mode for the tokenizer.

        Currently a placeholder for future stochastic augmentations; kept
        for interface compatibility with other Torch Brain models.
        """
        self._tokenizer_eval = flag

    def set_tokenizer_opts(self, *, normalize_mode: str = "zscore") -> None:
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
    # Tokenizer (ShallowNet/EEGNet-compatible output dict)
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
            raise ValueError("Sample missing EEG at data.eeg.sig")

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

        # label + weight (mirrors ShallowNet/EEGNet behavior)
        if hasattr(data, "trials") and len(getattr(data.trials, "label", [])) > 0:
            try:
                y_val = int(data.trials.label[0])
                y_w = 1.0
            except Exception:
                y_val, y_w = 0, 0.0
        else:
            # unlabeled → dummy label, zero weight (ignored in loss)
            y_val, y_w = 0, 0.0

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
            "kernel_time_1": self.kernel_time_1,
            "kernel_time_2": self.kernel_time_2,
            "kernel_time_3": self.kernel_time_3,
            "kernel_time_4": self.kernel_time_4,
            "pool_time_size": self.pool_time_size,
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
        outputs : FloatTensor [B, n_classes]
            Either log-probabilities (if use_logsoftmax=True) or logits.
        """
        feats = self._forward_features(x)
        out = self.conv_classifier(feats)
        out = self.activation(out)
        out = out.squeeze(-1).squeeze(-1)  # [B, n_classes]
        return out
