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
    Shallow ConvNet for EEG window classification.

    Adapted from *Schirrmeister et al., 2017* and the Braindecode implementation,
    this model integrates seamlessly with the Torch Brain tokenizer interface.

    It performs single-label classification for each EEG window.

    Input
    -----
    x : FloatTensor [B, C, T] or [B, C, T, 1]
        EEG batch with `B` samples, `C` channels, and `T` time points.
        The optional singleton spatial dimension (`1`) can represent session context.

    Output
    ------
    logits : FloatTensor [B, n_classes]
        Class scores suitable for `torch.nn.CrossEntropyLoss`.
        If `logsoftmax=True`, the outputs are log-probabilities.
        Otherwise, raw scores are returned—these can be used for alternative loss
        functions or as embeddings when using the model as an encoder.

    Notes
    -----
    - Normalization utilities are imported from
    `torch_brain.utils.preprocessing` (e.g., `z_score_normalize`, etc.).
    - All convolutional and batch normalization layers are initialized
    following the scheme used in *Schirrmeister et al., 2017*:
    Xavier-uniform for convolutional weights and constant (1/0) for
    batch norm parameters.
    - Minimum valid time length:
    ``min_T = filter_time_length + pool_time_length - 1``.
    Inputs shorter than this are automatically padded during tokenization,
    as convolutional layers cannot operate on insufficient temporal context.

    """

    def __init__(
        self,
        in_chans: int,
        in_times: int,
        n_classes: int,
        filter_time_length: int = 25,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        final_conv_length: Union[int, str] = "auto",
        dropout_p: float = 0.5,
        logsoftmax: bool = True,
        auto_kernel: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            in_chans: Number of EEG input channels.
            in_times: Number of time samples per window.
            n_classes: Number of output classes.
            filter_time_length: Temporal convolution kernel size.
            n_filters_time: Number of temporal filters.
            n_filters_spat: Number of spatial filters.
            pool_time_length: Pooling kernel length.
            pool_time_stride: Pooling stride.
            final_conv_length: Temporal length for final conv ("auto" computes analytically).
            dropout_p: Dropout probability.
            logsoftmax: Apply log-softmax to output logits.
            auto_kernel: Automatically infer kernel sizes from `in_times`.
            verbose: Print auto-kernel settings if True.
        """
        super().__init__()
        self.in_chans = in_chans
        self.in_times = in_times

        # ---- kernel selection (optional) ----
        if auto_kernel:
            filter_time_length, pool_time_length, pool_time_stride = (
                self.suggest_kernels(in_times)
            )
            if verbose:
                print(
                    f"[ShallowNet] Auto kernel settings: "
                    f"filter_time_length={filter_time_length}, "
                    f"pool_time_length={pool_time_length}, "
                    f"pool_time_stride={pool_time_stride}"
                )

        # ---- type checks ----
        for name, val in {
            "filter_time_length": filter_time_length,
            "pool_time_length": pool_time_length,
            "pool_time_stride": pool_time_stride,
        }.items():
            if not isinstance(val, int):
                raise TypeError(f"{name} must be int, got {type(val).__name__}")

        # store on self so other methods (pad_input) can use them
        self.filter_time_length = filter_time_length
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride

        # ---- layers ----
        self.conv_time = nn.Conv2d(
            1, n_filters_time, (self.filter_time_length, 1), bias=True
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, (1, in_chans), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_spat, momentum=0.1)
        self.pool = nn.AvgPool2d(
            kernel_size=(self.pool_time_length, 1), stride=(self.pool_time_stride, 1)
        )
        self.dropout = nn.Dropout(p=dropout_p)

        # analytic final conv length (time axis only)
        if final_conv_length == "auto":
            L1 = (in_times - self.filter_time_length) // 1 + 1
            L3 = (L1 - self.pool_time_length) // self.pool_time_stride + 1
            self.final_conv_length = max(1, L3)  # safety
        else:
            self.final_conv_length = int(final_conv_length)

        self.conv_classifier = nn.Conv2d(
            n_filters_spat,
            n_classes,
            kernel_size=(self.final_conv_length, 1),
            bias=True,
        )
        self.logsoftmax = nn.LogSoftmax(dim=1) if logsoftmax else nn.Identity()

        # init
        nn.init.xavier_uniform_(self.conv_time.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv_spat.weight, gain=1.0)
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)
        nn.init.xavier_uniform_(self.conv_classifier.weight, gain=1.0)
        nn.init.constant_(self.conv_classifier.bias, 0.0)
        if self.conv_time.bias is not None:
            nn.init.constant_(self.conv_time.bias, 0.0)

        # ---- tokenizer state ----
        self._tokenizer_eval = False  # deterministic cropping when True
        self._normalize_mode = "zscore"  # "zscore" | "none"

    # ---------- Kernel suggestion ----------
    def suggest_kernels(self, in_times: int) -> Tuple[int, int, int]:
        """
        Heuristics:
        - filter_time_length ~ 1% of in_times (min 10, max 100)
        - pool_time_length ~ 10 × filter_time_length (cap at in_times//2, min 25)
        - pool_time_stride ~ 20% of pool_time_length (min 5)
        Constraint we care about:
            in_times >= filter_time_length + pool_time_length - 1
        """
        k_t = max(10, min(int(in_times * 0.01), 100))
        p = max(25, min(k_t * 10, max(25, in_times // 2)))
        s = max(5, p // 5)
        return int(k_t), int(p), int(s)

    # ---------- Built-in tokenizer controls ----------
    def set_tokenizer_eval(self, flag: bool = True) -> None:
        """
        Toggle deterministic evaluation mode for the tokenizer.

        Args:
            flag: If True, enforces deterministic preprocessing (no random
                augmentations). Currently a placeholder for future use.

        Notes
        -----
        Retained for interface consistency across EEG baselines and future
        Hydra-based control of stochastic vs. deterministic tokenization.
        """
        self._tokenizer_eval = flag

    def set_tokenizer_opts(
        self,
        *,
        normalize_mode: str = "zscore",
    ) -> None:
        """
        Configure tokenizer options.

        Args:
            normalize_mode: One of {"zscore", "none"}.
                Controls how EEG signals are normalized before model input.

        Notes
        -----
        Designed for Hydra-driven configuration (e.g., via YAML):
        tokenizer.normalize_mode: zscore
        Additional normalization strategies and options may be added later.
        """
        self._normalize_mode = normalize_mode

    # ---------- Built-in tokenizer ----------
    def tokenize(self, data: Data) -> Dict[str, Any]:
        """
        Convert a temporaldata.Data sample into tensors compatible with DataLoader.

        Args:
           data: TemporalData sample with `data.eeg.signal` of shape [T, C].


        Returns:
            dict: Contains:
                - input_values (FloatTensor[C, T])
                - input_mask (BoolTensor[T])
                - target_values (LongTensor[])
                - target_weights (FloatTensor[])
                - model_hints (Dict[str, Any])
        """
        # 1) get EEG numpy
        sig = getattr(data.eeg, "signal", None)
        if sig is None:
            raise ValueError("Sample missing EEG at data.eeg.signal")

        # 2) convert and fix orientation: temporaldata uses [T, C]
        x = np.asarray(sig, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"EEG must be 2D, got {x.shape}")
        x = x.T  # [C, T]

        # 3) to torch and normalize per-channel over time
        x_t = torch.from_numpy(x)  # [C, T]
        if self._normalize_mode == "zscore":
            x_t = z_score_normalize(x_t)
        elif self._normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize_mode={self._normalize_mode}")

        # 4) target + weight
        if (
            hasattr(data, "trials")
            and len(data.trials) > 0
            and hasattr(data.trials, "label")
        ):
            try:
                y, w = int(data.trials.label[0]), 1.0
            except Exception:
                y, w = 0, 0.0
        else:
            # Fallback for dummy or unlabeled data
            y, w = 0, 0.0  # assign default label so loss is computable
            # alternative: y, w = 0, 0.0  # ignore in loss

        # 5) pad time (only if needed) + mask
        C, T = x_t.shape
        min_T = self.filter_time_length + self.pool_time_length - 1  # e.g., 25+75-1=99
        if T < min_T:
            pad_t = min_T - T
            input_mask = torch.cat(
                [torch.ones(T, dtype=torch.bool), torch.zeros(pad_t, dtype=torch.bool)],
                dim=0,
            )
            x_t = F.pad(x_t, (0, pad_t))  # -> [C, min_T]
            T = min_T
        else:
            input_mask = torch.ones(T, dtype=torch.bool)

        return {
            "input_values": x_t,  # [C, T]
            "input_mask": input_mask,  # [T]
            "target_values": torch.tensor(y, dtype=torch.long),  # []
            "target_weights": torch.tensor(w, dtype=torch.float32),
            "model_hints": {
                "in_chans": C,
                "in_times": T,
                "filter_time_length": self.filter_time_length,
                "pool_time_length": self.pool_time_length,
                "pool_time_stride": self.pool_time_stride,
                "min_time_required": min_T,
            },
        }

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: FloatTensor of shape [B, C, T] or [B, C, T, 1].

        Returns:
            FloatTensor [B, n_classes]: classification logits.
        """
        # accept (B, C, T) or (B, C, T, 1)
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # -> (B, C, T, 1)

        # reorder to (B, 1, T, C) for Conv2d over time then spatial
        x = x.permute(0, 3, 2, 1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # safe_log
        x = self.dropout(x)
        x = self.conv_classifier(x)
        x = self.logsoftmax(x)
        x = x.squeeze(3).squeeze(2)  # (B, n_classes)
        return x
