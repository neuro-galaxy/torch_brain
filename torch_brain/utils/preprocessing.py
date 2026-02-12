"""
Preprocessing utilities for EEG and other temporal signals.

These functions provide reusable normalization and scaling operations
that can be applied to [C, T] or [B, C, T] tensors.

All functions are differentiable and implemented in pure PyTorch.

Available functions
-------------------
- **z_score_normalize**: standard per-channel z-score normalization.
- **mean_center**: removes DC bias (zero-centers each channel).
- **min_max_normalize**: rescales each channel to a [min, max] range.
- **robust_scale**: scales by median and interquartile range (IQR).

Example
-------
>>> from torch_brain.utils.preprocessing import (
...     z_score_normalize,
...     mean_center,
...     min_max_normalize,
...     robust_scale,
... )
>>> x = torch.randn(32, 64, 250)  # [B, C, T]
>>> x_z = z_score_normalize(x)
>>> x_mm = min_max_normalize(x)

"""

import torch


def z_score_normalize(data: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """
    Standard (z-score) normalization per channel over the last dimension.

    Args:
        data (Tensor): [C, T] or [B, C, T] tensor.
        eps (float): Small constant to avoid divide-by-zero.

    Returns:
        Tensor: Normalized tensor of same shape.
    """
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, unbiased=False, keepdim=True)
    return (data - mean) / (std + eps)


def mean_center(data: torch.Tensor) -> torch.Tensor:
    """
    Subtracts the mean from each channel (zero-centering) but does not scale.

    Args:
        data (Tensor): [C, T] or [B, C, T].

    Returns:
        Tensor: Zero-centered tensor.
    """
    mean = data.mean(dim=-1, keepdim=True)
    return data - mean


def min_max_normalize(
    data: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0, eps: float = 1e-11
) -> torch.Tensor:
    """
    Scales each channel to the [min_val, max_val] range.

    Args:
        data (Tensor): [C, T] or [B, C, T].
        min_val (float): Lower bound of output range.
        max_val (float): Upper bound of output range.
        eps (float): Small constant for numerical stability.

    Returns:
        Tensor: Range-scaled tensor.
    """
    d_min = data.min(dim=-1, keepdim=True).values
    d_max = data.max(dim=-1, keepdim=True).values
    normed = (data - d_min) / (d_max - d_min + eps)
    return normed * (max_val - min_val) + min_val


def robust_scale(data: torch.Tensor, eps: float = 1e-11) -> torch.Tensor:
    """
    Robust scaling based on median and IQR (interquartile range).

    Args:
        data (Tensor): [C, T] or [B, C, T].
        eps (float): Small constant to avoid divide-by-zero.

    Returns:
        Tensor: Scaled tensor where each channel has roughly unit IQR.
    """
    median = data.median(dim=-1, keepdim=True).values
    q75 = data.quantile(0.75, dim=-1, keepdim=True)
    q25 = data.quantile(0.25, dim=-1, keepdim=True)
    iqr = q75 - q25
    return (data - median) / (iqr + eps)
