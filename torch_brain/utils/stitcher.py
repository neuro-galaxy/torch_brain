import torch


def stitch(
    timestamps: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Pools values that share the same timestamp using mean or mode operations.

    This function is useful when you have multiple predictions or values for the same
    timestamp (e.g., from overlapping windows) and need to combine them into a single
    value per timestamp.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps. Shape: (N,)
        values (torch.Tensor): A tensor of values corresponding to the timestamps.
            Shape:
                - For floating point types: (N, ...)
                - For categorical types (torch.long only): (N,)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - unique_timestamps: A 1D tensor of sorted unique timestamps
            - pooled_values: A tensor containing the pooled values for each unique timestamp
                - For continuous data (float types): Uses mean pooling
                - For categorical data (long type): Uses mode pooling

    Examples:
        >>> # Mean pooling for continuous values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3])
        >>> values = torch.tensor([0.1, 0.3, 0.2, 0.4, 0.6])
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([0.2000, 0.2000, 0.5000]))

        >>> # Mode pooling for categorical values
        >>> timestamps = torch.tensor([1, 1, 2, 3, 3, 3])
        >>> values = torch.tensor([1, 1, 2, 3, 3, 1], dtype=torch.long)
        >>> stitch(timestamps, values)
        (tensor([1, 2, 3]), tensor([1, 2, 3]))
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    if values.dtype == torch.long:
        # Mode pooling for categorical values

        if values.ndim != 1:
            raise ValueError(
                "For categorical values (long type), only 1D tensors are supported. "
                f"Got values with shape {values.shape} instead."
            )

        # 1. Construct a N x C class-wise vote tensor
        votes = values.new_zeros((len(unique_timestamps), values.max() + 1))
        votes.index_put_((indices, values), torch.ones_like(indices), accumulate=True)
        # 2. Mode class is the one with most votes
        mode_values = votes.argmax(dim=-1)
        return unique_timestamps, mode_values

    elif torch.is_floating_point(values):
        # Mean-pool for floating points
        # 1. Count occurrences of each unique timestamp
        counts = torch.zeros_like(unique_timestamps, dtype=torch.long)
        counts.index_add_(0, indices, torch.ones_like(indices))
        if values.dim() > 1:
            counts = counts.unsqueeze(-1)
        # 2. Accumulate and average values for each unique timestamp
        avg_values = values.new_zeros((len(unique_timestamps), *values.shape[1:]))
        avg_values.index_add_(0, indices, values).div_(counts)
        # Regarding division by zero: all elements of counts will be >= 1.
        # Reasoning: Since it was built using unique_timestamps, each index will have
        # atleast one timestamp attached to it.

        return unique_timestamps, avg_values

    else:
        raise TypeError(
            f"Unsupported dtype {values.dtype} for stitching. "
            "Only floating points supported for mean pooling, "
            " and torch.long type supported for mode pooling."
        )



def _deprecated_import_error(name):
    raise ImportError(
        f"`{name}` has been moved to `torch_brain.utils.callbacks`. "
        f"Please update your import to: `from torch_brain.utils.callbacks import {name}`"
    )


class DecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DecodingStitchEvaluator")


class DataForDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DataForDecodingStitchEvaluator")


class MultiTaskDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("MultiTaskDecodingStitchEvaluator")


class DataForMultiTaskDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DataForMultiTaskDecodingStitchEvaluator")
