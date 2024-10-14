from collections import defaultdict
import torch
import torch.distributed as dist

r"""Utility functions for validation and testing.
"""


def all_gather_dict_of_dict_of_tensor(obj):
    r"""All-gather and concatenate dictionary-of-dictionary-of-tensor objects
    Args:
        obj (dict): A dictionary of dictionaries of tensors to be gathered.
    """
    gathered_objlist = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_objlist, obj)

    # Concatenate all tensors in the dictionaries
    gathered_obj = defaultdict(lambda: defaultdict(list))
    for objlist in gathered_objlist:
        for outer_key, inner_dict in objlist.items():
            for inner_key, tensor in inner_dict.items():
                gathered_obj[outer_key][inner_key].append(tensor)

    # now actually concatenate the tensors in the innermost lists
    for outer_key, inner_dict in gathered_obj.items():
        for inner_key, tensor_list in inner_dict.items():
            gathered_obj[outer_key][inner_key] = torch.cat(tensor_list)

    dist.barrier()
    return gathered_obj


def avg_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    r"""This function performs pooling operations (mean or mode) on a tensor based on
    unique timestamps and the datatype of the values.

    Args:
        timestamps (torch.Tensor): A 1D tensor containing timestamps.
        values (torch.Tensor): A tensor of values that correspond to the timestamps. It
            expects a tensor of shape (N, ...), where N is the number of timestamps.

    Returns:
        torch.Tensor: A tensor with the pooled values for each unique timestamp. If the
          values are continuous, the function performs mean pooling, averaging the
          values for each unique timestamp. If the values are categorical (labels),
          the function returns the mode of the values for each unique timestamp.

    Note:
        For mean pooling, this function leverages `torch.scatter_add_` to efficiently
        aggregate values for each unique timestamp
    """
    # Find unique timestamps and their inverse indices
    unique_timestamps, indices = torch.unique(
        timestamps, return_inverse=True, sorted=True
    )

    # Prepare a tensor for summing values for each unique timestamp
    pooled_sum = torch.zeros(
        (len(unique_timestamps), *values.shape[1:]),
        device=values.device,
        dtype=values.dtype,
    )

    # Use mode for integers
    if values.dtype == torch.long:
        # NOT IDEAL, IT IS FASTER TO AVERAGE THE LOGITS THAN TO PERFORM A VOTE
        mode_values = torch.zeros_like(pooled_sum)
        for i, timestamp in enumerate(unique_timestamps):
            mask = timestamps == timestamp
            group_values = values[mask]
            mode, _ = torch.mode(group_values, dim=0)
            mode_values[i] = mode
        return mode_values

    # Count occurrences of each unique timestamp
    counts = torch.zeros(
        len(unique_timestamps), device=timestamps.device, dtype=values.dtype
    )
    counts = counts.scatter_add_(
        0, indices, torch.ones_like(indices, dtype=values.dtype)
    )
    # Accumulate values for each unique timestamp
    indices_expanded = indices.unsqueeze(-1).expand_as(values)
    pooled_sum.scatter_add_(0, indices_expanded, values)
    # Calculate the average
    epsilon = 1e-8  # small constant to prevent division by zero
    averages = torch.div(pooled_sum, counts.unsqueeze(-1) + epsilon)

    return averages


def gt_pool(timestamps: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    r"""Wrapper over `avg_pool` specifically for pooling ground truth categorical
    values.
    """
    return (
        torch.round(avg_pool(timestamps, values.float().view(-1, 1))).long().squeeze()
    )
