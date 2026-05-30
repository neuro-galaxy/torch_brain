import copy
from collections import namedtuple
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data._utils.collate import collate as _collate, default_collate_fn_map

import numpy as np


# pad
PaddedObject = namedtuple("PaddedObject", ["obj"])


def pad(obj):
    r"""Wrap an object to specify that it (or any of its members) should be padded to
    the maximum length in the batch. The object can be any of the objects that PyTorch's
    :obj:`default_collate` already supports.

    Args:
        obj: Can be tensors, numpy arrays, lists, tuples, or dictionaries.
    """
    return PaddedObject(obj)


def track_mask(input: Union[torch.Tensor, np.ndarray]):
    r"""Wrap an array or tensor to specify that its padding mask should be tracked.

    Args:
        input: An array or tensor.
    """
    return pad(torch.ones((len(input)), dtype=torch.bool))


def pad_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # todo this will be more optimal than any code we'll write? it's in C++
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


pad_collate_fn_map = copy.deepcopy(default_collate_fn_map)
pad_collate_fn_map[torch.Tensor] = pad_collate_tensor_fn


def pad_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate([e.obj for e in batch], collate_fn_map=pad_collate_fn_map)


# pad8
Padded8Object = namedtuple("Padded8Object", ["obj"])


def pad8(obj):
    r"""Wrap an object to specify that it (or any of its members) should be padded to
    the maximum length in the batch. This function is similar to :obj:`pad` except that
    the padding length is rounded up to the nearest multiple of 8.

    Args:
        obj: Can be tensors, numpy arrays, lists, tuples, or dictionaries.
    """
    return Padded8Object(obj)


def track_mask8(input: Union[torch.Tensor, np.ndarray]):
    r"""Wrap an array or tensor to specify that its padding mask should be tracked. This
    is used in conjunction with :obj:`pad8`.

    Args:
        input: An array or tensor.
    """
    return pad8(torch.ones((len(input)), dtype=torch.bool))


def pad8_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    max_len = max([elem.shape[0] for elem in batch])

    if max_len % 8 == 0:
        return pad_collate_tensor_fn(batch)

    elem = batch[0]
    batch.append(
        torch.zeros(
            (max_len + 8 - (max_len % 8), *elem.shape[1:]), dtype=batch[0].dtype
        )
    )

    return pad_collate_tensor_fn(batch)[:-1]


pad8_collate_fn_map = copy.deepcopy(default_collate_fn_map)
pad8_collate_fn_map[torch.Tensor] = pad8_collate_tensor_fn


def pad8_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate([e.obj for e in batch], collate_fn_map=pad8_collate_fn_map)


# chain
ChainObject = namedtuple("ChainObject", ["obj", "allow_missing_keys"])
ChainBatchTrackerObject = namedtuple("ChainBatchTrackerObject", ["obj"])


def chain(obj, allow_missing_keys: bool = False):
    r"""Wrap an object to specify that it (or any of its members) should be stacked
    along the first dimension when batching. This approach is similar to PyTorch
    Geometric's collate approach for graphs. This function will chain all the sequences
    in the batch into one large sequence. To track which sample from the batch an
    element of the sequence came from use :func:`track_batch`.

    Args:
        obj: Can be tensors, numpy arrays, lists, tuples, or dictionaries.
        allow_missing_keys: If set to :obj:`True`, this allows the wrapped dictionary to
            have inconsistent keys across the batch. If an instance is missing keys,
            the corresponding values will be skipped when collating. Defaults to
            :obj:`False`.
    """
    if allow_missing_keys and not isinstance(obj, dict):
        raise TypeError(
            f"allow_missing_keys can only be used with dictionaries, got {type(obj)}."
        )
    return ChainObject(obj, allow_missing_keys)


def track_batch(input: Union[torch.Tensor, np.ndarray]):
    r"""Wrap an array or tensor to track the batch_index.

    Args:
        input: An array or tensor.
    """
    return ChainBatchTrackerObject(torch.ones((len(input)), dtype=torch.long))


def chain_collate_str_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # No-op.
    return batch


def chain_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return torch.cat(batch, dim=0)


def chain_batch_tracker_collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    return _collate(
        [i * e.obj for i, e in enumerate(batch)],
        collate_fn_map=chain_collate_fn_map,
    )


chain_collate_fn_map = copy.deepcopy(default_collate_fn_map)
chain_collate_fn_map[str] = chain_collate_str_fn
chain_collate_fn_map[torch.Tensor] = chain_collate_tensor_fn
chain_collate_fn_map[ChainBatchTrackerObject] = chain_batch_tracker_collate_tensor_fn


def chain_collate_object_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    allow_missing_keys = batch[0].allow_missing_keys
    # check that flag is consistent
    if any((elem.allow_missing_keys != allow_missing_keys) for elem in batch):
        raise ValueError(
            "attribute 'allow_missing_keys' must be the same for all elements in the "
            "batch. Some elements allow missing keys while others do not."
        )

    if not allow_missing_keys:
        return _collate(
            [elem.obj for elem in batch], collate_fn_map=chain_collate_fn_map
        )
    else:
        unique_keys = set().union(*[elem.obj.keys() for elem in batch])
        return {
            key: _collate(
                [elem.obj[key] for elem in batch if key in elem.obj],
                collate_fn_map=chain_collate_fn_map,
            )
            for key in unique_keys
        }


# add all new types to collate fn map
# note that once a recipe is selected for a given object it cannot be overwritten
# this is to avoid the following example scenario where pad(obj) is called but obj a
# dict and one of its values was already wrapped in chain.
collate_fn_map = copy.deepcopy(default_collate_fn_map)
collate_fn_map[PaddedObject] = pad_collate_object_fn
collate_fn_map[Padded8Object] = pad8_collate_object_fn
collate_fn_map[ChainObject] = chain_collate_object_fn
collate_fn_map[ChainBatchTrackerObject] = chain_batch_tracker_collate_tensor_fn


def collate(batch):
    r"""Extension of PyTorch's :obj:`default_collate` function to enable more advanced
    collation of samples of variable lengths.

    To specify how the collation recipe, wrap the objects using :obj:`pad` or :obj:`chain`.
    If the wrapped object is an Iterable or Mapping, all its elements will inherite
    the collation recipe. All objects that are already supported by :obj:`default_collate`
    can be wrapped.

    If an object is not wrapped, the default collation recipe will be used. i.e. the
    outcome will be identical to :obj:`default_collate`.

    :obj:`pad` or :obj:`chain` do not track any padding masks or batch index, since that might not
    always be needed. Use :obj:`track_mask` or :obj:`track_batch` to track masks or batch index
    for a particular array or tensor.

    Args:
        batch: a single batch to be collated
    """
    return _collate(batch, collate_fn_map=collate_fn_map)
