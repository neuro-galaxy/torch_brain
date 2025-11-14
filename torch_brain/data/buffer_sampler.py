import math
import logging
from typing import List, Dict, Tuple, Optional, TypeVar, Iterator, Iterable
from functools import cached_property

import torch
import torch.distributed as dist

from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex
import numpy as np

class BufferedSampler(torch.utils.data.Sampler):
    """
    Wraps any existing sampler that yields DatasetIndex(session, start, end) and returns (session, start-buffer, end+buffer) instead. 
    If the buffered window would fall outside the original sampling interval, that window will be dropped.

    Args:
        base_sampler: sampler (DatasetIndex)
        sampling_intervals: Dict[session, Interval], where Interval has
            start and end iterables of same length (same as original sampler uses).
        buffer: float (same units as starts/ends) 
        drop_overflow: if True (default), buffered windows that exceed interval bounds are dropped.
                      (set False if you want clipping.)
    """
    def __init__(self,
        *,
        base_sampler,
        sampling_intervals: Dict[str, 'Interval'],
        buffer_len: float,
        drop_overflow: bool = True,
    ):
        self.base_sampler = base_sampler
        self.sampling_intervals = sampling_intervals
        self.buffer_len = float(buffer_len)
        self.drop_overflow = drop_overflow
        self.buffered_indices = None

        self._interval_map = {}
        for session_name, sampling_intervals in self.sampling_intervals.items():
            self._interval_map[session_name] = list(zip(sampling_intervals.start, sampling_intervals.end))

    def __iter__(self):
        if self.buffered_indices is not None:
            for idx in self.buffered_indices:
                yield idx
            return
        
        self.buffered_indices = []
        for base_idx in self.base_sampler:
            sess = base_idx.session_name
            orig_start = float(base_idx.start)
            orig_end = float(base_idx.end)

            new_start = orig_start - self.buffer_len
            new_end = orig_end + self.buffer_len

            # find which interval the sample belongs to
            intervals = self._interval_map.get(sess, [])
            if not intervals:
                raise ValueError(f"Sample interval for session '{sess}' is empty!")
            intervals_array = np.array(intervals)  # shape (N,2)
            starts = intervals_array[:, 0]  
            ends   = intervals_array[:, 1]
            mask = (orig_start >= starts) & (orig_end <= ends)
            index = np.where(mask)[0]

            if index.size == 0:
                continue

            # check if buffered window fits inside the containing interval
            s_i, e_i = float(starts[index[0]]), float(ends[index[0]])
            if (new_start >= s_i) and (new_end <= e_i):
                idx = DatasetIndex(sess, new_start, new_end)
                self.buffered_indices.append(idx)
                yield idx
            else:
                # overflow
                if self.drop_overflow:  # drop this epoch
                    continue
                else:
                    # clip to bounds if user wanted clipping instead of drop
                    clipped_start = max(new_start, s_i)
                    clipped_end = min(new_end, e_i)
                    # ensure clipped_end > clipped_start and preserves minimum length
                    if clipped_end > clipped_start:
                        idx = DatasetIndex(sess, clipped_start, clipped_end)
                        self.buffered_indices.append(idx)
                        yield idx
                    else: # effectively too small, drop
                        continue

    def __len__(self):
        # compute once if not cached
        if self.buffered_indices is None:
            self.buffered_indices = list(self.__iter__())
        return len(self.buffered_indices)
