from __future__ import annotations

import copy
from typing import List, Tuple, Union
import logging

import h5py
import numpy as np
import pandas as pd

from .arraydict import ArrayDict


class Interval(ArrayDict):
    r"""An interval object is a set of time intervals each defined by a start time and
    an end time. For :obj:`Interval`, we do not need to define a domain, since the
    interval itself is its own domain.

    Args:
        start: an array of start times of shape (N,) or a float.
        end: an array of end times of shape (N,) or a float.
        timekeys: a list of strings that specify which attributes are time-based
            attributes.
        **kwargs: arrays that shares the same first dimension N.

    Example ::

        >>> import numpy as np
        >>> from temporaldata import Interval

        >>> intervals = Interval(
        ...    start=np.array([0., 1., 2.]),
        ...    end=np.array([1., 2., 3.]),
        ...    go_cue_time=np.array([0.5, 1.5, 2.5]),
        ...    drifting_gratings_dir=np.array([0, 45, 90]),
        ...    timekeys=["start", "end", "go_cue_time"],
        ... )

        >>> intervals
        Interval(
          start=[3],
          end=[3],
          go_cue_time=[3],
          drifting_gratings_dir=[3]
        )

        >>> intervals.keys()
        ['start', 'end', 'go_cue_time', 'drifting_gratings_dir']

        >>> intervals.is_sorted()
        True

        >>> intervals.is_disjoint()
        True

        >>> intervals.slice(1.5, 2.5)
        Interval(
          start=[2],
          end=[2],
          go_cue_time=[2],
          drifting_gratings_dir=[2]
        )

    An :obj:`Interval` object with a single interval can be simply created by passing
    a single float to the :obj:`start` and :obj:`end` arguments.

    Example ::

        >>> Interval(0., 1.)
        Interval(
          start=[1],
          end=[1]
        )

    """

    _sorted = None
    _timekeys = None
    _allow_split_mask_overlap = False

    def __init__(
        self,
        start: Union[float, np.ndarray],
        end: Union[float, np.ndarray],
        *,
        timekeys=None,
        **kwargs,
    ):
        # we allow for scalar start and end, since it is common to have a single
        # interval especially when defining a domain
        if isinstance(start, (int, float)):
            start = np.array([start], dtype=np.float64)

        if isinstance(end, (int, float)):
            end = np.array([end], dtype=np.float64)

        super().__init__(start=start, end=end, **kwargs)

        # time keys
        if timekeys is None:
            timekeys = []
        if "start" not in timekeys:
            timekeys.append("start")
        if "end" not in timekeys:
            timekeys.append("end")
        for key in timekeys:
            assert key in self.keys(), f"Time attribute {key} not found in data."

        self._timekeys = timekeys

    def timekeys(self):
        r"""Returns a list of all time-based attributes."""
        return self._timekeys

    def register_timekey(self, timekey: str):
        r"""Register a new time-based attribute."""
        if timekey not in self.keys():
            raise ValueError(f"'{timekey}' cannot be found in \n {self}.")
        if timekey not in self._timekeys:
            self._timekeys.append(timekey)

    def __setattr__(self, name, value):
        super(Interval, self).__setattr__(name, value)

        if name == "start" or name == "end":
            assert value.ndim == 1, f"{name} must be 1D."
            assert ~np.isnan(value).any(), f"{name} cannot contain NaNs."
            if value.dtype != np.float64:
                logging.warning(f"{name} is of type {value.dtype} not of type float64.")
            # start or end have been updated, we no longer know whether it is sorted
            # or not
            self._sorted = None

    def __iter__(self):
        r"""Iterates over the intervals. Will return a tuple of (start, end).
        This iterator will not include other optional attributes.

        .. Example ::

            >>> import numpy as np
            >>> from temporaldata import Interval

            >>> intervals = Interval(
            ...     start=np.array([0., 1., 2.]),
            ...     end=np.array([1., 2., 3.]),
            ...     some_other_attribute=np.array([0, 1, 2]),
            ... )

            >>> for start, end in intervals:
            ...     print(start, end)
            0.0 1.0
            1.0 2.0
            2.0 3.0
        """
        for s, e in zip(self.start, self.end):
            yield (s, e)

    def is_disjoint(self):
        r"""Returns :obj:`True` if the intervals are disjoint, i.e. if no two intervals
        overlap."""
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if not self.is_sorted():
            # make a copy and sorted it
            tmp_copy = copy.deepcopy(self)
            # attempt to sort it, this will fail if interval is not disjoint
            try:
                tmp_copy.sort()
            except ValueError:
                # ValueError is returned if intervals are not disjoint
                return False
            return tmp_copy.is_disjoint()
        return bool((self.end[:-1] <= self.start[1:]).all())

    def is_sorted(self):
        r"""Returns :obj:`True` if the intervals are sorted."""
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None:
            self._sorted = bool(
                (self.start[1:] >= self.start[:-1]).all()
                and (self.end[1:] >= self.end[:-1]).all()
            )
        return self._sorted

    def sort(self):
        r"""Sorts the intervals, and reorders the other attributes accordingly.
        This method is done in place.

        .. note:: This method only works if the intervals are disjoint. If the intervals
            overlap, it is not possible to resolve the order of the intervals, and this
            method will raise an error.
        """
        if not self.is_sorted():
            sorted_indices = np.argsort(self.start)
            for key in self.keys():
                self.__dict__[key] = self.__dict__[key][sorted_indices]
        self._sorted = True

        if not self.is_disjoint():
            raise ValueError("Intervals must be disjoint.")
        return self

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`Interval` object that contains the data between the
        start and end times. An interval is included if it has any overlap with the
        slicing window. The end time is exclusive.

        If :obj:`reset_origin` is set to :obj:`True`, all time attributes will be
        updated to be relative to the new start time.

        .. warning::
            If the intervals are not sorted, they will be automatically sorted in place.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
        """

        if not self.is_sorted():
            self.sort()

        # anything that starts before the end of the slicing window
        idx_l = np.searchsorted(self.end, start, side="right")

        # anything that will end after the start of the slicing window
        idx_r = np.searchsorted(self.start, end)

        out = self.__class__.__new__(self.__class__)
        out._timekeys = self._timekeys

        for key in self.keys():
            out.__dict__[key] = self.__dict__[key][idx_l:idx_r].copy()

        if reset_origin:
            for key in self._timekeys:
                out.__dict__[key] = out.__dict__[key] - start
        return out

    def select_by_mask(self, mask: np.ndarray):
        r"""Return a new :obj:`Interval` object where all array attributes
        are indexed using the boolean mask.
        """
        out = super().select_by_mask(mask, timekeys=self._timekeys)
        out._sorted = self._sorted
        return out

    def select_by_interval(self, interval: Interval):
        r"""Return a new :obj:`IrregularTimeSeries` object where all timestamps are
        within the interval.

        Args:
            interval: Interval object.
        """

        idx_l = np.searchsorted(self.end, interval.start, side="right")
        idx_r = np.searchsorted(self.start, interval.end)

        mask = np.zeros(len(self), dtype=bool)
        for i in range(len(interval)):
            mask[idx_l[i] : idx_r[i]] = True

        out = self.select_by_mask(mask)
        return out

    def dilate(self, size: float, max_len=None):
        r"""Dilates the intervals by a given size. The dilation is performed in both
        directions. This operation is designed to not create overlapping intervals,
        meaning for a given interval and a given direction, dilation will stop if
        another interval is too close. If distance between two intervals is less than
        :obj:`size`, both of them will dilate until they meet halfway but will never
        overlap. You can think of dilation as inflating ballons that will never merge,
        and will stop each other from moving too far.

        Args:
            size: The size of the dilation.
            max_len: Dilation will not exceed this maximum length. For intervals that
                are already longer than :obj:`max_len`, there will be no dilation. By
                default, there is no maximum length.
        """
        out = copy.deepcopy(self)

        if len(out) == 0:
            # empty interval, nothing to dilate
            return out

        dilation_size = size
        size = np.full_like(out.start, dilation_size)
        if max_len is not None:
            interval_len = out.end - out.start
            size = np.minimum(size, (max_len - interval_len) / 2)
            size = np.clip(size, 0, None)

        half_way = (self.end[:-1] + self.start[1:]) / 2

        # TODO(mehdi) should check that this does not violate domain
        out.start[0] = out.start[0] - size[0]
        out.start[1:] = np.maximum(out.start[1:] - size[1:], half_way)

        # update size
        size = np.full_like(out.start, dilation_size)
        if max_len is not None:
            interval_len = out.end - out.start
            size = np.minimum(size, (max_len - interval_len))
            size = np.clip(size, 0, None)

        out.end[:-1] = np.minimum(self.end[:-1] + size[:-1], half_way)
        out.end[-1] = out.end[-1] + size[-1]
        return out

    def coalesce(self, eps=1e-6):
        r"""Coalesces the intervals that are closer than :obj:`eps`. This operation
        returns a new :obj:`Interval` object, and does not resolve the existing
        attributes.

        Args:
            eps: The distance threshold for coalescing the intervals. Defaults to 1e-6.
        """
        if not self.is_sorted():
            self.sort()

        start = []
        end = []

        current_start = self.start[0]
        current_end = self.end[0]

        for s, e in zip(self.start[1:], self.end[1:]):
            if s - current_end < eps:
                # we have an overlap
                current_end = e
            else:
                start.append(current_start)
                end.append(current_end)
                current_start = s
                current_end = e

        start.append(current_start)
        end.append(current_end)

        return Interval(start=np.array(start), end=np.array(end))

    def difference(self, other):
        r"""Returns the difference between two sets of intervals. The intervals are
        redefined as to not intersect with any interval in :obj:`other`.
        """
        if not self.is_disjoint():
            raise ValueError("left Interval object must be disjoint.")
        if not other.is_disjoint():
            raise ValueError("right Interval object must be disjoint.")
        if not self.is_sorted():
            raise ValueError("left Interval object must be sorted.")
        if not other.is_sorted():
            raise ValueError("right Interval object must be sorted.")

        # new start and end arrays where the intersection will be stored
        start = np.array([])
        end = np.array([])

        # we use a variable to store the current opening time
        current_start = None
        interval_open_left = False
        interval_open_right = False

        for ptime, pop, pl in sorted_traversal(self, other):
            if pop:
                # opening
                if pl:
                    if not interval_open_right:
                        current_start = ptime
                    interval_open_left = True
                else:
                    interval_open_right = True
                    # we have an opening and a closing paranthesis
                    if (
                        interval_open_left
                        and current_start is not None
                        and current_start != ptime
                    ):
                        # we have a non-zero interval
                        start = np.append(start, current_start)
                        end = np.append(end, ptime)
                    current_start = None
            else:
                # closing
                if pl:
                    if current_start is not None and current_start != ptime:
                        # we have a non-zero interval
                        start = np.append(start, current_start)
                        end = np.append(end, ptime)
                        current_start = None
                    interval_open_left = False
                else:
                    interval_open_right = False
                    if interval_open_left:
                        current_start = ptime

        return Interval(start=start, end=end)

    def split(
        self,
        sizes: Union[List[int], List[float]],
        *,
        shuffle=False,
        random_seed=None,
    ):
        r"""Splits the set of intervals into multiple subsets. This will
        return a number of new :obj:`Interval` objects equal to the number of elements
        in `sizes`. If `shuffle` is set to :obj:`True`, the intervals will be shuffled
        before splitting.

        Args:
            sizes: A list of integers or floats.

                - **Integers**: The list must sum to the number of intervals.
                  Example: ``[60, 20, 20]`` for 100 intervals.

                - **Floats**: The list must sum to 1.0.
                  Example: ``[0.6, 0.2, 0.2]`` for a 60/20/20 split.

            shuffle: If :obj:`True`, the intervals will be shuffled before splitting.
            random_seed: The random seed to use for shuffling.

        Returns:
            A list of :obj:`Interval` objects, one for each element in ``sizes``.

        .. note::
            This method will not guarantee that the resulting sets will be disjoint, if
            the intervals are not already disjoint.

        Examples:
            Split 10 intervals into 60/20/20 sets using integers:

            >>> from temporaldata import Interval
            >>> intervals = Interval.linspace(0, 1, 10)
            >>> train, val, test = intervals.split([6, 2, 2])
            >>> print(len(train), len(val), len(test))
            6 2 2

            Split using proportions (floats):

            >>> intervals = Interval.linspace(0, 1, 100)
            >>> train, val, test = intervals.split([0.7, 0.15, 0.15])
            >>> print(len(train), len(val), len(test))
            70 15 15

            Split with shuffling:

            >>> intervals = Interval.linspace(0, 1, 100)
            >>> train, test = intervals.split(
            ...     [0.8, 0.2],
            ...     shuffle=True,
            ...     random_seed=42
            ... )
            >>> print(len(train), len(test))
            80 20
        """

        assert len(sizes) > 1, "must split into at least two sets"
        assert len(sizes) < len(self), f"cannot split {len(self)} intervals into "
        " {len(sizes)} sets"

        # if sizes are floats, convert them to integers
        if all(isinstance(x, float) for x in sizes):
            assert sum(sizes) == 1.0, "sizes must sum to 1.0"
            sizes = [round(x * len(self)) for x in sizes]
            # there might be rounding errors
            # make sure that the sum of sizes is still equal to the number of intervals
            largest = np.argmax(sizes)
            sizes[largest] = len(self) - (sum(sizes) - sizes[largest])
        elif all(isinstance(x, int) for x in sizes):
            assert sum(sizes) == len(self), "sizes must sum to the number of intervals"
        else:
            raise ValueError("sizes must be either all floats or all integers")

        # shuffle
        if shuffle:
            rng = np.random.default_rng(random_seed)  # Create a new generator instance
            idx = rng.permutation(len(self))  # Use the generator for permutation
        else:
            idx = np.arange(len(self))  # Create a sequential index array

        # split
        splits = []
        start = 0
        for size in sizes:
            mask = np.zeros(len(self), dtype=bool)
            mask[idx[start : start + size]] = True
            splits.append(self.select_by_mask(mask))
            start += size

        return splits

    def subdivide(
        self,
        step: float,
        drop_short: bool = False,
    ) -> Interval:
        r"""Subdivides each interval into fixed-duration segments while preserving
        attributes.

        If the last segment of an interval is shorter than :obj:`step`, it will be
        included by default. Set :obj:`drop_short` to :obj:`True` to exclude these
        partial segments. If an interval is shorter than :obj:`step`, it will be
        treated as a partial segment (kept if :obj:`drop_short` is :obj:`False`,
        dropped otherwise).

        Args:
            step: The duration of each segment.
            drop_short: If :obj:`True`, excludes segments shorter than :obj:`step`.
                Defaults to :obj:`False`.

        Returns:
            A new :obj:`Interval` object with the subdivided segments.

        Example ::

            >>> from temporaldata import Interval
            >>> import numpy as np

            >>> interval = Interval(
            ...     start=np.array([0.0, 20.0]),
            ...     end=np.array([10.0, 30.0]),
            ...     trial_id=np.array([1, 2])
            ... )
            >>> subdivided = interval.subdivide(2.5)
            >>> subdivided
            Interval(
              start=[8],
              end=[8],
              trial_id=[8]
            )
            >>> subdivided.trial_id
            array([1, 1, 1, 1, 2, 2, 2, 2])
        """
        if len(self) == 0:
            return copy.deepcopy(self)

        subdivided_intervals_starts = []
        subdivided_intervals_ends = []
        original_indices = []

        for i, (start, end) in enumerate(zip(self.start, self.end)):
            subdivided = Interval.arange(
                start, end, step=step, include_end=not drop_short
            )
            subdivided_intervals_starts.append(subdivided.start)
            subdivided_intervals_ends.append(subdivided.end)
            original_indices.extend([i] * len(subdivided))

        all_starts = np.concatenate(subdivided_intervals_starts)
        all_ends = np.concatenate(subdivided_intervals_ends)

        kwargs = {}
        for key in self.keys():
            if key in ["start", "end"]:
                continue
            val = getattr(self, key)
            kwargs[key] = val[original_indices]

        return Interval(
            start=all_starts, end=all_ends, timekeys=self.timekeys(), **kwargs
        )

    def add_split_mask(
        self,
        name: str,
        interval: Interval,
    ):
        """Adds a boolean mask as an array attribute, which is defined for each
        interval in the object, and is set to :obj:`True` if the interval intersects
        with the provided :obj:`Interval` object. The mask attribute will be called
        :obj:`<name>_mask`.

        This is used to mark intervals as part of train, validation,
        or test sets, and is useful to ensure that there is no data leakage.

        If an interval belongs to multiple splits, an error will be raised, unless this
        is expected, in which case the method :meth:`allow_split_mask_overlap` should be
        called.

        Args:
            name: name of the split, e.g. "train", "valid", "test".
            interval: a set of intervals defining the split domain.
        """
        assert f"{name}_mask" not in self.keys(), (
            f"Attribute {name}_mask already exists. Use another mask name, or rename "
            f"the existing attribute."
        )

        mask_array = np.zeros_like(self.start, dtype=bool)
        for start, end in zip(interval.start, interval.end):
            mask_array |= (self.start < end) & (self.end > start)

        setattr(self, f"{name}_mask", mask_array)

    def allow_split_mask_overlap(self):
        r"""Disables the check for split mask overlap. This means there could be an
        overlap between the intervals across different splits. This is useful when
        an interval is allowed to belong to multiple splits."""
        logging.warning(
            f"You are disabling the check for split mask overlap. "
            f"This means there could be an overlap between the intervals "
            f"across different splits. "
        )
        self._allow_split_mask_overlap = True

    @classmethod
    def linspace(cls, start: float, end: float, steps: int):
        r"""Create a regular interval with a given number of samples.

        Args:
            start: Start time.
            end: End time.
            steps: Number of samples.

        Example ::

            >>> from temporaldata import Interval

            >>> interval = Interval.linspace(0., 10., 100)

            >>> interval
            Interval(
              start=[100],
              end=[100]
            )
        """
        timestamps = np.linspace(start, end, steps + 1, dtype=np.float64)
        return cls(
            start=timestamps[:-1],
            end=timestamps[1:],
        )

    @classmethod
    def arange(cls, start: float, end: float, step: float, include_end: bool = True):
        r"""Create a grid of intervals with a given step size. If the last step cannot
        reach the end time, a smaller interval will be added, it will stop at the end
        time, and will be shorter than obj:`step`. This behavior can be
        changed by setting `include_end` to :obj:`False`.

        Args:
            start: Start time.
            end: End time.
            step: Step size.
            include_end: Whether to include a partial interval at the end.
        """
        whole_steps = np.floor((end - start) / step).astype(int)
        timestamps = np.linspace(
            start, start + whole_steps * step, whole_steps + 1, dtype=np.float64
        )

        if include_end and timestamps[-1] < end:
            timestamps = np.append(timestamps, end)

        return cls(
            start=timestamps[:-1],
            end=timestamps[1:],
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, unsigned_to_long: bool = True, **kwargs):
        r"""Create an :obj:`Interval` object from a pandas DataFrame. The dataframe
        must have a start time and end time columns. The names of these columns need
        to be "start" and "end" (use `pd.Dataframe.rename` if needed).

        The columns in the DataFrame are converted to arrays when possible, otherwise
        they will be skipped.

        Args:
            df (pandas.DataFrame): DataFrame.
            unsigned_to_long (bool, optional): Whether to automatically convert unsigned
              integers to int64 dtype. Defaults to :obj:`True`.
        """
        assert "start" in df.columns, f"Column 'start' not found in dataframe."
        assert "end" in df.columns, f"Column 'end' not found in dataframe."

        return super().from_dataframe(
            df,
            unsigned_to_long=unsigned_to_long,
            **kwargs,
        )

    @classmethod
    def from_list(cls, interval_list: List[Tuple[float, float]]):
        r"""Create an :obj:`Interval` object from a list of (start, end) tuples.

        Args:
            interval_list: List of (start, end) tuples.

        Example ::

            >>> from temporaldata import Interval

            >>> interval_list = [(0, 1), (1, 2), (2, 3)]
            >>> interval = Interval.from_list(interval_list)

            >>> interval.start, interval.end
            (array([0., 1., 2.]), array([1., 2., 3.]))
        """
        start, end = zip(*interval_list)  # Unzip the list of tuples
        return cls(
            start=np.array(start, dtype=np.float64),
            end=np.array(end, dtype=np.float64),
        )

    def to_hdf5(self, file):
        r"""Saves the data object to an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

                import h5py
                from temporaldata import Interval

                interval = Interval(
                    start=np.array([0, 1, 2]),
                    end=np.array([1, 2, 3]),
                    go_cue_time=np.array([0.5, 1.5, 2.5]),
                    drifting_gratins_dir=np.array([0, 45, 90]),
                    timekeys=["start", "end", "go_cue_time"],
                )

                with h5py.File("data.h5", "w") as f:
                    interval.to_hdf5(f)
        """
        _unicode_keys = []
        for key in self.keys():
            value = getattr(self, key)

            if value.dtype.kind == "U":  # if its a unicode string type
                try:
                    # convert string arrays to fixed length ASCII bytes
                    value = value.astype("S")
                except UnicodeEncodeError:
                    raise NotImplementedError(
                        f"Unable to convert column '{key}' from numpy 'U' string type "
                        "to fixed-length ASCII (np.dtype('S')). HDF5 does not support "
                        "numpy 'U' strings."
                    )
                # keep track of the keys of the arrays that were originally unicode
                _unicode_keys.append(key)
            file.create_dataset(key, data=value)

        file.attrs["_unicode_keys"] = np.array(_unicode_keys, dtype="S")
        file.attrs["timekeys"] = np.array(self._timekeys, dtype="S")
        file.attrs["allow_split_mask_overlap"] = self._allow_split_mask_overlap
        file.attrs["object"] = self.__class__.__name__

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyInterval.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from temporaldata import Interval

            with h5py.File("data.h5", "r") as f:
                interval = Interval.from_hdf5(f)
        """
        assert file.attrs["object"] == cls.__name__, "object type mismatch"
        data = {}
        _unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()
        for key, value in file.items():
            data[key] = value[:]
            # if the values were originally unicode but stored as fixed length ASCII bytes
            if key in _unicode_keys:
                data[key] = data[key].astype("U")
        timekeys = file.attrs["timekeys"].astype(str).tolist()
        obj = cls(**data, timekeys=timekeys)

        if file.attrs["allow_split_mask_overlap"]:
            obj.allow_split_mask_overlap()

        return obj

    def __and__(self, other):
        """Intersection of two intervals.
        Only start/end times are considered for the intersection,
        and only start/end times are returned in the resulting Interval
        """
        if not self.is_disjoint():
            raise ValueError("left Interval object must be disjoint.")
        if not other.is_disjoint():
            raise ValueError("right Interval object must be disjoint.")
        if not self.is_sorted():
            raise ValueError("left Interval object must be sorted.")
        if not other.is_sorted():
            raise ValueError("right Interval object must be sorted.")

        # new start and end arrays where the intersection will be stored
        start = np.array([])
        end = np.array([])

        # we use a variable to store the current opening time
        current_start = None
        interval_open_left = False
        interval_open_right = False
        for ptime, pop, pl in sorted_traversal(self, other):
            if pop:
                # this is an opening paranthesis
                # update current_start
                current_start = ptime
                if pl:
                    interval_open_left = True
                else:
                    interval_open_right = True
            else:
                # this is a closing paranthesis
                if (
                    current_start is not None
                    and interval_open_left
                    and interval_open_right
                ):
                    # we have an opening and a closing paranthesis
                    if current_start != ptime:
                        # we have a non-zero interval
                        start = np.append(start, current_start)
                        end = np.append(end, ptime)
                    current_start = None
                if pl:
                    interval_open_left = False
                else:
                    interval_open_right = False

        return Interval(start=start, end=end)

    def __or__(self, other):
        """Union of two intervals.
        Only start/end times are considered for the union,
        and only start/end times are returned in the resulting Interval
        """
        if not self.is_disjoint():
            raise ValueError("left Interval object must be disjoint.")
        if not other.is_disjoint():
            raise ValueError("right Interval object must be disjoint.")
        if not self.is_sorted():
            raise ValueError("left Interval object must be sorted.")
        if not other.is_sorted():
            raise ValueError("right Interval object must be sorted.")

        # new start and end arrays where the intersection will be stored
        start = np.array([])
        end = np.array([])

        # we use a variable to store the current opening time
        current_start = None
        current_end = None
        current_start_is_from_left = None
        end_still_coming = False

        for ptime, pop, pl in sorted_traversal(self, other):
            if pop:
                if current_end is None:
                    if current_start is None:
                        current_start = ptime
                        current_start_is_from_left = pl
                        end_still_coming = True
                else:
                    assert current_start is not None
                    if not end_still_coming:
                        # Check if this opening time matches the previous closing time
                        # If they match, continue the current interval instead of creating a new one
                        if ptime != current_end:
                            # we have an opening and a closing paranthesis
                            if current_start != current_end:
                                # we have a non-zero interval
                                start = np.append(start, current_start)
                                end = np.append(end, current_end)
                            current_start = ptime
                            current_end = None
                        end_still_coming = True
                        current_start_is_from_left = pl
            else:
                if pl == current_start_is_from_left:
                    end_still_coming = False
                current_end = ptime

        assert current_end is not None
        assert current_start is not None

        # we have an opening and a closing paranthesis
        if current_start != current_end:
            # we have a non-zero interval
            start = np.append(start, current_start)
            end = np.append(end, current_end)

        return Interval(start=start, end=end)


class LazyInterval(Interval):
    r"""Lazy variant of :obj:`Interval`. The data is not loaded until it is accessed.
    This class is meant to be used when the data is too large to fit in memory, and
    is intended to be intantiated via. :obj:`LazyInterval.from_hdf5`.

    .. note:: To access an attribute without triggering the in-memory loading use
        self.__dict__[key] otherwise using self.key or getattr(self, key) will trigger
        the lazy loading and will automatically convert the h5py dataset to a numpy
        array as well as apply any outstanding masks.
    """

    _lazy_ops = dict()
    _unicode_keys = []

    def _maybe_first_dim(self):
        if "unresolved_slice" in self._lazy_ops:
            return self.start.shape[0]
        elif "mask" in self._lazy_ops:
            return self._lazy_ops["mask"].sum()
        elif isinstance(self.__dict__["start"], np.ndarray):
            return self.start.shape[0]
        return super()._maybe_first_dim()

    def __getattribute__(self, name):
        if not name in ["__dict__", "keys"]:
            # intercept attribute calls
            if name in self.keys():
                out = self.__dict__[name]

                if isinstance(out, h5py.Dataset):
                    # convert into numpy array
                    if "unresolved_slice" in self._lazy_ops:
                        self._resolve_start_end_after_slice()
                    if "slice" in self._lazy_ops:
                        idx_l, idx_r, start, origin_translation = self._lazy_ops[
                            "slice"
                        ]
                        out = out[idx_l:idx_r]
                        if name in self._timekeys:
                            out = out - origin_translation
                    if "mask" in self._lazy_ops:
                        out = out[self._lazy_ops["mask"]]
                    if len(self._lazy_ops) == 0:
                        out = out[:]

                    if name in self._unicode_keys:
                        # convert back to unicode
                        out = out.astype("U")

                    # store it
                    self.__dict__[name] = out

                # If all attributes are loaded, we can remove the lazy flag
                all_loaded = all(
                    isinstance(self.__dict__[key], np.ndarray) for key in self.keys()
                )
                if all_loaded:
                    self.__class__ = Interval
                    del self._lazy_ops, self._unicode_keys

                return out
        return super(LazyInterval, self).__getattribute__(name)

    def select_by_mask(self, mask: np.ndarray):
        assert mask.ndim == 1, f"mask must be 1D, got {mask.ndim}D mask"
        assert mask.dtype == bool, f"mask must be boolean, got {mask.dtype}"

        first_dim = self._maybe_first_dim()
        if mask.shape[0] != first_dim:
            raise ValueError(
                f"mask length {mask.shape[0]} does not match first dimension of arrays "
                f"({first_dim})."
            )

        # make a copy
        out = self.__class__.__new__(self.__class__)
        out._unicode_keys = self._unicode_keys
        out._timekeys = self._timekeys
        out._lazy_ops = {}

        for key in self.keys():
            value = self.__dict__[key]
            if isinstance(value, h5py.Dataset):
                out.__dict__[key] = value
            else:
                out.__dict__[key] = value[mask].copy()

        if "mask" not in self._lazy_ops:
            out._lazy_ops["mask"] = mask
        else:
            out._lazy_ops["mask"] = self._lazy_ops["mask"].copy()
            out._lazy_ops["mask"][out._lazy_ops["mask"]] = mask

        if "slice" in self._lazy_ops:
            out._lazy_ops["slice"] = self._lazy_ops["slice"]

        return out

    def _resolve_start_end_after_slice(self):
        start, end, origin_translation = self._lazy_ops["unresolved_slice"]

        # todo confirm sorted
        # assert self.is_sorted()

        # anything that starts before the end of the slicing window
        start_vec = self.__dict__["start"][:]
        end_vec = self.__dict__["end"][:]
        idx_l = np.searchsorted(end_vec, start, side="right")

        # anything that will end after the start of the slicing window
        idx_r = np.searchsorted(start_vec, end)

        del self._lazy_ops["unresolved_slice"]
        self._lazy_ops["slice"] = (idx_l, idx_r, start, origin_translation)
        self.__dict__["start"] = (
            self.__dict__["start"][idx_l:idx_r] - origin_translation
        )
        self.__dict__["end"] = self.__dict__["end"][idx_l:idx_r] - origin_translation

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`Interval` object that contains the data between the
        start and end times. An interval is included if it has any overlap with the
        slicing window.
        """
        out = self.__class__.__new__(self.__class__)
        out._unicode_keys = self._unicode_keys
        out._lazy_ops = {}
        out._timekeys = self._timekeys

        if isinstance(self.__dict__["start"], h5py.Dataset):
            assert "slice" not in self._lazy_ops, "slice already exists"
            origin_translation = start if reset_origin else 0.0
            if "unresolved_slice" not in self._lazy_ops:
                out._lazy_ops["unresolved_slice"] = (start, end, origin_translation)
            else:
                curr_start, _, curr_origin_translation = self._lazy_ops[
                    "unresolved_slice"
                ]
                out._lazy_ops["unresolved_slice"] = (
                    curr_origin_translation + start,
                    curr_origin_translation + end,
                    curr_origin_translation + origin_translation,
                )

            idx_l = idx_r = None
            # out.__dict__["start"] = self.__dict__["start"]
            # out.__dict__["end"] = self.__dict__["end"]

        else:
            if not self.is_sorted():
                self.sort()

            # anything that starts before the end of the slicing window
            idx_l = np.searchsorted(self.end, start, side="right")

            # anything that will end after the start of the slicing window
            idx_r = np.searchsorted(self.start, end)

            origin_translation = start if reset_origin else 0.0
            if "slice" not in self._lazy_ops:
                out._lazy_ops["slice"] = (idx_l, idx_r, start, origin_translation)
            else:
                out._lazy_ops["slice"] = (
                    self._lazy_ops["slice"][0] + idx_l,
                    self._lazy_ops["slice"][0] + idx_r,
                    start,
                    self._lazy_ops["slice"][3] + origin_translation,
                )

        for key in self.keys():
            value = self.__dict__[key]
            if isinstance(value, h5py.Dataset):
                out.__dict__[key] = value
            else:
                if idx_l is None:
                    raise NotImplementedError(
                        f"An attribute ({key}) was accessed, but timestamps failed "
                        "to load. This is an edge case that was not handled."
                    )
                out.__dict__[key] = value[idx_l:idx_r].copy()
                if reset_origin and key in self._timekeys:
                    out.__dict__[key] = out.__dict__[key] - start

        if "mask" in self._lazy_ops:
            if idx_l is None:
                raise NotImplementedError(
                    "A mask was somehow created without accessing any attribute in the "
                    "data. This has not been taken into account."
                )
            out._lazy_ops["mask"] = self._lazy_ops["mask"][idx_l:idx_r]
        return out

    def to_hdf5(self, file):
        raise NotImplementedError("Cannot save a lazy interval object to hdf5.")

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

            import h5py
            from temporaldata import ArrayDict

            with h5py
        """
        # todo improve error message
        assert file.attrs["object"] == Interval.__name__, "object type mismatch"

        obj = cls.__new__(cls)
        for key, value in file.items():
            obj.__dict__[key] = value

        obj._unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()
        obj._timekeys = file.attrs["timekeys"].astype(str).tolist()
        obj._sorted = True
        obj._lazy_ops = {}

        return obj


def sorted_traversal(lintervals, rintervals):
    # we use an index to iterate over the intervals from both left and right objects
    lidx, ridx = 0, 0
    # to track whether we are looking at start or end, we use a binary flag that
    # denotes whether the current pointer is an "opening paranthesis" (lop=True)
    # or a "closing paranthesis" (lop=False)
    lop, rop = True, True

    len_lintervals = len(lintervals)
    len_rintervals = len(rintervals)

    while (lidx < len_lintervals) or (ridx < len_rintervals):
        # retrieve the time of the pointer in the left object
        if lidx < len_lintervals:
            # retrieve the time of the next interval in left object
            ltime = lintervals.start[lidx] if lop else lintervals.end[lidx]
        else:
            # exhausted all intervals in left object
            ltime = np.inf

        # retrieve the time of the pointer in the right object
        if ridx < len_rintervals:
            # retrieve the time of the next interval in right object
            rtime = rintervals.start[ridx] if rop else rintervals.end[ridx]
        else:
            # exhausted all intervals in right object
            rtime = np.inf

        # figure out which is the next pointer to process
        if ltime < rtime:
            # the next timestamps to consider is from the left object
            ptime = ltime  # time of the current pointer
            pop = lop  # True if pointer is opening
            pl = True  # True if pointer is from left object

            # move the left pointer accordingly
            if lop:
                # we only considered the start time, we now need to consider the
                # end before moving to the next interval
                lop = False
            else:
                # move to the next interval
                lop = True
                lidx += 1
        elif rtime < ltime:
            # the next timestamps to consider is from the right object
            ptime = rtime
            pop = rop
            pl = False
            if rop:
                rop = False
            else:
                rop = True
                ridx += 1
        else:  # ltime == rtime
            # When times are equal, prioritize closings over openings for union operations
            if not lop and rop:  # left is closing, right is opening
                ptime = ltime
                pop = lop  # False (closing)
                pl = True
                lop = True
                lidx += 1
            elif lop and not rop:  # left is opening, right is closing
                ptime = rtime
                pop = rop  # False (closing)
                pl = False
                rop = True
                ridx += 1
            elif lop and rop:  # both are openings
                # Process left opening first (arbitrary but consistent)
                ptime = ltime
                pop = lop
                pl = True
                lop = False
            else:  # both are closings
                # Process left closing first (arbitrary but consistent)
                ptime = ltime
                pop = lop
                pl = True
                lop = True
                lidx += 1
        yield ptime, pop, pl
