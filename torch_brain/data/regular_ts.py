from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import h5py
import numpy as np

from .arraydict import ArrayDict
from .interval import Interval
from .irregular_ts import IrregularTimeSeries
from .typing import ArrayLike

_NP_DTYPE_KINDS = {"b", "i", "u", "f", "c", "m", "M", "O", "S", "U", "V"}
# ^ From https://numpy.org/doc/2.2/reference/generated/numpy.dtype.kind.html

_DEFAULT_GAP_VALUE = {
    "b": False,  # boolean
    "i": -1,  # signed integers
    "u": 0,  # unsigned integers
    "f": np.nan,  # floating
}


def _validate_gap_value_dict(gap_value):
    for k, v in gap_value.items():
        if k not in _NP_DTYPE_KINDS:
            raise ValueError(
                f"gap_value dict has unsupported key {k!r}; valid keys "
                f"are {sorted(_NP_DTYPE_KINDS)} "
            )
        # bool is a subclass of int in Python, so check it explicitly first.
        is_bool = isinstance(v, (bool, np.bool_))
        is_int = isinstance(v, (int, np.integer)) and not is_bool
        is_float = isinstance(v, (float, np.floating))
        if k == "b" and not is_bool:
            raise ValueError(f"gap_value['b'] must be a bool, got {v!r}")
        if k == "i" and not is_int:
            raise ValueError(f"gap_value['i'] must be an integer, got {v!r}")
        if k == "u":
            if not is_int:
                raise ValueError(f"gap_value['u'] must be an integer, got {v!r}")
            if v < 0:
                raise ValueError(f"gap_value['u'] must be non-negative, got {v}")
        if k == "f" and not (is_int or is_float):
            raise ValueError(f"gap_value['f'] must be a number, got {v!r}")


def _validate_gap_value_matches_array_dtype(v, array: np.ndarray, name: str):
    """Validate that `v` is legal to be used with all input array dtypes

    Logic: cast gap value into target dtype. If:
        1. cast changes the value, we raise
        2. the cast emits a warning, we raise
    """

    src = np.array(v)

    # doing the cast here:
    # Numpy sometimes emits RuntmeWarning when doing a risky cast
    # and we want to catch that
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        try:
            dst = src.astype(array.dtype)
        except RuntimeWarning as _:
            raise ValueError(
                f"gap_value={v} cannot be losslessly stored in {name!r}; "
                f"cannot cast {src.dtype!r} into {array.dtype!r}"
            )

    if not np.array_equal(src, dst, equal_nan=True):
        raise ValueError(
            f"gap_value={v} cannot be losslessly stored in {name!r}; "
            f"numpy would silently cast it from {src.item()!r} to {dst.item()!r}"
        )


class RegularTimeSeries(ArrayDict):
    r"""A regular time series is the same as an irregular time series, but it has a
    regular sampling rate. This allows for faster indexing, possibility of patching data
    and meaningful Fourier operations. The first dimension of all attributes must be
    the time dimension.

    .. note::

        If you have a matrix of shape :math:`(N, T)`, where :math:`N` is the number of
        channels and :math:`T` is the number of time points, you should transpose it to
        :math:`(T, N)` before passing it to the constructor, since the first dimension
        should always be time.

    Args:
        sampling_rate: Sampling rate in Hz.
        domain_start: Absolute starting time offset (in seconds) of this signal. Defaults to :obj:`0.0`.
        **kwargs: Arbitrary keyword arguments where the values are arbitrary
            multi-dimensional (2d, 3d, ..., nd) arrays with shape (N, \*).

    See Also:
        :meth:`from_gappy_timeseries` to construct from regular timeseries that has
        gaps or missing values.

    Example ::

        >>> import numpy as np
        >>> from torch_brain.data import RegularTimeSeries

        >>> lfp = RegularTimeSeries(
        ...     raw=np.zeros((1000, 128)),
        ...     sampling_rate=250.,
        ... )

        >>> lfp.slice(0, 1)
        RegularTimeSeries(
          raw=[250, 128]
        )

        >>> lfp.to_irregular()
        IrregularTimeSeries(
          timestamps=[1000],
          raw=[1000, 128]
        )
    """

    _domain: Interval

    def __init__(
        self,
        *,
        sampling_rate: float,  # in Hz
        domain_start: float = 0.0,
        **kwargs: ArrayLike,
    ):
        if "domain" in kwargs:
            domain = kwargs.pop("domain")
            if domain == "auto":
                warnings.warn(
                    "The `domain` argument of `RegularTimeSeries` is deprecated "
                    "and will be removed in a future version. The domain is "
                    "always computed automatically as "
                    "[domain_start, domain_start + len(self) / sampling_rate); "
                    'you can drop `domain="auto"` from your call.',
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "Manually setting the domain of `RegularTimeSeries` to a "
                    "custom `Interval` is no longer supported; the domain is "
                    "always computed automatically as "
                    "[domain_start, domain_start + len(self) / sampling_rate) "
                    "so that its boundaries stay aligned to the sample grid. "
                    "Use `domain_start` to set the start time."
                )

        super().__init__(**kwargs)

        self._sampling_rate = sampling_rate

        if not isinstance(domain_start, (int, float)):
            raise ValueError(
                f"domain_start must be a number, got {type(domain_start)}."
            )

        self._domain = Interval(
            start=domain_start,
            end=domain_start + len(self) / sampling_rate,
        )

    @property
    def sampling_rate(self) -> float:
        r"""Sampling rate in Hz"""
        return self._sampling_rate

    @property
    def timestamps(self) -> np.ndarray:
        r"""Sample timestamps"""
        return (
            self.domain.start[0]
            + np.arange(len(self), dtype=np.float64) / self.sampling_rate
        )

    @property
    def domain(self) -> Interval:
        r"""Domain of this time series"""
        return self._domain

    def index_mask(self) -> np.ndarray:
        r"""Boolean mask marking which samples fall inside :attr:`domain`.

        For a gappy :obj:`RegularTimeSeries` (one whose :attr:`domain` consists
        of more than one interval), some positions along the time axis are
        fill values rather than real observations. This method returns a
        1-D boolean array of length ``len(self)`` where ``True`` marks a real
        sample and ``False`` marks a gap (fill).

        For a contiguous :obj:`RegularTimeSeries` (single-interval domain) the
        result is all ``True``.

        Returns:
            np.ndarray: 1-D boolean array of shape ``(len(self),)``.

        Example ::

            >>> import numpy as np
            >>> from torch_brain.data import RegularTimeSeries

            >>> # Contiguous (non-gappy) series: every sample is real.
            >>> rts = RegularTimeSeries(
            ...     raw=np.arange(4), sampling_rate=100.0,
            ... )
            >>> rts.index_mask()
            array([ True,  True,  True,  True])

            >>> # Gappy series: 0.02s and 0.05s samples are missing.
            >>> ts = [0.0, 0.01, 0.03, 0.04, 0.06]
            >>> raw = [1, 2, 3, 4, 5]
            >>> rts = RegularTimeSeries.from_gappy_timeseries(
            ...     ts, sampling_rate=100.0, raw=raw,
            ... )
            >>> rts.index_mask()
            array([ True,  True, False,  True,  True, False,  True])
            >>> rts.raw  # contains fill values
            array([ 1,  2, -1,  3,  4, -1,  5])
            >>> rts.raw[rts.index_mask()]
            array([1, 2, 3, 4, 5])
        """
        n = len(self)
        domain = self.domain

        if len(domain) == 1:
            return np.full(n, True, dtype=bool)

        sampling_rate = self.sampling_rate
        start_ts, end_ts = domain.start, domain.end
        start_id = np.round((start_ts - start_ts[0]) * sampling_rate).astype(int)
        end_id = np.round((end_ts - start_ts[0]) * sampling_rate).astype(int)

        if end_id[-1] != n:
            raise RuntimeError(  # pragma: no cover
                f"This should never happen. Debug info:\n"
                f"{n=}\n"
                f"{start_id=}\n"
                f"{end_id=}\n"
            )

        # Create an array that marks start of a True run by +1
        # and start of a False run by -1
        diff = np.zeros(n + 1, dtype=np.int8)
        diff[start_id] = 1
        diff[end_id] = -1
        # Cumsum would convert it to runs of ones and zeros corresponding
        # to valid and invalid timestamps
        ans = diff.cumsum()[:n].astype(bool)
        # Why this way? to avoid python for-loops; numpy vector ops should be faster

        return ans

    def select_by_mask(self, mask: np.ndarray):
        """Raises a NotImplementedError as this method is not supported
        for :obj:`RegularTimeSeries`.

        Raises:
            NotImplementedError: Always, because this method cannot
                be implemented for this class.
        """
        # TODO: Implement once we support "gappy" regular timeseries
        raise NotImplementedError("Not implemented for RegularTimeSeries.")

    def _time_to_idx(
        self,
        time: float,
        eps: float = 1e-9,
    ) -> tuple[int, float]:
        """Converts a timestamp to a sample index and its exact reconstructed time.
        Args:
            time: The timestamp to convert.
            eps: Tolerance for floating-point precision. If the calculated index
                is within ``eps`` of an integer, it is snapped to that integer.
                This prevents tiny precision errors (e.g., 3.999999999999999) from
                causing off-by-one errors when applying ``math.ceil``.
        Returns:
            tuple[int, float]: A tuple containing:
                * **index**: The calculated integer sample index within the array.
                * **reconstructed_time**: The exact timestamp in seconds that corresponds
                  to the selected **index** (i.e. the actual time of the sample).
        """
        domain_start = self.domain.start[0]
        domain_end = self.domain.end[-1]

        # Clamp to domain bounds
        if time <= domain_start:
            return 0, domain_start

        if time > domain_end:
            return len(self), domain_end

        # Calculate relative index
        rel_t = time - domain_start
        idx_float = rel_t * self.sampling_rate

        # Precision check: if it's "close enough" to an integer, treat it as that integer
        rounded = round(idx_float)
        if abs(idx_float - rounded) < eps:
            idx_float = float(rounded)

        # Determine index and reconstruct the actual timestamp of that sample
        idx = math.ceil(idx_float)

        actual_time = domain_start + (idx / self.sampling_rate)

        return idx, actual_time

    def slice(
        self,
        start: float,
        end: float,
        reset_origin: bool = True,
        eps: float = 1e-9,
    ):
        r"""Returns a new :obj:`RegularTimeSeries` object that contains the data between
        the start (inclusive) and end (exclusive) times (i.e., [start, end)).

        :obj:`start` and :obj:`end` are snapped up to the next grid point (the next
        multiple of ``1/sampling_rate``).

        - Gap-filled samples at the start or end of the result are trimmed, so
          returned data always begins and ends on real samples.
        - Gaps in the middle of the window are preserved as-is and remain filled
          with the gap value.
        - Slices that fall fully outside the domain or entirely within a gap
          return empty data.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
            eps: A tiny 'rounding buffer' to handle floating-point noise when computing indices.
                If your sampling rate is very high, you may need to increase
                this (e.g., to 1e-7) to avoid off-by-one errors.

        Returns:
            RegularTimeSeries: A new instance of the same class
            containing a subset of the data. The new object will have a modified
            :obj:`Interval` domain reflecting the actual sampled boundaries.
        """
        start_id, out_start = self._time_to_idx(start, eps=eps)
        end_id, out_end = self._time_to_idx(end, eps=eps)

        # Intersect with the (possibly multi-interval) domain
        new_domain = self.domain & Interval(out_start, out_end)

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate

        # No real samples
        is_empty = len(new_domain) == 0 or new_domain.start[0] == new_domain.end[-1]
        if is_empty:
            out._domain = (
                Interval(start=0.0, end=0.0)
                if reset_origin
                else Interval(start=out_start, end=out_start)
            )
            for key in self.keys():
                out.__dict__[key] = self.__dict__[key][0:0].copy()
            return out

        # Trim leading/trailing gap samples, Internal gaps stay in the array as gap-filled values.
        leading_trim = int(
            round((new_domain.start[0] - out_start) * self.sampling_rate)
        )
        trailing_trim = int(round((out_end - new_domain.end[-1]) * self.sampling_rate))
        start_id += leading_trim
        end_id -= trailing_trim

        if reset_origin:
            new_domain.start = new_domain.start - start
            new_domain.end = new_domain.end - start

        out._domain = new_domain

        for key in self.keys():
            out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        return out

    def to_irregular(self):
        r"""Converts the :obj:`RegularTimeSeries` object to an :obj:`IrregularTimeSeries` object.

        Gap-fill samples (where :meth:`index_mask` is :obj:`False`) are dropped.

        The returned arrays (timestamps, values, and domain) are independent
        copies; mutating them will not affect this :obj:`RegularTimeSeries`.

        Returns:
            :obj:`IrregularTimeSeries` with timestamps and all attributes copied.

        Example ::

            >>> import numpy as np
            >>> from torch_brain.data import RegularTimeSeries

            >>> # Contiguous (non-gappy) series: every sample is kept.
            >>> rts = RegularTimeSeries(raw=np.arange(4), sampling_rate=10.0)
            >>> irts = rts.to_irregular()
            >>> irts.timestamps
            array([0. , 0.1, 0.2, 0.3])
            >>> irts.raw
            array([0, 1, 2, 3])

            >>> # Gappy series: gap-fill samples are dropped.
            >>> ts = [0.0, 0.01, 0.03, 0.04, 0.06]
            >>> raw = [1, 2, 3, 4, 5]
            >>> rts = RegularTimeSeries.from_gappy_timeseries(
            ...     ts, sampling_rate=100.0, raw=raw,
            ... )
            >>> rts.raw  # contains fill values
            array([ 1,  2, -1,  3,  4, -1,  5])
            >>> irts = rts.to_irregular()
            >>> irts.timestamps
            array([0.  , 0.01, 0.03, 0.04, 0.06])
            >>> irts.raw
            array([1, 2, 3, 4, 5])
        """
        if not self.is_gappy():
            # Every sample is real, skip the mask.
            return IrregularTimeSeries(
                timestamps=self.timestamps,
                **{k: getattr(self, k).copy() for k in self.keys()},
                domain=copy.deepcopy(self.domain),
            )

        mask = self.index_mask()
        return IrregularTimeSeries(
            timestamps=self.timestamps[mask],
            **{k: getattr(self, k)[mask] for k in self.keys()},
            domain=copy.deepcopy(self.domain),
        )

    def to_hdf5(self, file):
        r"""Saves the data object to an HDF5 file.

        Args:
            file: HDF5 file.

        .. code-block:: python

                import h5py
                from torch_brain.data import RegularTimeSeries

                data = RegularTimeSeries(
                    raw=np.zeros((1000, 128)),
                    sampling_rate=250.,
                )

                with h5py.File("data.h5", "w") as f:
                    data.to_hdf5(f)
        """
        for key in self.keys():
            value = getattr(self, key)
            file.create_dataset(key, data=value)

        # domain is of type Interval
        grp = file.create_group("domain")
        self._domain.to_hdf5(grp)

        file.attrs["object"] = self.__class__.__name__
        file.attrs["sampling_rate"] = self.sampling_rate

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file: HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyRegularTimeSeries.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from torch_brain.data import RegularTimeSeries

            with h5py.File("data.h5", "r") as f:
                data = RegularTimeSeries.from_hdf5(f)
        """
        assert file.attrs["object"] == cls.__name__, "object type mismatch"

        data = {}
        for key, value in file.items():
            if key != "domain":
                data[key] = value[:]

        domain = Interval.from_hdf5(file["domain"])
        obj = cls(
            **data,
            sampling_rate=file.attrs["sampling_rate"],
            domain_start=float(domain.start[0]),
        )
        obj._domain = domain

        return obj

    @classmethod
    def from_gappy_timeseries(
        cls,
        timestamps: ArrayLike,
        sampling_rate: float,
        gap_value: Any | dict[str, Any] | None = None,
        rtol: float = 1e-3,
        **kwargs: ArrayLike,
    ) -> RegularTimeSeries:
        r"""Regularize an approximately-regular but gappy timeseries.

        Construct a :obj:`RegularTimeSeries` from approximately-regular but
        gappy timestamps and value arrays by snapping each sample to a regular
        grid at :obj:`sampling_rate` and filling missing samples with
        :obj:`gap_value`.

        Useful for signals that are nominally regular (e.g. behavioral streams
        at a fixed sampling rate) but contain missing samples, which would
        otherwise have to be carried as an :obj:`IrregularTimeSeries` and would
        suffer numerical-precision issues during slicing.

        Args:
            timestamps: 1-D array-like of timestamps, strictly increasing.
                Each entry must lie within :obj:`rtol` samples of a regular
                grid at :obj:`sampling_rate`, anchored at
                :obj:`timestamps[0]`.
            sampling_rate: Sampling rate in Hz.
            gap_value: Value used to fill missing samples. May be:

                * :obj:`None` (default) — uses per-kind defaults: ``-1`` for
                  signed integers, ``0`` for unsigned integers,
                  :obj:`numpy.nan` for floats, ``False`` for bools.
                * A scalar (``int``, ``float``, or ``bool``) — used for every
                  kwarg array regardless of dtype.
                * A ``dict`` mapping :obj:`numpy.dtype.kind` codes to fill
                  values. Recognized kinds: ``'b'`` (bool), ``'i'`` (signed
                  int), ``'u'`` (unsigned int), ``'f'`` (float). Example:
                  ``{'i': -1, 'u': 0, 'f': np.nan}``. Raises :obj:`KeyError`
                  if a kwarg's dtype kind is not in the dict.
            rtol: Maximum allowed deviation, in samples, of any input timestamp
                from the regular grid.
            **kwargs: Named array-like values whose first dimension equals
                ``len(timestamps)``.

        Returns:
            RegularTimeSeries: A regular time series with the same named
            arrays, gaps filled with :obj:`gap_value`.

        Raises:
            ValueError: If timestamps deviate from the regular grid by more than :obj:`rtol`

        See Also:
            * :meth:`is_gappy` to check whether a series has gaps.
            * :meth:`index_mask` for a boolean mask of real vs. gap-fill samples.

        Example ::

            >>> import numpy as np
            >>> from torch_brain.data import RegularTimeSeries

            >>> # 4 samples at 100 Hz, the 0.02s sample is missing.
            >>> rts = RegularTimeSeries.from_gappy_timeseries(
            ...     ts=[0.0, 0.01, 0.03, 0.04],
            ...     sampling_rate=100.0,
            ...     raw=[1.0, 2.0, 3.0, 4.0],
            ... )
            >>> rts.raw
            array([ 1.,  2., nan,  3.,  4.])
            >>> rts.domain.start
            array([0.  , 0.03])
            >>> rts.domain.end
            array([0.02, 0.05])
            >>> rts.index_mask()  # indicates valid and filled-in timestamps
            array([ True,  True, False,  True,  True])
        """
        timestamps = np.asarray(timestamps)
        if timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1-D, got shape {timestamps.shape}")
        if len(timestamps) < 2:
            raise ValueError(
                f"timestamps must have at least 2 entries, got {len(timestamps)}"
            )
        if not (np.diff(timestamps) > 0).all():
            raise ValueError("timestamps must be strictly increasing")

        if gap_value is None:
            gap_value = _DEFAULT_GAP_VALUE

        if isinstance(gap_value, dict):
            _validate_gap_value_dict(gap_value)

        start_time = float(timestamps[0])
        rel_idx = (timestamps - start_time) * sampling_rate
        grid_idx = np.round(rel_idx).astype(np.int64)

        max_dev = float(np.max(np.abs(rel_idx - grid_idx)))
        if max_dev > rtol:
            raise ValueError(
                f"timestamps deviate from a regular grid at sampling_rate="
                f"{sampling_rate} Hz by up to {max_dev:.3g} samples, "
                f"exceeding rtol={rtol}. Pick a different sampling_rate, "
                f"increase rtol, or use IrregularTimeSeries if this signal "
                f"is inherently irregular."
            )

        idx_diffs = np.diff(grid_idx)
        min_idx_gap = int(idx_diffs.min())
        if min_idx_gap < 1:
            raise ValueError(
                f"timestamps contain duplicate or sub-sample-spaced entries "
                f"at sampling_rate={sampling_rate} Hz"
            )
        if min_idx_gap > 1:
            raise ValueError(
                f"sampling_rate={sampling_rate} appears too high: the smallest "
                f"gap between consecutive timestamps is {min_idx_gap} grid "
                f"steps (expected 1). The true sampling rate may be closer to "
                f"{sampling_rate / min_idx_gap}."
            )

        num_timesteps = int(grid_idx[-1]) + 1

        # Build a multi-interval domain that excludes gaps
        gap_after = idx_diffs > 1
        is_run_start = np.concatenate([[True], gap_after])
        is_run_end = np.concatenate([gap_after, [True]])
        domain = Interval(
            start=start_time + grid_idx[is_run_start] / sampling_rate,
            end=start_time + (grid_idx[is_run_end] + 1) / sampling_rate,
        )

        filled: dict[str, np.ndarray] = {}
        for key, arr in kwargs.items():
            arr = np.asarray(arr)
            if len(arr) != len(timestamps):
                raise ValueError(
                    f"{key!r} has length {len(arr)}, expected "
                    f"{len(timestamps)} to match timestamps"
                )

            if isinstance(gap_value, dict):
                kind = arr.dtype.kind
                if kind not in gap_value:
                    raise KeyError(
                        f"{key!r} has dtype {arr.dtype} (kind {kind!r}) which is "
                        f"not in gap_value dict (keys: {list(gap_value)})"
                    )
                _gap_value = gap_value[kind]
            else:
                _gap_value = gap_value

            _validate_gap_value_matches_array_dtype(_gap_value, array=arr, name=key)

            out = np.full((num_timesteps, *arr.shape[1:]), _gap_value, dtype=arr.dtype)
            out[grid_idx] = arr
            filled[key] = out

        obj = cls(sampling_rate=sampling_rate, domain_start=start_time, **filled)
        obj._domain = domain  # replace single-interval auto domain with gappy one
        return obj

    def is_gappy(self) -> bool:
        r"""Returns :obj:`True` if this :obj:`RegularTimeSeries` has gaps.

        A series is *gappy* when its :attr:`domain` is made up of more than one
        interval; positions inside the gaps are filled with the configured
        gap value (see :meth:`from_gappy_timeseries`). A contiguous series
        (single-interval domain) returns :obj:`False`.

        Returns:
            bool: :obj:`True` if the domain has more than one interval.

        See Also:
            :meth:`index_mask` for a boolean mask of real vs. gap-fill samples.

        Example ::

            >>> import numpy as np
            >>> from torch_brain.data import RegularTimeSeries

            >>> rts = RegularTimeSeries(raw=np.arange(4), sampling_rate=100.0)
            >>> rts.is_gappy()
            False

            >>> rts = RegularTimeSeries.from_gappy_timeseries(
            ...     [0.0, 0.01, 0.03], sampling_rate=100.0, raw=[1, 2, 3],
            ... )
            >>> rts.is_gappy()
            True
        """
        return len(self.domain) > 1


class LazyRegularTimeSeries(RegularTimeSeries):
    r"""Lazy variant of :obj:`RegularTimeSeries`. The data is not loaded until it is
    accessed. This class is meant to be used when the data is too large to fit in
    memory, and is intended to be intantiated via.
    :obj:`LazyRegularTimeSeries.from_hdf5`.

    .. note:: To access an attribute without triggering the in-memory loading use
        self.__dict__[key] otherwise using self.key or getattr(self, key) will trigger
        the lazy loading and will automatically convert the h5py dataset to a numpy
        array as well as apply any outstanding masks.
    """

    _lazy_ops: dict

    def __init__(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot be constructed directly; use from_hdf5."
        )

    def _maybe_first_dim(self):
        if len(self.keys()) == 0:
            return None
        else:
            # todo check _lazy_ops
            for key in self.keys():
                value = self.__dict__[key]
                if isinstance(value, np.ndarray):
                    return value.shape[0]

            if "slice" in self._lazy_ops:
                # TODO add more constraints to the domain in RegularTimeSeries

                # TODO it is always better to resolve another attribute before timestamps
                # this is because we are dealing with numerical noise
                # we know the domain and the sampling rate, we can infer the number of pts
                domain_length = self.domain.end[-1] - self.domain.start[0]
                return int(np.round(domain_length * self.sampling_rate))

            # otherwise nothing was loaded, return the first dim of the h5py dataset
            return self.__dict__[self.keys()[0]].shape[0]

    def __getattribute__(self, name):
        if not name in ["__dict__", "keys"]:
            # intercept attribute calls
            if name in self.keys():
                out = self.__dict__[name]

                if isinstance(out, h5py.Dataset):
                    # convert into numpy array
                    if "slice" in self._lazy_ops:
                        idx_l, idx_r = self._lazy_ops["slice"]
                        out = out[idx_l:idx_r]
                    else:
                        out = out[:]

                    # store it
                    self.__dict__[name] = out

                # If all attributes are loaded, we can remove the lazy flag
                all_loaded = all(
                    isinstance(self.__dict__[key], np.ndarray) for key in self.keys()
                )
                if all_loaded:
                    self.__class__ = RegularTimeSeries
                    del self._lazy_ops

                return out
        return super().__getattribute__(name)

    def slice(
        self,
        start: float,
        end: float,
        reset_origin: bool = True,
        eps: float = 1e-9,
    ):
        r"""Returns a new :obj:`RegularTimeSeries` object that contains the data between
        the start (inclusive) and end (exclusive) times (i.e., [start, end)).

        :obj:`start` and :obj:`end` are snapped up to the next grid point (the next
        multiple of ``1/sampling_rate``).

        - Gap-filled samples at the start or end of the result are trimmed, so
          returned data always begins and ends on real samples.
        - Gaps in the middle of the window are preserved as-is and remain filled
          with the gap value.
        - Slices that fall fully outside the domain or entirely within a gap
          return empty data.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
            eps: A tiny 'rounding buffer' to handle floating-point noise when computing indices.
                If your sampling rate is very high, you may need to increase
                this (e.g., to 1e-7) to avoid off-by-one errors.
        Returns:
            LazyRegularTimeSeries: A new instance of the same class
            containing a subset of the data. The new object will have a modified
            :obj:`Interval` domain reflecting the actual sampled boundaries.
        """
        start_id, out_start = self._time_to_idx(start, eps=eps)
        end_id, out_end = self._time_to_idx(end, eps=eps)

        # Intersect with the (possibly multi-interval) domain
        new_domain = self.domain & Interval(out_start, out_end)

        is_empty = len(new_domain) == 0 or new_domain.start[0] == new_domain.end[-1]
        if is_empty:
            # No data to defer-load; return an eager RegularTimeSeries.
            out = RegularTimeSeries.__new__(RegularTimeSeries)
            out._sampling_rate = self.sampling_rate
            out._domain = (
                Interval(start=0.0, end=0.0)
                if reset_origin
                else Interval(start=out_start, end=out_start)
            )
            for key in self.keys():
                out.__dict__[key] = self.__dict__[key][0:0]
            return out

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate
        out._lazy_ops = {}

        parent_offset = self._lazy_ops["slice"][0] if "slice" in self._lazy_ops else 0

        # Trim leading/trailing gap samples
        leading_trim = int(
            round((new_domain.start[0] - out_start) * self.sampling_rate)
        )
        trailing_trim = int(round((out_end - new_domain.end[-1]) * self.sampling_rate))
        start_id += leading_trim
        end_id -= trailing_trim

        if reset_origin:
            new_domain.start = new_domain.start - start
            new_domain.end = new_domain.end - start

        out._domain = new_domain

        for key in self.keys():
            if isinstance(self.__dict__[key], h5py.Dataset):
                out.__dict__[key] = self.__dict__[key]
            else:
                out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        out._lazy_ops["slice"] = (
            parent_offset + start_id,
            parent_offset + end_id,
        )

        return out

    def to_hdf5(self, file):
        raise NotImplementedError("Cannot save a lazy array dict to hdf5.")

    @classmethod
    def from_gappy_timeseries(cls, *_args, **_kwargs):
        r"""Not implemented for :obj:`LazyRegularTimeSeries`.

        Use :meth:`RegularTimeSeries.from_gappy_timeseries` instead.
        """
        raise NotImplementedError(
            "from_gappy_timeseries is not available on LazyRegularTimeSeries; "
            "use RegularTimeSeries.from_gappy_timeseries instead."
        )

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file: HDF5 file.

        .. code-block:: python

            import h5py
            from torch_brain.data import ArrayDict

            with h5py.File("data.h5", "r") as f:
                data = ArrayDict.from_hdf5(f)
        """
        assert (
            file.attrs["object"] == RegularTimeSeries.__name__
        ), "object type mismatch"

        obj = cls.__new__(cls)
        for key, value in file.items():
            if key == "domain":
                obj.__dict__["_domain"] = Interval.from_hdf5(file[key])
            else:
                obj.__dict__[key] = value
        obj._lazy_ops = {}
        obj._sampling_rate = file.attrs["sampling_rate"]

        return obj
