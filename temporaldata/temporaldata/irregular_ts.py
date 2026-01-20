from __future__ import annotations

from typing import Dict, List, Union
import logging

import h5py
import numpy as np
import pandas as pd

from .arraydict import ArrayDict
from .interval import Interval


class IrregularTimeSeries(ArrayDict):
    r"""An irregular time series is defined by a set of timestamps and a set of
    attributes that must share the same first dimension as the timestamps.
    This data object is ideal for event-based data as well as irregularly sampled time
    series.

    Args:
        timestamps: an array of timestamps of shape (N,).
        timekeys: a list of strings that specify which attributes are time-based
            attributes, this ensures that these attributes are updated appropriately
            when slicing.
        domain: an :obj:`Interval` object that defines the domain over which the
            timeseries is defined. If set to :obj:`"auto"`, the domain will be
            automatically the interval defined by the minimum and maximum timestamps.
        **kwargs: arrays that shares the same first dimension N.

    Example ::

        >>> import numpy as np
        >>> from temporaldata import IrregularTimeSeries

        >>> spikes = IrregularTimeSeries(
        ...     unit_index=np.array([0, 0, 1, 0, 1, 2]),
        ...     timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        ...     waveforms=np.zeros((6, 48)),
        ...     domain="auto",
        ... )

        >>> spikes
        IrregularTimeSeries(
          timestamps=[6],
          unit_index=[6],
          waveforms=[6, 48]
        )

        >>> spikes.domain.start, spikes.domain.end
        (array([0.1]), array([0.6]))

        >>> spikes.keys()
        ['timestamps', 'unit_index', 'waveforms']

        >>> spikes.is_sorted()
        True

        >>> slice_of_spikes = spikes.slice(0.2, 0.5)
        >>> slice_of_spikes
        IrregularTimeSeries(
          timestamps=[3],
          unit_index=[3],
          waveforms=[3, 48]
        )

        >>> slice_of_spikes.domain.start, slice_of_spikes.domain.end
        (array([0.]), array([0.3]))

        >>> slice_of_spikes.timestamps
        array([0. , 0.1, 0.2])
    """

    _sorted = None
    _timekeys = None
    _domain = None

    def __init__(
        self,
        timestamps: np.ndarray,
        *,
        timekeys: List[str] = None,
        domain: Union[Interval, str],
        **kwargs: Dict[str, np.ndarray],
    ):
        super().__init__(timestamps=timestamps, **kwargs)

        # timekeys
        if timekeys is None:
            timekeys = []
        if "timestamps" not in timekeys:
            timekeys.append("timestamps")

        for key in timekeys:
            assert key in self.keys(), f"Time attribute {key} does not exist."

        self._timekeys = timekeys

        # domain
        if domain == "auto":
            domain = Interval(
                start=self._maybe_start(),
                end=self._maybe_end(),
            )
        else:
            if not isinstance(domain, Interval):
                raise ValueError(
                    f"domain must be an Interval object or 'auto', got {type(domain)}."
                )

            if not domain.is_disjoint():
                raise ValueError("The domain intervals must not be overlapping.")

            if not domain.is_sorted():
                domain.sort()

        self._domain = domain

    # todo add setter for domain
    @property
    def domain(self):
        r"""The time domain over which the time series is defined. Usually a single
        interval, but could also be a set of intervals."""
        return self._domain

    @domain.setter
    def domain(self, value: Interval):
        if not isinstance(value, Interval):
            raise ValueError(f"domain must be an Interval object, got {type(value)}.")
        self._domain = value

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
        super(IrregularTimeSeries, self).__setattr__(name, value)

        if name == "timestamps":
            assert value.ndim == 1, "timestamps must be 1D."
            assert ~np.isnan(value).any(), f"timestamps cannot contain NaNs."
            if value.dtype != np.float64:
                logging.warning(f"{name} is of type {value.dtype} not of type float64.")
            # timestamps has been updated, we no longer know whether it is sorted or not
            self._sorted = None

    def is_sorted(self):
        r"""Returns :obj:`True` if the timestamps are sorted."""
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None:
            self._sorted = bool((self.timestamps[1:] >= self.timestamps[:-1]).all())
        return self._sorted

    def _maybe_start(self) -> float:
        r"""Returns the start time of the time series. If the time series is not sorted,
        the start time is the minimum timestamp."""
        if self.is_sorted():
            return np.float64(self.timestamps[0])
        else:
            return np.float64(self.timestamps.min())

    def _maybe_end(self) -> float:
        r"""Returns the end time of the time series. If the time series is not sorted,
        the end time is the maximum timestamp."""
        if self.is_sorted():
            return np.float64(self.timestamps[-1])
        else:
            return np.float64(self.timestamps.max())

    def sort(self):
        r"""Sorts the timestamps, and reorders the other attributes accordingly.
        This method is applied in place."""
        if not self.is_sorted():
            sorted_indices = np.argsort(self.timestamps)
            for key in self.keys():
                self.__dict__[key] = self.__dict__[key][sorted_indices]
        self._sorted = True

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`IrregularTimeSeries` object that contains the data
        between the start and end times. The end time is exclusive, the slice will
        only include data in :math:`[\textrm{start}, \textrm{end})`.

        If :obj:`reset_origin` is :obj:`True`, all time attributes are updated to
        be relative to the new start time. The domain is also updated accordingly.

        .. warning::
            If the time series is not sorted, it will be automatically sorted in place.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
        """
        if not self.is_sorted():
            logging.warning("time series is not sorted, sorting before slicing")
            self.sort()

        idx_l = np.searchsorted(self.timestamps, start)
        idx_r = np.searchsorted(self.timestamps, end)

        out = self.__class__.__new__(self.__class__)

        # private attributes
        out._timekeys = self._timekeys
        out._sorted = True  # we know the sequence is sorted
        out._domain = self._domain & Interval(start=start, end=end)
        if reset_origin:
            out._domain.start = out._domain.start - start
            out._domain.end = out._domain.end - start

        # array attributes
        for key in self.keys():
            out.__dict__[key] = self.__dict__[key][idx_l:idx_r].copy()

        if reset_origin:
            for key in self._timekeys:
                out.__dict__[key] = out.__dict__[key] - start
        return out

    def select_by_mask(self, mask: np.ndarray):
        r"""Return a new :obj:`IrregularTimeSeries` object where all array attributes
        are indexed using the boolean mask.

        Note that this will not update the domain, as it is unclear how to resolve the
        domain when the mask is applied. If you wish to update the domain, you should
        do so manually.
        """
        out = super().select_by_mask(mask, timekeys=self._timekeys, domain=self.domain)
        out._sorted = self._sorted
        return out

    def select_by_interval(self, interval: Interval):
        r"""Return a new :obj:`IrregularTimeSeries` object where all timestamps are
        within the interval.

        Args:
            interval: Interval object.
        """
        idx_l = np.searchsorted(self.timestamps, interval.start)
        idx_r = np.searchsorted(self.timestamps, interval.end)

        mask = np.zeros(len(self), dtype=bool)
        for i in range(len(interval)):
            mask[idx_l[i] : idx_r[i]] = True

        out = self.select_by_mask(mask)
        out._domain = out._domain & interval
        return out

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        domain: Union[str, Interval] = "auto",
        unsigned_to_long: bool = True,
    ):
        r"""Create an :obj:`IrregularTimeseries` object from a pandas DataFrame.
        The dataframe must have a timestamps column, with the name :obj:`"timestamps"`
        (use `pd.Dataframe.rename` if needed).

        The columns in the DataFrame are converted to arrays when possible, otherwise
        they will be skipped.

        Args:
            df: DataFrame.
            unsigned_to_long: Whether to automatically convert unsigned
              integers to int64 dtype. Defaults to :obj:`True`.
            domain (optional): The domain over which the time
                series is defined. If set to :obj:`"auto"`, the domain will be
                automatically the interval defined by the minimum and maximum
                timestamps. Defaults to :obj:`"auto"`.
        """
        if "timestamps" not in df.columns:
            raise ValueError("Column 'timestamps' not found in dataframe.")

        return super().from_dataframe(
            df,
            unsigned_to_long=unsigned_to_long,
            domain=domain,
        )

    def to_hdf5(self, file):
        r"""Saves the data object to an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. warning::
            If the time series is not sorted, it will be automatically sorted in place.

        .. code-block:: python

            import h5py
            from temporaldata import IrregularTimeseries

            data = IrregularTimeseries(
                unit_index=np.array([0, 0, 1, 0, 1, 2]),
                timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                waveforms=np.zeros((6, 48)),
                domain="auto",
            )

            with h5py.File("data.h5", "w") as f:
                data.to_hdf5(f)
        """
        if not self.is_sorted():
            logging.warning("time series is not sorted, sorting before saving to h5")
            self.sort()

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

        # in case we want to do lazy loading, we need to store some map to the
        # irregularly sampled timestamps
        # we use a 1 second resolution
        grid_timestamps = np.arange(
            self.domain.start[0],
            self.domain.end[-1] + 1.0,
            1.0,
            dtype=np.float64,
        )
        file.create_dataset(
            "timestamp_indices_1s",
            data=np.searchsorted(self.timestamps, grid_timestamps),
        )

        # domain is of type Interval
        grp = file.create_group("domain")
        self.domain.to_hdf5(grp)

        # save other private attributes
        file.attrs["_unicode_keys"] = np.array(_unicode_keys, dtype="S")
        file.attrs["timekeys"] = np.array(self._timekeys, dtype="S")
        file.attrs["object"] = self.__class__.__name__

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyIrregularTimeSeries.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from temporaldata import IrregularTimeSeries

            with h5py.File("data.h5", "r") as f:
                data = IrregularTimeSeries.from_hdf5(f)
        """
        if file.attrs["object"] != cls.__name__:
            raise ValueError(
                f"File contains data for a {file.attrs['object']} object, expected "
                f"{cls.__name__} object."
            )

        _unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()

        data = {}
        for key, value in file.items():
            # skip timestamp_indidces_1s since we're not lazy loading here
            if key not in ["timestamp_indices_1s", "domain"]:
                data[key] = value[:]
                # if the values were originally unicode but stored as fixed length ASCII bytes
                if key in _unicode_keys:
                    data[key] = data[key].astype("U")

        timekeys = file.attrs["timekeys"].astype(str).tolist()
        domain = Interval.from_hdf5(file["domain"])

        obj = cls(**data, timekeys=timekeys, domain=domain)
        # only sorted data could be saved to hdf5, so we know it's sorted
        obj._sorted = True

        return obj


class LazyIrregularTimeSeries(IrregularTimeSeries):
    r"""Lazy variant of :obj:`IrregularTimeSeries`. The data is not loaded until it is
    accessed. This class is meant to be used when the data is too large to fit in
    memory, and is intended to be intantiated via.
    :obj:`LazyIrregularTimeSeries.from_hdf5`.

    .. note:: To access an attribute without triggering the in-memory loading use
        self.__dict__[key] otherwise using self.key or getattr(self, key) will trigger
        the lazy loading and will automatically convert the h5py dataset to a numpy
        array as well as apply any outstanding masks.
    """

    _lazy_ops = dict()
    _unicode_keys = []

    def _maybe_first_dim(self):
        if len(self.keys()) == 0:
            return None
        else:
            # if slice is waiting to be resolved, we need to resolve it now to get the
            # first dimension
            if "unresolved_slice" in self._lazy_ops:
                return self.timestamps.shape[0]

            # if slicing already took place, than some attribute would have already
            # been loaded. look for any numpy array
            for key in self.keys():
                value = self.__dict__[key]
                if isinstance(value, np.ndarray):
                    return value.shape[0]

            # no array was loaded, check if some lazy masking is planned
            if "mask" in self._lazy_ops:
                return self._lazy_ops["mask"].sum()

            # otherwise nothing was loaded, return the first dim of the h5py dataset
            return self.__dict__[self.keys()[0]].shape[0]

    def load(self):
        r"""Loads all the data from the HDF5 file into memory."""
        # simply access all attributes to trigger the lazy loading
        for key in self.keys():
            getattr(self, key)

    def __getattribute__(self, name):
        if not name in ["__dict__", "keys"]:
            # intercept attribute calls
            if name in self.keys():
                # out could either be a numpy array or a reference to a h5py dataset
                # if is not loaded, now is the time to load it and apply any outstanding
                # slicing or masking.
                out = self.__dict__[name]

                if isinstance(out, h5py.Dataset):
                    # convert into numpy array

                    # first we check if timestamps was resolved
                    if "unresolved_slice" in self._lazy_ops:
                        # slice and unresolved_slice cannot both be queued
                        assert "slice" not in self._lazy_ops
                        # slicing never happened, and we need to resolve timestamps
                        # to identify the time points that we need
                        self._resolve_timestamps_after_slice()
                        # after this "unresolved_slice" is replaced with "slice"

                    # timestamps are resolved and there is a "slice"
                    if "slice" in self._lazy_ops:
                        idx_l, idx_r, start, origin_translation = self._lazy_ops[
                            "slice"
                        ]
                        out = out[idx_l:idx_r]
                        if name in self._timekeys:
                            out = out - origin_translation

                    # there could have been masking, so apply it
                    if "mask" in self._lazy_ops:
                        out = out[self._lazy_ops["mask"]]

                    # no lazy operations found, just load the entire array
                    if len(self._lazy_ops) == 0:
                        out = out[:]

                    if name in self._unicode_keys:
                        # convert back to unicode
                        out = out.astype("U")

                    # store it in memory now that it is loaded
                    self.__dict__[name] = out

                # if all attributes are loaded, we can remove the lazy flag
                all_loaded = all(
                    isinstance(self.__dict__[key], np.ndarray) for key in self.keys()
                )
                if all_loaded:
                    # simply change classes
                    self.__class__ = IrregularTimeSeries
                    # delete unnecessary attributes
                    del self._lazy_ops, self._unicode_keys
                    if hasattr(self, "_timestamp_indices_1s"):
                        del self._timestamp_indices_1s

                return out
        return super(LazyIrregularTimeSeries, self).__getattribute__(name)

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
        out._domain = self._domain
        out._lazy_ops = {}

        for key in self.keys():
            value = self.__dict__[key]
            if isinstance(value, h5py.Dataset):
                out.__dict__[key] = value
            else:
                out.__dict__[key] = value[mask].copy()

        # store the mask operation in _lazy_ops for differed execution of attributes
        # that are not yet loaded
        if "mask" not in self._lazy_ops:
            out._lazy_ops["mask"] = mask
        else:
            # if a mask already exists, it is easy to combine the masks
            out._lazy_ops["mask"] = self._lazy_ops["mask"].copy()
            out._lazy_ops["mask"][out._lazy_ops["mask"]] = mask

        if "slice" in self._lazy_ops:
            out._lazy_ops["slice"] = self._lazy_ops["slice"]

        return out

    def _resolve_timestamps_after_slice(self):
        start, end, sequence_start, origin_translation = self._lazy_ops[
            "unresolved_slice"
        ]
        # sequence_start: Time corresponding to _timstamps_indices_1s[0]

        start_closest_sec_idx = np.clip(
            np.floor(start - sequence_start).astype(int),
            0,
            len(self._timestamp_indices_1s) - 1,
        )
        end_closest_sec_idx = np.clip(
            np.ceil(end - sequence_start).astype(int),
            0,
            len(self._timestamp_indices_1s) - 1,
        )

        idx_l = self._timestamp_indices_1s[start_closest_sec_idx]
        idx_r = self._timestamp_indices_1s[end_closest_sec_idx]

        timestamps = self.__dict__["timestamps"][idx_l:idx_r]

        idx_dl = np.searchsorted(timestamps, start)
        idx_dr = np.searchsorted(timestamps, end)
        timestamps = timestamps[idx_dl:idx_dr]

        idx_r = idx_l + idx_dr
        idx_l = idx_l + idx_dl

        del self._lazy_ops["unresolved_slice"]
        self._lazy_ops["slice"] = (idx_l, idx_r, start, origin_translation)
        self.__dict__["timestamps"] = timestamps - origin_translation

    def slice(self, start: float, end: float, reset_origin: bool = True):
        out = self.__class__.__new__(self.__class__)
        out._unicode_keys = self._unicode_keys
        out._lazy_ops = {}
        out._timekeys = self._timekeys

        out._domain = self._domain & Interval(start=start, end=end)
        if reset_origin:
            out._domain.start = out._domain.start - start
            out._domain.end = out._domain.end - start

        if isinstance(self.__dict__["timestamps"], h5py.Dataset):
            # lazy loading, we will only resolve timestamps if an attribute is accessed
            assert "slice" not in self._lazy_ops, "slice already exists"
            if "unresolved_slice" not in self._lazy_ops:
                origin_translation = start if reset_origin else 0.0
                out._lazy_ops["unresolved_slice"] = (
                    start,
                    end,
                    self._domain.start[0],
                    origin_translation,
                )
            else:
                # for some reason, blind slicing was done twice, and there is no need to
                # resolve the timestamps again
                curr_start, curr_end, sequence_start, origin_translation = (
                    self._lazy_ops["unresolved_slice"]
                )
                out._lazy_ops["unresolved_slice"] = (
                    start + origin_translation,
                    min(end + origin_translation, curr_end),
                    sequence_start,
                    origin_translation + (start if reset_origin else 0.0),
                )

            idx_l = idx_r = None
            out.__dict__["timestamps"] = self.__dict__["timestamps"]
            out._timestamp_indices_1s = self._timestamp_indices_1s
        else:
            assert (
                "unresolved_slice" not in self._lazy_ops
            ), "unresolved slice already exists"
            assert self.is_sorted(), "time series is not sorted, cannot slice"

            timestamps = self.timestamps
            idx_l = np.searchsorted(timestamps, start)
            idx_r = np.searchsorted(timestamps, end)

            timestamps = timestamps[idx_l:idx_r]
            out.__dict__["timestamps"] = timestamps - (start if reset_origin else 0.0)

            origin_translation = start if reset_origin else 0.0

            if "slice" not in self._lazy_ops:
                out._lazy_ops["slice"] = (idx_l, idx_r, start, origin_translation)
            else:
                out._lazy_ops["slice"] = (
                    self._lazy_ops["slice"][0] + idx_l,
                    self._lazy_ops["slice"][0] + idx_r,
                    self._lazy_ops["slice"][2] - start,
                    self._lazy_ops["slice"][3] + origin_translation,
                )

        for key in self.keys():
            if key != "timestamps":
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
        raise NotImplementedError("Cannot save a lazy array dict to hdf5.")

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

            import h5py
            from temporaldata import ArrayDict

            with h5py.File("data.h5", "r") as f:
                data = ArrayDict.from_hdf5(f)
        """
        assert (
            file.attrs["object"] == IrregularTimeSeries.__name__
        ), "object type mismatch"

        obj = cls.__new__(cls)
        for key, value in file.items():
            if key == "domain":
                obj.__dict__["_domain"] = Interval.from_hdf5(file[key])
            elif key == "timestamp_indices_1s":
                obj.__dict__["_timestamp_indices_1s"] = value[:]
            else:
                obj.__dict__[key] = value

        obj._unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()
        obj._timekeys = file.attrs["timekeys"].astype(str).tolist()
        obj._sorted = True
        obj._lazy_ops = {}

        return obj
