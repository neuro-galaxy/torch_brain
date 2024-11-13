from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Type
import collections
import logging
import warnings

import h5py
import numpy as np
import pandas as pd


class ArrayDict(object):
    r"""A dictionary of arrays that share the same first dimension. The number of
    dimensions for each array can be different, but they need to be at least
    1-dimensional.

    Args:
        **kwargs: arrays that shares the same first dimension.

    .. code-block:: python

        import numpy as np
        from temporaldata import ArrayDict

        units = ArrayDict(
            unit_id=np.array(["unit01", "unit02"]),
            brain_region=np.array(["M1", "M1"]),
            waveform_mean=np.zeros((2, 48)),
        )

        units
        >>> ArrayDict(
            unit_id=[2],
            brain_region=[2],
            waveform_mean=[2, 48]
        )

        len(units)
        >>> 2

        units.keys()
        >>> ['unit_id', 'brain_region', 'waveform_mean']

        'waveform_mean' in units
        >>> True


    .. note::
        Private attributes (starting with an underscore) do not need to be arrays,
        or have the same first dimension as the other attributes. They will not be
        listed in :meth:`keys()`.
    """

    def __init__(self, **kwargs: Dict[str, np.ndarray]):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def keys(self) -> List[str]:
        r"""Returns a list of all array attribute names."""
        return [x for x in self.__dict__.keys() if not x.startswith("_")]

    def _maybe_first_dim(self):
        # If self has at least one attribute, returns the first dimension of
        # the first attribute. Otherwise, returns :obj:`None`.
        if len(self.keys()) == 0:
            return None
        else:
            return self.__dict__[self.keys()[0]].shape[0]

    def __len__(self):
        r"""Returns the first dimension shared by all attributes."""
        first_dim = self._maybe_first_dim()
        if first_dim is None:
            raise ValueError(f"{self.__class__.__name__} is empty.")
        return first_dim

    def __setattr__(self, name, value):
        # for non-private attributes, we want to check that they are ndarrays
        # and that they match the first dimension of existing attributes
        if not name.startswith("_"):
            # only ndarrays are accepted
            assert isinstance(
                value, np.ndarray
            ), f"{name} must be a numpy array, got object of type {type(value)}"

            if value.ndim == 0:
                raise ValueError(
                    f"{name} must be at least 1-dimensional, got 0-dimensional array."
                )

            first_dim = self._maybe_first_dim()
            if first_dim is not None and value.shape[0] != first_dim:
                raise ValueError(
                    f"All elements of {self.__class__.__name__} must have the same "
                    f"first dimension. The first dimension of {name} is "
                    f"{value.shape[0]} but the first dimension of existing attributes "
                    f"is {first_dim}."
                )
        super(ArrayDict, self).__setattr__(name, value)

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the data."""
        return key in self.keys()

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        hidden_keys = ["train_mask", "valid_mask", "test_mask"]
        info = [
            size_repr(k, self.__dict__[k], indent=2)
            for k in self.keys()
            if k not in hidden_keys
        ]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"

    def select_by_mask(self, mask: np.ndarray, **kwargs):
        r"""Return a new :obj:`ArrayDict` object where all array attributes are indexed
        using the boolean mask.

        Args:
            mask: Boolean array used for masking. The mask needs to be 1-dimensional,
                and of equal length as the first dimension of the :obj:`ArrayDict`.
            **kwargs: Private attributes that will not be masked will need to be passed
                as arguments.

        .. code-block:: python

            import numpy as np
            from temporaldata import ArrayDict

            data = ArrayDict(
                unit_id=np.array(["unit01", "unit02"]),
                brain_region=np.array(["M1", "M1"]),
                waveform_mean=np.zeros((2, 48)),
            )

            data.select_by_mask(np.array([True, False]))
            >>> ArrayDict(
                unit_id=[1],
                brain_region=[1],
                waveform_mean=[1, 48]
            )
        """
        assert mask.ndim == 1, f"mask must be 1D, got {mask.ndim}D mask"
        assert mask.dtype == bool, f"mask must be boolean, got {mask.dtype}"

        first_dim = self._maybe_first_dim()
        if mask.shape[0] != first_dim:
            raise ValueError(
                f"mask length {mask.shape[0]} does not match first dimension of arrays "
                f"({first_dim})."
            )

        # kwargs are other private attributes
        # TODO automatically add private attributes
        return self.__class__(
            **{k: getattr(self, k)[mask].copy() for k in self.keys()}, **kwargs
        )

    @classmethod
    def from_dataframe(cls, df, unsigned_to_long=True, **kwargs):
        r"""Creates an :obj:`ArrayDict` object from a pandas DataFrame.

        The columns in the DataFrame are converted to arrays when possible, otherwise
        they will be skipped.

        Args:
            df (pandas.DataFrame): DataFrame.
            unsigned_to_long (bool, optional): If :obj:`True`, automatically converts
                unsigned integers to int64. Defaults to :obj:`True`.
        """
        data = {**kwargs}
        for column in df.columns:
            if column in cls.__dict__.keys():
                # We don't let users override existing attributes with this method,
                # since that is most likely a mistake.
                # Example: A dataframe might contain a 'split' attribute signifying
                # train/val/test splits.
                raise ValueError(
                    f"Attribute '{column}' already exists. Cannot override this "
                    f"attribute with the from_dataframe method. Please rename the "
                    f"attribute in the dataframe. If you really meant to override "
                    f"this attribute, please do so manually after the object is "
                    f"created."
                )
            if pd.api.types.is_numeric_dtype(df[column]):
                # Directly convert numeric columns to numpy arrays
                np_arr = df[column].to_numpy()
                # Convert unsigned integers to long
                if np.issubdtype(np_arr.dtype, np.unsignedinteger) and unsigned_to_long:
                    np_arr = np_arr.astype(np.int64)
                data[column] = np_arr
            elif df[column].apply(lambda x: isinstance(x, np.ndarray)).all():
                # Check if all ndarrays in the column have the same shape
                ndarrays = df[column]
                first_shape = ndarrays.iloc[0].shape
                if all(
                    arr.shape == first_shape
                    for arr in ndarrays
                    if isinstance(arr, np.ndarray)
                ):
                    # If all elements in the column are ndarrays with the same shape,
                    # stack them
                    np_arr = np.stack(df[column].values)
                    if (
                        np.issubdtype(np_arr.dtype, np.unsignedinteger)
                        and unsigned_to_long
                    ):
                        np_arr = np_arr.astype(np.int64)
                    data[column] = np_arr
                else:
                    logging.warning(
                        f"The ndarrays in column '{column}' do not all have the same shape."
                    )
            elif isinstance(df[column].iloc[0], str):
                try:  # try to see if unicode strings can be converted to fixed length ASCII bytes
                    df[column].to_numpy(dtype="S")
                except UnicodeEncodeError:
                    logging.warning(
                        f"Unable to convert column '{column}' to a numpy array. Skipping."
                    )
                else:
                    data[column] = df[column].to_numpy()

            else:
                logging.warning(
                    f"Unable to convert column '{column}' to a numpy array. Skipping."
                )
        return cls(**data)

    def to_hdf5(self, file):
        r"""Saves the data object to an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

            import h5py
            from temporaldata import ArrayDict

            data = ArrayDict(
                unit_id=np.array(["unit01", "unit02"]),
                brain_region=np.array(["M1", "M1"]),
                waveform_mean=np.zeros((2, 48)),
            )

            with h5py.File("data.h5", "w") as f:
                data.to_hdf5(f)
        """
        # save class name
        file.attrs["object"] = self.__class__.__name__

        # save attributes
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

        # save a list of the keys of the arrays that were originally unicode to
        # convert them back to unicode when loading
        file.attrs["_unicode_keys"] = np.array(_unicode_keys, dtype="S")

    @classmethod
    def from_hdf5(cls, file):
        r"""Loads the data object from an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyArrayDict.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from temporaldata import ArrayDict

            with h5py.File("data.h5", "r") as f:
                data = ArrayDict.from_hdf5(f)
        """
        if file.attrs["object"] != cls.__name__:
            raise ValueError(
                f"File contains data for a {file.attrs['object']} object, expected "
                f"{cls.__name__} object."
            )

        _unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()

        data = {}
        for key, value in file.items():
            data[key] = value[:]
            # if the values were originally unicode but stored as fixed length ASCII bytes
            if key in _unicode_keys:
                data[key] = data[key].astype("U")
        obj = cls(**data)

        return obj

    def __copy__(self):
        # create a shallow copy of the object
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        # create a deep copy of the object
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, h5py.Dataset):
                # h5py.File objects cannot be deepcopied
                result.__dict__[k] = v
            else:
                result.__dict__[k] = copy.deepcopy(v, memo)
        return result

    def materialize(self):
        r"""Materializes the data object, i.e., loads into memory all of the data that
        is still referenced in the HDF5 file."""
        for key in self.keys():
            # simply access all attributes to trigger the lazy loading
            getattr(self, key)


class LazyArrayDict(ArrayDict):
    r"""Lazy variant of :obj:`ArrayDict`. The data is not loaded until it is accessed.
    This class is meant to be used when the data is too large to fit in memory, and
    is intended to be intantiated via. :obj:`LazyArrayDict.from_hdf5`.

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
            for key in self.keys():
                value = self.__dict__[key]
                # check if an array is already loaded, return its first dimension
                if isinstance(value, np.ndarray):
                    return value.shape[0]

            # no array was loaded, check if there is a mask in _lazy_ops
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
            # intercept attribute calls. this is where data that is not loaded is loaded
            # and when any lazy operations are applied
            if name in self.keys():
                out = self.__dict__[name]

                if isinstance(out, h5py.Dataset):
                    # apply any mask, and return the numpy array
                    if "mask" in self._lazy_ops:
                        out = out[self._lazy_ops["mask"]]
                    else:
                        out = out[:]

                    # if the array was originally unicode, convert it back to unicode
                    if name in self._unicode_keys:
                        out = out.astype("U")

                    # store it, now the array is loaded
                    self.__dict__[name] = out

                # if all attributes are loaded, we can remove the lazy flag
                all_loaded = all(
                    isinstance(self.__dict__[key], np.ndarray) for key in self.keys()
                )
                if all_loaded:
                    self.__class__ = ArrayDict
                    # delete special private attributes
                    del self._lazy_ops, self._unicode_keys
                return out

        return super(LazyArrayDict, self).__getattribute__(name)

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
        # private attributes
        out._unicode_keys = self._unicode_keys
        out._lazy_ops = {}

        # array attributes
        for key in self.keys():
            value = self.__dict__[key]
            if isinstance(value, h5py.Dataset):
                # the mask will be applied when the getattr is called for this key
                # the details of the mask operation are stored in _lazy_ops
                out.__dict__[key] = value
            else:
                # this is a numpy array that is already loaded in memory, apply the mask
                out.__dict__[key] = value[mask].copy()

        # store the mask operation in _lazy_ops for differed execution
        if "mask" not in self._lazy_ops:
            out._lazy_ops["mask"] = mask
        else:
            # if a mask was already applied, we need to combine the masks
            out._lazy_ops["mask"] = self._lazy_ops["mask"].copy()
            out._lazy_ops["mask"][out._lazy_ops["mask"]] = mask

        return out

    @classmethod
    def from_dataframe(cls, df, unsigned_to_long=True):
        raise NotImplementedError("Cannot convert a dataframe to a lazy array dict.")

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
        assert file.attrs["object"] == ArrayDict.__name__, (
            f"File contains data for a {file.attrs['object']} object, expected "
            f"{ArrayDict.__name__} object."
        )

        obj = cls.__new__(cls)
        for key, value in file.items():
            obj.__dict__[key] = value

        obj._unicode_keys = file.attrs["_unicode_keys"].astype(str).tolist()
        obj._lazy_ops = {}

        return obj


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

    .. code-block:: python

        import numpy as np
        from temporaldata import IrregularTimeSeries

        spikes = IrregularTimeSeries(
            unit_index=np.array([0, 0, 1, 0, 1, 2]),
            timestamps=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            waveforms=np.zeros((6, 48)),
            domain="auto",
        )

        spikes
        >>> IrregularTimeseries(
            unit_index=[6],
            timestamps=[6],
            waveforms=[6, 48]
        )

        spikes.domain.start, spikes.domain.end
        >>> (array([0.1]), array([0.6]))

        spikes.keys()
        >>> ['unit_index', 'timestamps', 'waveforms']

        spikes.is_sorted()
        >>> True

        slice_of_spikes = spikes.slice(0.2, 0.5)
        >>> IrregularTimeseries(
            unit_index=[3],
            timestamps=[3],
            waveforms=[3, 48]
        )

        slice_of_spikes.domain.start, slice_of_spikes.domain.end
        >>> (array([0.]), array([0.3]))

        slice_of_spikes.timestamps
        >>> array([0. , 0.1, 0.2])
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
            assert ~np.any(np.isnan(value)), f"timestamps cannot contain NaNs."
            if value.dtype != np.float64:
                logging.warning(f"{name} is of type {value.dtype} not of type float64.")
            # timestamps has been updated, we no longer know whether it is sorted or not
            self._sorted = None

    def is_sorted(self):
        r"""Returns :obj:`True` if the timestamps are sorted."""
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None:
            self._sorted = np.all(self.timestamps[1:] >= self.timestamps[:-1])
        return self._sorted

    def _maybe_start(self) -> float:
        r"""Returns the start time of the time series. If the time series is not sorted,
        the start time is the minimum timestamp."""
        if self.is_sorted():
            return np.float64(self.timestamps[0])
        else:
            return np.float64(np.min(self.timestamps))

    def _maybe_end(self) -> float:
        r"""Returns the end time of the time series. If the time series is not sorted,
        the end time is the maximum timestamp."""
        if self.is_sorted():
            return np.float64(self.timestamps[-1])
        else:
            return np.float64(np.max(self.timestamps))

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

    def add_split_mask(self, name: str, interval: Interval):
        """Adds a boolean mask as an array attribute, which is defined for each
        timestamp, and is set to :obj:`True` for all timestamps that are within
        :obj:`interval`. The mask attribute will be called :obj:`<name>_mask`.

        This is used to mark points in the time series, as part of train, validation,
        or test sets, and is useful to ensure that there is no data leakage.

        Args:
            name: name of the split, e.g. "train", "valid", "test".
            interval: a set of intervals defining the split domain.
        """
        assert not hasattr(self, f"{name}_mask"), (
            f"Attribute {name}_mask already exists. Use another mask name, or rename "
            f"the existing attribute."
        )

        mask_array = np.zeros(len(self), dtype=bool)
        for start, end in zip(interval.start, interval.end):
            mask_array |= (self.timestamps >= start) & (self.timestamps < end)

        setattr(self, f"{name}_mask", mask_array)

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


class RegularTimeSeries(ArrayDict):
    """A regular time series is the same as an irregular time series, but it has a
    regular sampling rate. This allows for faster indexing, possibility of patching data
    and meaningful Fourier operations. The first dimension of all attributes must be
    the time dimension.

    .. note:: If you have a matrix of shape (N, T), where N is the number of channels and T is the number of time points, you should transpose it to (T, N) before passing it to the constructor, since the first dimension should always be time.

    Args:
        sampling_rate: Sampling rate in Hz.
        domain: an :obj:`Interval` object that defines the domain over which the
            timeseries is defined. It is not possible to set domain to :obj:`"auto"`.
        **kwargs: Arbitrary keyword arguments where the values are arbitrary
            multi-dimensional (2d, 3d, ..., nd) arrays with shape (N, \*).


    .. code-block:: python

        import numpy as np
        from temporaldata import RegularTimeSeries

        lfp = RegularTimeSeries(
            raw=np.zeros((1000, 128)),
            sampling_rate=250.,
            domain=Interval(0., 4.),
        )

        lfp.slice(0, 1)
        >>> RegularTimeSeries(
            raw=[250, 128]
        )

        lfp.to_irregular()
        >>> IrregularTimeSeries(
            timestamps=[1000],
            raw=[1000, 128]
        )
    """

    def __init__(
        self,
        *,
        sampling_rate: float,  # in Hz
        domain: Interval = None,
        domain_start=0.0,
        **kwargs: Dict[str, np.ndarray],
    ):
        super().__init__(**kwargs)

        self._sampling_rate = sampling_rate

        if domain == "auto":
            if not isinstance(domain_start, (int, float)):
                raise ValueError(
                    f"domain_start must be a number, got {type(domain_start)}."
                )
            domain = Interval(
                start=np.array([domain_start]),
                end=np.array([domain_start + (len(self) - 1) / sampling_rate]),
            )
        self._domain = domain

    @property
    def sampling_rate(self) -> float:
        r"""Returns the sampling rate in Hz."""
        return self._sampling_rate

    @property
    def domain(self) -> Interval:
        r"""Returns the domain of the time series."""
        return self._domain

    def timekeys(self):
        r"""Returns a list of all time-based attributes."""
        return self._timekeys

    def select_by_mask(self, mask: np.ndarray):
        raise NotImplementedError("Not implemented for RegularTimeSeries.")

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`RegularTimeSeries` object that contains the data between
        the start (inclusive) and end (exclusive) times.

        When slicing, the start and end times are rounded to the nearest timestamp.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
        """
        # we allow the start and end to be outside the domain of the time series
        if start < self.domain.start[0]:
            start_id = 0
        else:
            start_id = int(np.ceil((start - self.domain.start[0]) * self.sampling_rate))

        if end > self.domain.end[0]:
            end_id = len(self) + 1
        else:
            end_id = int(np.floor((end - self.domain.start[0]) * self.sampling_rate))

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate
        out._domain = copy.deepcopy(self._domain)
        if reset_origin:
            out._domain.start, out._domain.end = (
                out._domain.start - start,
                out._domain.end - start,
            )

        for key in self.keys():
            out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        return out

    def add_split_mask(
        self,
        name: str,
        interval: Interval,
    ):
        """Adds a boolean mask as an array attribute, which is defined for each
        timestamp, and is set to :obj:`True` for all timestamps that are within
        :obj:`interval`. The mask attribute will be called :obj:`<name>_mask`.

        This is used to mark points in the time series, as part of train, validation,
        or test sets, and is useful to ensure that there is no data leakage.

        Args:
            name: name of the split, e.g. "train", "valid", "test".
            interval: a set of intervals defining the split domain.
        """
        assert not hasattr(self, f"{name}_mask"), (
            f"Attribute {name}_mask already exists. Use another mask name, or rename "
            f"the existing attribute."
        )

        mask_array = np.zeros_like(self.timestamps, dtype=bool)
        for start, end in zip(interval.start, interval.end):
            if start < self.domain.start[0]:
                start_id = 0
            else:
                start_id = int(
                    np.ceil((start - self.domain.start[0]) * self.sampling_rate)
                )

            if end > self.domain.end[0]:
                end_id = len(self) + 1
            else:
                end_id = int(
                    np.floor((end - self.domain.start[0]) * self.sampling_rate)
                )

            assert not np.any(mask_array[start_id:end_id])
            mask_array[start_id:end_id] = True

        setattr(self, f"{name}_mask", mask_array)

    def to_irregular(self):
        r"""Converts the time series to an irregular time series."""
        return IrregularTimeSeries(
            timestamps=self.timestamps,
            **{k: getattr(self, k) for k in self.keys()},
            domain=self.domain,
        )

    @property
    def timestamps(self):
        r"""Returns the timestamps of the time series."""
        return (
            self.domain.start[0]
            + np.arange(len(self), dtype=np.float64) / self.sampling_rate
        )

    def to_hdf5(self, file):
        r"""Saves the data object to an HDF5 file.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

                import h5py
                from temporaldata import RegularTimeSeries

                data = RegularTimeSeries(
                    raw=np.zeros((1000, 128)),
                    sampling_rate=250.,
                    domain=Interval(0., 4.),
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
            file (h5py.File): HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyRegularTimeSeries.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from temporaldata import RegularTimeSeries

            with h5py.File("data.h5", "r") as f:
                data = RegularTimeSeries.from_hdf5(f)
        """
        assert file.attrs["object"] == cls.__name__, "object type mismatch"

        data = {}
        for key, value in file.items():
            if key != "domain":
                data[key] = value[:]

        domain = Interval.from_hdf5(file["domain"])
        obj = cls(**data, sampling_rate=file.attrs["sampling_rate"], domain=domain)

        return obj


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

    _lazy_ops = dict()

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
                return int(
                    np.round(
                        (self.domain.end[-1] - self.domain.start[0])
                        * self.sampling_rate
                    )
                )

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
        return super(LazyRegularTimeSeries, self).__getattribute__(name)

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`RegularTimeSeries` object that contains the data between
        the start and end times.
        """

        start_id = int(np.floor((start - self.domain.start[0]) * self.sampling_rate))
        end_id = int(np.floor((end - self.domain.start[0]) * self.sampling_rate))

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate
        out._lazy_ops = {}
        if reset_origin:
            out._domain = Interval(start=np.array([0.0]), end=np.array([end - start]))
        else:
            out._domain = self._domain & Interval(start=start, end=end)

        for key in self.keys():
            if isinstance(self.__dict__[key], h5py.Dataset):
                out.__dict__[key] = self.__dict__[key]
            else:
                out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        if "slice" not in self._lazy_ops:
            out._lazy_ops["slice"] = (start_id, end_id)
        else:
            out._lazy_ops["slice"] = (
                self._lazy_ops["slice"][0] + start_id,
                self._lazy_ops["slice"][0] + end_id,
            )

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

    .. code-block:: python

        import numpy as np
        from temporaldata import Interval

        intervals = Interval(
            start=np.array([0, 1, 2]),
            end=np.array([1, 2, 3]),
            go_cue_time=np.array([0.5, 1.5, 2.5]),
            drifting_gratings_dir=np.array([0, 45, 90]),
            timekeys=["start", "end", "go_cue_time"],
        )

        intervals
        >>> Interval(
            start=[3],
            end=[3],
            go_cue_time=[3],
            drifting_gratings_dir=[3]
        )

        intervals.keys()
        >>> ['start', 'end', 'go_cue_time', 'drifting_gratings_dir']

        intervals.is_sorted()
        >>> True

        intervals.is_disjoint()
        >>> True

        intervals.slice(0.5, 2.5)
        >>> Interval(
            start=[2],
            end=[2],
            go_cue_time=[2],
            drifting_gratings_dir=[2]
        )

        Interval(0., 1.)
        >>> Interval(
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
        if timekey not in self.keys:
            raise ValueError(f"'{timekey}' cannot be found in \n {self}.")
        if timekey not in self._timekeys:
            self._timekeys.append(timekey)

    def __setattr__(self, name, value):
        super(Interval, self).__setattr__(name, value)

        if name == "start" or name == "end":
            assert value.ndim == 1, f"{name} must be 1D."
            assert ~np.any(np.isnan(value)), f"{name} cannot contain NaNs."
            if value.dtype != np.float64:
                logging.warning(f"{name} is of type {value.dtype} not of type float64.")
            # start or end have been updated, we no longer know whether it is sorted
            # or not
            self._sorted = None

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
        return np.all(self.end[:-1] <= self.start[1:])

    def is_sorted(self):
        r"""Returns :obj:`True` if the intervals are sorted."""
        # check if we already know that the sequence is sorted
        # if lazy loading, we'll have to skip this check
        if self._sorted is None:
            self._sorted = np.all(self.start[1:] >= self.start[:-1]) and np.all(
                self.end[1:] >= self.end[:-1]
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
            sizes: A list of integers or floats. If integers, the list must sum to the
            number of intervals. If floats, the list must sum to 1.0.
            shuffle: If :obj:`True`, the intervals will be shuffled before splitting.
            random_seed: The random seed to use for shuffling.

        .. note::
            This method will not guarantee that the resulting sets will be disjoint, if
            the intervals are not already disjoint.
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

        .. code-block:: python

            from temporaldata import Interval

            interval = Interval.linspace(0., 10., 100)

            interval
            >>> Interval(
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

        .. code-block:: python

            from temporaldata import Interval

            interval_list = [(0, 1), (1, 2), (2, 3)]
            interval = Interval.from_list(interval_list)

            interval.start, interval.end
            >>> (array([0, 1, 2]), array([1, 2, 3]))
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

    while (lidx < len(lintervals)) or (ridx < len(rintervals)):
        # retrieve the time of the pointer in the left object
        if lidx < len(lintervals):
            # retrieve the time of the next interval in left object
            ltime = lintervals.start[lidx] if lop else lintervals.end[lidx]
        else:
            # exhausted all intervals in left object
            ltime = np.inf

        # retrieve the time of the pointer in the right object
        if ridx < len(rintervals):
            # retrieve the time of the next interval in right object
            rtime = rintervals.start[ridx] if rop else rintervals.end[ridx]
        else:
            # exhausted all intervals in right object
            rtime = np.inf

        # figure out which is the next pointer to process
        if (ltime < rtime) or (ltime == rtime and lop):
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
        else:
            # the next timestamps to consider is from the right object
            ptime = rtime
            pop = rop
            pl = False
            if rop:
                rop = False
            else:
                rop = True
                ridx += 1
        yield ptime, pop, pl


class Data(object):
    r"""A data object is a container for other data objects such as :obj:`ArrayDict`,
     :obj:`RegularTimeSeries`, :obj:`IrregularTimeSeries`, and :obj:`Interval` objects.
     But also regular objects like sclars, strings and numpy arrays.

    Args:
        start: Start time.
        end: End time.
        **kwargs: Arbitrary attributes.

    .. code-block:: python

        import numpy as np
        from temporaldata import (
            ArrayDict,
            IrregularTimeSeries,
            RegularTimeSeries,
            Interval,
            Data,
        )

        data = Data(
            session_id="session_0",
            spikes=IrregularTimeSeries(
                timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
                unit_index=np.array([0, 0, 1, 0, 1, 2]),
                waveforms=np.zeros((6, 48)),
                domain=Interval(0., 3.),
            ),
            lfp=RegularTimeSeries(
                raw=np.zeros((1000, 3)),
                sampling_rate=250.,
                domain=Interval(0., 4.),
            ),
            units=ArrayDict(
                id=np.array(["unit_0", "unit_1", "unit_2"]),
                brain_region=np.array(["M1", "M1", "PMd"]),
            ),
            trials=Interval(
                start=np.array([0, 1, 2]),
                end=np.array([1, 2, 3]),
                go_cue_time=np.array([0.5, 1.5, 2.5]),
                drifting_gratings_dir=np.array([0, 45, 90]),
            ),
            drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
            domain=Interval(0., 4.),
        )

        data
        >>> Data(
            session_id='session_0',
            spikes=IrregularTimeSeries(
                timestamps=[6],
                unit_index=[6],
                waveforms=[6, 48]
            ),
            lfp=RegularTimeSeries(
                raw=[1000, 3]
            ),
            units=ArrayDict(
                id=[3],
                brain_region=[3]
            ),
            trials=Interval(
                start=[3],
                end=[3],
                go_cue_time=[3],
                drifting_gratings_dir=[3]
            ),
            drifting_gratings_imgs=[8, 3, 32, 32]
            _domain=Interval(
                start=[1],
                end=[1]
            ),
        )

        data.slice(1, 3)
        >>> Data(
            session_id='session_0',
            spikes=IrregularTimeSeries(
                timestamps=[3],
                unit_index=[3],
                waveforms=[3, 48]
            ),
            lfp=RegularTimeSeries(
                raw=[500, 3]
            ),
            units=ArrayDict(
                id=[3],
                brain_region=[3]
            ),
            trials=Interval(
                start=[2],
                end=[2],
                go_cue_time=[2],
                drifting_gratings_dir=[2]
            ),
            drifting_gratings_imgs=[8, 3, 32, 32],
            _domain=Interval(
                start=[1],
                end=[1]
            ),
            _absolute_start=1.0,
            )
    """

    _absolute_start = 0.0
    _domain = None

    def __init__(
        self,
        *,
        domain=None,
        **kwargs: Dict[str, Union[str, float, int, np.ndarray, ArrayDict]],
    ):
        if domain == "auto":
            # the domain is the union of the domains of the attributes
            domain = Interval(np.array([]), np.array([]))
            for key, value in kwargs.items():
                if isinstance(value, (IrregularTimeSeries, RegularTimeSeries)):
                    domain = domain | value.domain
                if isinstance(value, Interval):
                    domain = domain | value
                if isinstance(value, Data) and value.domain is not None:
                    domain = domain | value.domain

        if domain is not None and not isinstance(domain, Interval):
            raise ValueError("domain must be an Interval object.")

        self._domain = domain

        for key, value in kwargs.items():
            setattr(self, key, value)

        # these variables will hold the original start and end times
        # and won't be modified when slicing
        # self.original_start = start
        # self.original_end = end

        # if any time-based attribute is present, start and end must be specified
        # todo check domain, also check when a new attribute is set

    def __setattr__(self, name, value):
        if name != "_domain" and (
            (
                isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval))
                and self.domain is None
            )
            or (
                isinstance(value, Data)
                and self.domain is None
                and value.domain is not None
            )
        ):
            raise ValueError(
                f"Data object must have a domain if it contains a time-based attribute "
                f"({name})."
            )
        super().__setattr__(name, value)

    @property
    def domain(self):
        r"""Returns the domain of the data object."""
        return self._domain

    @property
    def start(self):
        r"""Returns the start time of the data object."""
        return self.domain.start[0] if self.domain is not None else None

    @property
    def end(self):
        r"""Returns the end time of the data object."""
        return self.domain.end[-1] if self.domain is not None else None

    @property
    def absolute_start(self):
        r"""Returns the start time of this slice relative to the original start time.
        Should be 0. if the data object has not been sliced.

        .. code-block:: python

            # Assuming `data` is a Data object that hasn't been sliced
            data.absolute_start
            >>> 0.0

            data = data.slice(1, 3)
            data.absolute_start
            >>> 1.0

            data = data.slice(0.4, 1.4)
            data.absolute_start
            >>> 1.4
        """
        return self._absolute_start if self.domain is not None else None

    def slice(self, start: float, end: float, reset_origin: bool = True):
        r"""Returns a new :obj:`Data` object that contains the data between the start
        and end times. This method will slice all time-based attributes that are present
        in the data object.

        Args:
            start: Start time.
            end: End time.
            reset_origin: If :obj:`True`, all time attributes will be updated to be
                relative to the new start time. Defaults to :obj:`True`.
        """
        if self.domain is None:
            raise ValueError(
                "Data object does not contain any time-based attributes, "
                "and can thus not be sliced."
            )

        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            # todo update domain
            if key != "_domain" and (
                isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval))
                or (isinstance(value, Data) and value.domain is not None)
            ):
                out.__dict__[key] = value.slice(start, end, reset_origin)
            else:
                out.__dict__[key] = copy.copy(value)

        # update domain
        out._domain = copy.copy(self._domain) & Interval(start, end)
        if reset_origin:
            out._domain.start -= start
            out._domain.end -= start

            # update slice start time
            out._absolute_start = self._absolute_start + start

        return out

    def select_by_interval(self, interval: Interval):
        r"""Return a new :obj:`IrregularTimeSeries` object where all timestamps are
        within the interval.

        Args:
            interval: Interval object.
        """
        if self.domain is None:
            raise ValueError(
                "Data object does not contain any time-based attributes, "
                "and can thus not be sliced."
            )

        out = self.__class__.__new__(self.__class__)

        for key, value in self.__dict__.items():
            # todo update domain
            if key != "_domain" and (
                isinstance(value, (IrregularTimeSeries, RegularTimeSeries, Interval))
                or (isinstance(value, Data) and value.domain is not None)
            ):
                if isinstance(value, RegularTimeSeries):
                    value = value.to_irregular()
                out.__dict__[key] = value.select_by_interval(interval)
            else:
                out.__dict__[key] = copy.copy(value)

        out._domain = self._domain & interval
        return out

    def __repr__(self) -> str:
        cls = self.__class__.__name__

        info = ""
        for key, value in self.__dict__.items():
            if key == "_domain":
                continue
            if isinstance(value, ArrayDict):
                info = info + key + "=" + repr(value) + ",\n"
            elif value is not None:
                info = info + size_repr(key, value) + ",\n"
        info = info.rstrip()
        return f"{cls}(\n{info}\n)"

    def to_dict(self) -> Dict[str, Any]:
        r"""Returns a dictionary of stored key/value pairs."""
        return copy.deepcopy(self.__dict__)

    def to_hdf5(self, file, serialize_fn_map=None):
        r"""Saves the data object to an HDF5 file. This method will also call the
        `to_hdf5` method of all contained data objects, so that the entire data object
        is saved to the HDF5 file, i.e. no need to call `to_hdf5` for each contained
        data object.

        Args:
            file (h5py.File): HDF5 file.

        .. code-block:: python

                import h5py
                from temporaldata import Data

                data = Data(...)

                with h5py.File("data.h5", "w") as f:
                    data.to_hdf5(f)
        """
        for key in self.keys():
            value = getattr(self, key)
            if isinstance(value, (Data, ArrayDict)):
                grp = file.create_group(key)
                if isinstance(value, Data):
                    value.to_hdf5(grp, serialize_fn_map=serialize_fn_map)
                else:
                    value.to_hdf5(grp)
            elif isinstance(value, np.ndarray):
                # todo add warning if array is too large
                # recommend using ArrayDict
                file.create_dataset(key, data=value)
            elif value is not None:
                # each attribute should be small (generally < 64k)
                # there is no partial I/O; the entire attribute must be read
                value = serialize(value, serialize_fn_map=serialize_fn_map)
                file.attrs[key] = value

        if self._domain is not None:
            grp = file.create_group("domain")
            self._domain.to_hdf5(grp)

        file.attrs["object"] = "Data"
        file.attrs["absolute_start"] = self._absolute_start

    @classmethod
    def from_hdf5(cls, file, lazy=True):
        r"""Loads the data object from an HDF5 file. This method will also call the
        `from_hdf5` method of all contained data objects, so that the entire data object
        is loaded from the HDF5 file, i.e. no need to call `from_hdf5` for each contained
        data object.

        Args:
            file (h5py.File): HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyData.from_hdf5` instead.


        .. code-block:: python

            import h5py
            from temporaldata import Data

            with h5py.File("data.h5", "r") as f:
                data = Data.from_hdf5(f)
        """
        # check that the file is read-only
        if isinstance(file, h5py.File):
            assert file.mode == "r", "File must be opened in read-only mode."

        data = {}
        for key, value in file.items():
            if isinstance(value, h5py.Group):
                class_name = value.attrs["object"]
                if lazy and class_name != "Data":
                    group_cls = globals()[f"Lazy{class_name}"]
                else:
                    group_cls = globals()[class_name]
                data[key] = group_cls.from_hdf5(value)
            else:
                # if array, it will be loaded no matter what, always prefer ArrayDict
                data[key] = value[:]

        for key, value in file.attrs.items():
            if key == "object" or key == "absolute_start":
                continue
            data[key] = value

        obj = cls(**data)

        # restore the absolute start time
        obj._absolute_start = file.attrs["absolute_start"]

        return obj

    def set_train_domain(self, interval: Interval):
        """Set the train domain for all attributes."""
        self.train_domain = interval
        self.add_split_mask("train", interval)

    def set_valid_domain(self, interval: Interval):
        """Set the valid domain for all attributes."""
        self.valid_domain = interval
        self.add_split_mask("valid", interval)

    def set_test_domain(self, interval: Interval):
        """Set the test domain for all attributes."""
        self.test_domain = interval
        self.add_split_mask("test", interval)

    def add_split_mask(
        self,
        name: str,
        interval: Interval,
    ):
        """Create split masks for all Data, Interval & IrregularTimeSeries objects
        contained within this Data object.
        """
        for key in self.keys():
            if key.endswith("_domain"):
                # domains are not split
                assert isinstance(getattr(self, key), Interval)
                continue
            obj = getattr(self, key)
            if isinstance(obj, (RegularTimeSeries, IrregularTimeSeries, Interval)):
                obj.add_split_mask(name, interval)

    def _check_for_data_leakage(self, name):
        """Ensure that split masks are all True"""
        for key in self.keys():
            if key.endswith("_domain"):
                continue
            if key == "trials":
                # raise deprecation warning
                if (
                    not hasattr(obj, f"{name}_mask")
                    or not getattr(obj, f"{name}_mask").all()
                ):
                    warnings.warn(
                        "Data leakage was detected in 'trials'. This is a warning, but"
                        "in the future, this will raise an error. Please update your "
                        "prepare_data.py script to fix this issue.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                continue
            obj = getattr(self, key)
            if isinstance(obj, (IrregularTimeSeries, Interval)):
                assert hasattr(obj, f"{name}_mask"), (
                    f"Split mask for '{name}' not found in Data object. "
                    f"Please register this split in prepare_data.py using "
                    f"the session.register_split(...) method. In Data object: \n"
                    f"{self}"
                )
                assert getattr(obj, f"{name}_mask").all(), (
                    f"Data leakage detected split mask for '{name}' is not all True "
                    f"in self.{key}."
                )
            if isinstance(obj, Data):
                obj._check_for_data_leakage(name)

    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return [x for x in self.__dict__.keys() if not x.startswith("_")]

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys()

    def get_nested_attribute(self, path: str) -> Any:
        r"""Returns the attribute specified by the path. The path can be nested using
        dots. For example, if the path is "spikes.timestamps", this method will return
        the timestamps attribute of the spikes object.

        Args:
            path: Nested attribute path.
        """
        # Split key by dots, resolve using getattr
        components = path.split(".")
        out = self
        for c in components:
            try:
                out = getattr(out, c)
            except AttributeError:
                raise AttributeError(
                    f"Could not resolve {path} in data (specifically, at level {c}))"
                )
        return out

    def __copy__(self):
        # create a shallow copy of the object
        # the full skeleton of the Data object, i.e. including all ArrayDict children,
        # will be copied. However, the data itself (np.ndarray, etc.) will not be
        # copied.
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if isinstance(v, ArrayDict):
                setattr(result, k, copy.copy(v))
            else:
                setattr(result, k, v)
        return result

    def __deepcopy__(self, memo):
        # create a deep copy of the object
        # h5py objects will not be deepcopied, we only allow read-only access to the
        # HDF5 file, so this should not be an issue.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if isinstance(v, h5py.Dataset):
                # h5py.File objects cannot be deepcopied
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def materialize(self):
        r"""Materializes the data object, i.e., loads into memory all of the data that
        is still referenced in the HDF5 file."""
        for key in self.keys():
            # simply access all attributes to trigger the lazy loading
            if isinstance(getattr(self, key), (Data, ArrayDict)):
                getattr(self, key).materialize()


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, np.ndarray):
        out = str(list(value.shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, Sequence):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif (
        isinstance(value, Mapping)
        and len(value) == 1
        and not isinstance(list(value.values())[0], Mapping)
    ):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + pad + "}"
    else:
        out = str(value)
    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


def serialize(
    elem,
    serialize_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    r"""
    General serialization function that handles object types that are not supported
    by hdf5. The function also opens function registry to deal with specific element
    types through `serialize_fn_map`. This function will automatically be applied to
    elements in a nested sequence structure.

    Args:
        elem: a single element to be serialized.
        serialize_fn_map: Optional dictionary mapping from element type to the
            corresponding serialize function. If the element type isn't present in this
            dictionary, it will be skipped and the element will be returned as is.
    """
    elem_type = type(elem)

    if serialize_fn_map is not None:
        if elem_type in serialize_fn_map:
            return serialize_fn_map[elem_type](elem, serialize_fn_map=serialize_fn_map)

        for object_type in serialize_fn_map:
            if isinstance(elem, object_type):
                return serialize_fn_map[object_type](
                    elem, serialize_fn_map=serialize_fn_map
                )

    if isinstance(elem, (list, tuple)):
        return elem_type(
            [serialize(e, serialize_fn_map=serialize_fn_map) for e in elem]
        )

    # element does not need to be seralized, or type not supported
    return elem
