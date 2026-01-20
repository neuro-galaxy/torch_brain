from __future__ import annotations

from typing import Dict

import h5py
import numpy as np

from .arraydict import ArrayDict
from .interval import Interval
from .irregular_ts import IrregularTimeSeries


class RegularTimeSeries(ArrayDict):
    r"""A regular time series is the same as an irregular time series, but it has a
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


    Example ::

        >>> import numpy as np
        >>> from temporaldata import RegularTimeSeries

        >>> lfp = RegularTimeSeries(
        ...     raw=np.zeros((1000, 128)),
        ...     sampling_rate=250.,
        ...     domain=Interval(0., 4.),
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
            out_start = self.domain.start[0]
        else:
            start_id = int(np.ceil((start - self.domain.start[0]) * self.sampling_rate))
            out_start = self.domain.start[0] + start_id * 1.0 / self.sampling_rate

        if end > self.domain.end[0]:
            end_id = len(self) + 1
            out_end = self.domain.end[0]
        else:
            end_id = int(np.floor((end - self.domain.start[0]) * self.sampling_rate))
            out_end = self.domain.start[0] + (end_id - 1) * 1.0 / self.sampling_rate

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate

        out._domain = Interval(start=out_start, end=out_end)

        if reset_origin:
            out._domain.start = out._domain.start - start
            out._domain.end = out._domain.end - start

        for key in self.keys():
            out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        return out

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
                domain_length = self.domain.end[-1] - self.domain.start[0]
                return int(np.round(domain_length * self.sampling_rate)) + 1

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
        if start < self.domain.start[0]:
            start_id = 0
            out_start = self.domain.start[0]
        else:
            start_id = int(np.ceil((start - self.domain.start[0]) * self.sampling_rate))
            out_start = self.domain.start[0] + start_id * 1.0 / self.sampling_rate

        if end > self.domain.end[0]:
            end_id = len(self) + 1
            out_end = self.domain.end[0]
        else:
            end_id = int(np.floor((end - self.domain.start[0]) * self.sampling_rate))
            out_end = self.domain.start[0] + (end_id - 1) * 1.0 / self.sampling_rate

        out = self.__class__.__new__(self.__class__)
        out._sampling_rate = self.sampling_rate

        out._domain = Interval(start=out_start, end=out_end)

        if reset_origin:
            out._domain.start = out._domain.start - start
            out._domain.end = out._domain.end - start

        for key in self.keys():
            if isinstance(self.__dict__[key], h5py.Dataset):
                out.__dict__[key] = self.__dict__[key]
            else:
                out.__dict__[key] = self.__dict__[key][start_id:end_id].copy()

        out._lazy_ops = {}

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
