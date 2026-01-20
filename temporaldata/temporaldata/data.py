from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple, Union, Callable, Optional, Type
from pathlib import Path
import warnings

import h5py
import numpy as np

from .arraydict import ArrayDict, LazyArrayDict
from .irregular_ts import IrregularTimeSeries, LazyIrregularTimeSeries
from .regular_ts import RegularTimeSeries, LazyRegularTimeSeries
from .interval import Interval, LazyInterval
from .utils import _size_repr


class Data(object):
    r"""A data object is a container for other data objects such as :obj:`ArrayDict`,
     :obj:`RegularTimeSeries`, :obj:`IrregularTimeSeries`, and :obj:`Interval` objects.
     But also regular objects like sclars, strings and numpy arrays.

    Args:
        start: Start time.
        end: End time.
        **kwargs: Arbitrary attributes.

    Example ::

        >>> import numpy as np
        >>> from temporaldata import (
        ...     ArrayDict,
        ...     IrregularTimeSeries,
        ...     RegularTimeSeries,
        ...     Interval,
        ...     Data,
        ... )

        >>> data = Data(
        ...     session_id="session_0",
        ...     spikes=IrregularTimeSeries(
        ...         timestamps=np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
        ...         unit_index=np.array([0, 0, 1, 0, 1, 2]),
        ...         waveforms=np.zeros((6, 48)),
        ...         domain=Interval(0., 3.),
        ...     ),
        ...     lfp=RegularTimeSeries(
        ...         raw=np.zeros((1000, 3)),
        ...         sampling_rate=250.,
        ...         domain=Interval(0., 4.),
        ...     ),
        ...     units=ArrayDict(
        ...         id=np.array(["unit_0", "unit_1", "unit_2"]),
        ...         brain_region=np.array(["M1", "M1", "PMd"]),
        ...     ),
        ...     trials=Interval(
        ...         start=np.array([0, 1, 2]),
        ...         end=np.array([1, 2, 3]),
        ...         go_cue_time=np.array([0.5, 1.5, 2.5]),
        ...         drifting_gratings_dir=np.array([0, 45, 90]),
        ...     ),
        ...     drifting_gratings_imgs=np.zeros((8, 3, 32, 32)),
        ...     domain=Interval(0., 4.),
        ... )

        >>> data
        Data(
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
        drifting_gratings_imgs=[8, 3, 32, 32],
        )

        >>> data.slice(1, 3)
        Data(
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

        Example ::

            >>> from temporaldata import Data
            >>> data = Data(domain=Interval(0., 4.))

            >>> data.absolute_start
            0.0

            >>> data = data.slice(1, 3)
            >>> data.absolute_start
            1.0

            >>> data = data.slice(0.4, 1.4)
            >>> data.absolute_start
            1.4
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
                info = info + _size_repr(key, value) + ",\n"
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
                if class_name == "Data":
                    data[key] = group_cls.from_hdf5(value, lazy=lazy)
                else:
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

    @classmethod
    def load(cls, path: Union[Path, str], lazy: bool = True) -> Data:
        r"""Loads the :class:`Data` object from an HDF5 file given its file path.

        Args:
            path: The file path to the HDF5 file containing the :class:`Data` object.
            lazy: If True (default), load contained objects in lazy mode
                (using LazyArrayDict, LazyRegularTimeSeries, etc.); if False,
                read all data immediately into memory.

        Returns:
            Data: The loaded :class:`Data` object from the HDF5 file.

        .. code-block:: python

            from temporaldata import Data

            data_lazy = Data.load("data.h5")
            data_materialized = Data.load("data.h5", lazy=False)
        """
        file = h5py.File(path)
        return cls.from_hdf5(file, lazy=lazy)

    def set_train_domain(self, interval: Interval):
        """Deprecated no-op retained for backward compatibility."""
        warnings.warn(
            "set_train_domain() is being deprecated and will be removed in a future version. "
            "Please directly set the train_domain attribute of this Data object.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.train_domain = interval

    def set_valid_domain(self, interval: Interval):
        """Deprecated no-op retained for backward compatibility."""
        warnings.warn(
            "set_valid_domain() is being deprecated and will be removed in a future version. "
            "Please directly set the valid_domain attribute of this Data object.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.valid_domain = interval

    def set_test_domain(self, interval: Interval):
        """Deprecated no-op retained for backward compatibility."""
        warnings.warn(
            "set_test_domain() is being deprecated and will be removed in a future version. "
            "Please directly set the test_domain attribute of this Data object.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.test_domain = interval

    def _check_for_data_leakage(self, *args, **kwargs):
        """Deprecated no-op retained for backward compatibility.

        The ``_check_for_data_leakage`` method no longer performs any validation.
        Actual data leakage checks should be performed by the sampler.
        """
        warnings.warn(
            "_check_for_data_leakage() is being deprecated and will be removed in a future version. ",
            DeprecationWarning,
            stacklevel=2,
        )
        return True

    def keys(self) -> List[str]:
        r"""Returns a list of all attribute names."""
        return list(filter(lambda x: not x.startswith("_"), self.__dict__))

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
                    f"Could not resolve {path} in data (specifically, at level {c})"
                )
        return out

    def has_nested_attribute(self, path: str) -> bool:
        """Check if the attribute specified by the path exists in the Data object."""
        if not path:
            return False

        current_obj = self
        attribute_names = path.split(".")

        for name in attribute_names:
            try:
                current_obj = current_obj.__dict__[name]
            except KeyError:
                return False

        return True

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

    def materialize(self) -> Data:
        r"""Materializes the data object, i.e., loads into memory all of the data that
        is still referenced in the HDF5 file."""
        for key in self.keys():
            # simply access all attributes to trigger the lazy loading
            if isinstance(getattr(self, key), (Data, ArrayDict)):
                getattr(self, key).materialize()

        if self.domain is not None:
            self.domain.materialize()

        return self


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
