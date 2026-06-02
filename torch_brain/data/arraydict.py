from __future__ import annotations

import copy
import logging
from typing import List

import h5py
import numpy as np
import pandas as pd

from .typing import ArrayLike
from .utils import _size_repr, _validate_select_by_mask_input


class ArrayDict:
    r"""A dictionary of arrays that share the same first dimension. The number of
    dimensions for each array can be different, but they need to be at least
    1-dimensional.

    Args:
        **kwargs: arrays that shares the same first dimension.

    Example ::

        >>> from torch_brain.data import ArrayDict
        >>> import numpy as np

        >>> units = ArrayDict(
        ...     unit_id=["unit01", "unit02"],
        ...     brain_region=["M1", "M1"],
        ...     waveform_mean=np.random.rand(2, 48),
        ... )

        >>> units
        ArrayDict(
          unit_id=[2],
          brain_region=[2],
          waveform_mean=[2, 48]
        )
    """

    def __init__(self, **kwargs: ArrayLike):
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def keys(self) -> list[str]:
        r"""Returns a list of all array attribute names."""
        return list(filter(lambda x: not x.startswith("_"), self.__dict__))

    def _maybe_first_dim(self):
        # If self has at least one attribute, returns the first dimension of
        # the first attribute. Otherwise, returns :obj:`None`.
        keys = self.keys()
        if len(keys) == 0:
            return None
        else:
            return self.__dict__[keys[0]].shape[0]

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
            value = np.asarray(value)

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
        super().__setattr__(name, value)

    def __getattr__(self, name) -> np.ndarray:
        raise AttributeError(f"Attribute {name} not found.")

    def __contains__(self, key: str) -> bool:
        r"""Returns :obj:`True` if the attribute :obj:`key` is present in the data."""
        return key in self.keys()

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = [_size_repr(k, self.__dict__[k], indent=2) for k in self.keys()]
        info = ",\n".join(info)
        return f"{cls}(\n{info}\n)"

    def select_by_mask(self, mask: np.ndarray):
        r"""Index all arrays with a boolean mask and return a copy.

        Args:
            mask: Boolean array used for masking. The mask needs to be 1-dimensional,
                and of equal length as the object itself.

        Example ::

            >>> from torch_brain.data import ArrayDict
            >>> import numpy as np

            >>> units = ArrayDict(
            ...     unit_id=["unit01", "unit02"],
            ...     brain_region=["M1", "M1"],
            ...     waveform_mean=np.random.rand(2, 48),
            ... )

            >>> units_subset = units.select_by_mask([True, False])
            >>> units_subset
            ArrayDict(
              unit_id=[1],
              brain_region=[1],
              waveform_mean=[1, 48]
            )

        """
        _validate_select_by_mask_input(mask, len(self))

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                out.__dict__[key] = copy.deepcopy(value)
            else:
                out.__dict__[key] = value[mask].copy()

        return out

    @classmethod
    def from_dataframe(cls, df, unsigned_to_long=True, **kwargs):
        r"""Creates an :obj:`ArrayDict` object from a pandas DataFrame.

        The columns in the DataFrame are converted to arrays when possible, otherwise
        they will be skipped.

        Args:
            df: DataFrame.
            unsigned_to_long: If :obj:`True`, automatically converts
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
            file: HDF5 file.

        .. code-block:: python

            import h5py
            from torch_brain.data import ArrayDict

            data = ArrayDict(
                unit_id=["unit01", "unit02"],
                brain_region=["M1", "M1"],
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
            file: HDF5 file.

        .. note::
            This method will load all data in memory, if you would like to use lazy
            loading, call :meth:`LazyArrayDict.from_hdf5` instead.

        .. code-block:: python

            import h5py
            from torch_brain.data import ArrayDict

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

    def materialize(self) -> ArrayDict:
        r"""Materializes the data object, i.e., loads into memory all of the data that
        is still referenced in the HDF5 file."""
        for key in self.keys():
            # simply access all attributes to trigger the lazy loading
            getattr(self, key)

        return self


class LazyArrayDict(ArrayDict):
    r"""Lazy variant of :obj:`ArrayDict`. The data is not loaded until it is accessed.
    This class is meant to be used when the data is too large to fit in memory, and
    is intended to be intantiated via. :obj:`LazyArrayDict.from_hdf5`.

    .. note:: To access an attribute without triggering the in-memory loading use
        self.__dict__[key] otherwise using self.key or getattr(self, key) will trigger
        the lazy loading and will automatically convert the h5py dataset to a numpy
        array as well as apply any outstanding masks.
    """

    _lazy_ops: dict
    _unicode_keys: list[str]

    def __init__(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} cannot be constructed directly; use from_hdf5."
        )

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

        return super().__getattribute__(name)

    def select_by_mask(self, mask: np.ndarray):
        r"""Index all arrays with a boolean mask and return a copy.

        Lazy attributes will remain lazy, and masking will be applied
        to them upon access.

        Args:
            mask: Boolean array used for masking. The mask needs to be 1-dimensional,
                and of equal length as the object itself.
        """

        _validate_select_by_mask_input(mask, len(self))

        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                out.__dict__[key] = copy.deepcopy(value)
            elif isinstance(value, h5py.Dataset):
                # mask will be applied lazily on attribute access via _lazy_ops
                out.__dict__[key] = value
            elif isinstance(value, np.ndarray):
                out.__dict__[key] = value[mask].copy()
            else:
                raise RuntimeError(  # pragma: no cover
                    "Unknown state! Object has a non-private attribute that is neither "
                    "a np.ndarray, nor an h5py.Dataset"
                )

        # combine mask with any pre-existing lazy mask
        if "mask" not in out._lazy_ops:
            out._lazy_ops["mask"] = mask.copy()
        else:
            out._lazy_ops["mask"] = out._lazy_ops["mask"].copy()
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
            file: HDF5 file.

        .. code-block:: python

            import h5py
            from torch_brain.data import ArrayDict

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
