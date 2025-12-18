import copy
from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
import h5py
import numpy as np
import torch

from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix, set_nested_attribute_in_data
from temporaldata import Data, Interval


@dataclass
class DatasetIndex:
    r"""Index for accessing a specific time interval of a recording within a :class:`Dataset`.

    Args:
        recording_id: The unique identifier for the recording to access.
        start: Start time of the interval (in seconds or appropriate time units).
        end: End time of the interval (in seconds or appropriate time units).
        _namespace: Optional namespace prefix for attribute namespacing. Used internally by
            :class:`torch_brain.dataset.NestedDataset` to handle nested namespaced attributes.
    """

    recording_id: str
    start: float
    end: float
    _namespace: str = ""


class Dataset(torch.utils.data.Dataset):
    r"""PyTorch Dataset for loading time-slices of neural data recordings from HDF5 files.

    The dataset can be indexed by a :class:`DatasetIndex` object, which contains a
    recording id and a start and end times.

    This definition is a deviation from the standard PyTorch Dataset definition,
    In this case, the Dataset by itself does not provide you with samples, but rather the
    means to flexibly work and access complete recordings.
    Within this framework, it is the job of the sampler to provide the indices that
    are used to slice the dataset into samples (see Samplers).

    The lazy loading is done both in:
        - time: only the requested time interval is loaded, without having to load the entire
          recording into memory, and
        - attributes: attributes are not loaded until they are requested, this is useful when
          only a small subset of the attributes are actually needed.

    Args:
        dataset_dir: Path to the directory containing HDF5 recording files.
        recording_ids: Optional list of recording IDs to include. These correspond to the
            filenames of the HDF5 files in the dataset directory. If ``None``, all
            ``*.h5`` files in the dataset directory will be used.
        transform: Optional transform to apply to each data sample.
        keep_files_open: If ``True``, keeps HDF5 files open in memory for faster
            access. If ``False``, files are opened on-demand. Default is ``True``.
        namespace_attributes: List of nested attribute paths (e.g., "session.id")
            that should be namespaced when loading recordings in a :class:`torch_brain.dataset.NestedDataset`
            situation. Defaults to ``["session.id", "subject.id"]``.
            Example: With the deafult value, if you create a nested dataset with two datasets,
            ``ds1`` and ``ds2``, and you load a recording from ``ds1``, the recording's
            ``session.id`` and ``subject.id`` attributes will be prefixed with ``ds1/``.
    """

    def __init__(
        self,
        dataset_dir: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        keep_files_open: bool = True,
        namespace_attributes: list[str] = ["session.id", "subject.id"],
    ):

        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)

        if recording_ids is None:
            recording_ids = [x.stem for x in dataset_dir.glob("*.h5")]
            if len(recording_ids) == 0:
                raise ValueError(f"No recordings found at {str(dataset_dir)}")
        self._recording_ids = np.sort(np.array(recording_ids))

        self._filepaths = {r: dataset_dir / f"{r}.h5" for r in self._recording_ids}
        missing_files = [str(p) for p in self._filepaths.values() if not p.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"The following recording files do not exist: {missing_files}"
            )

        if keep_files_open:
            self._data_objects = {
                r: Data.from_hdf5(h5py.File(self._filepaths[r]))
                for r in self._recording_ids
            }

        self.transform = transform
        self.namespace_attributes = namespace_attributes

    @property
    def recording_ids(self) -> list[str]:
        """List of recording IDs in the dataset.

        Returns:
            Sorted list of recording ID strings.
        """
        return self._recording_ids.tolist()

    def get_recording(self, recording_id: str, _namespace: str = "") -> Data:
        """Get lazy-loaded :class:`temporaldata.Data` object for a recording.

        Args:
            recording_id: The ID of the recording to load (same as from :meth:`recording_ids`).
            _namespace: Optional namespace prefix to apply to attributes.

        Returns:
            Lazy :class:`temporaldata.Data` object containing the full recording.

        Raises:
            ValueError: If the ``recording_id`` is not found in the dataset.
        """
        if hasattr(self, "_data_objects"):
            data = copy.deepcopy(self._data_objects[recording_id])
        else:
            file = h5py.File(self._filepaths[recording_id], "r")
            data = Data.from_hdf5(file, lazy=True)

        self.get_recording_hook(data)
        if _namespace:
            self.apply_namespace(data, _namespace + "/")
        return data

    def __getitem__(self, index: DatasetIndex) -> Data:
        """Get a time-sliced sample from the dataset.

        If a transform was provided during construction, it will be applied to the sliced sample
        before returning.

        Args:
            index: Container for the recording ID and time interval.

        Returns:
            :class:`temporaldata.Data` object containing the sliced time interval, optionally transformed.
        """
        index = _ensure_index_has_namespace(index)

        data = self.get_recording(index.recording_id, index._namespace)
        sample = data.slice(index.start, index.end)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(self, *args, **kwargs) -> dict[str, Interval]:
        """Get the time domain intervals for all recordings in the dataset.

        Returns:
            Dictionary mapping recording IDs to their time domain intervals.
        """
        return {rid: self.get_recording(rid).domain for rid in self._recording_ids}

    def apply_namespace(self, data: Data, namespace: str) -> Data:
        """Apply a namespace prefix to specified nested attributes in the data.

        This method modifies the data object in-place by prepending the namespace
        to string attributes or string arrays specified in :attr:`namespace_attributes`.

        Can be overridden by subclasses to apply the namespace in a custom way.

        Args:
            data: The Data object to modify.
            namespace: The namespace prefix to prepend (e.g., "experiment1/").

        Returns:
            The modified :class:`temporaldata.Data` object (same instance, modified in-place).
        """
        for attrib in self.namespace_attributes:
            value = data.get_nested_attribute(attrib)
            if isinstance(value, str):
                value = namespace + value
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                value = np_string_prefix(namespace, value.astype(str))
            else:
                raise TypeError(
                    f"Attribute '{attrib}' is of unsupported type: {type(value)}. "
                    "Expected str or np.ndarray of shape (N,)."
                )
            set_nested_attribute_in_data(data, attrib, value)

        return data

    def get_recording_hook(self, data: Data) -> None:
        """Hook method called after loading a recording in :meth:`get_recording`.

        Subclasses can override this method to perform custom processing
        on recordings after they are loaded but before they are returned.

        Args:
            data: The Data object that was just loaded.
        """
        pass

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        n_rec = len(self._recording_ids)
        attrs = []
        if self.transform is not None:
            attrs.append(f"transform={self.transform}")
        return f"{cls}(n_recordings={n_rec}{', ' if attrs else ''}{', '.join(attrs)})"


def _ensure_index_has_namespace(index: DatasetIndex) -> DatasetIndex:
    r"""Ensure a DatasetIndex has a _namespace attribute for backwards compatibility.
    This is a temporary solution and should be deprecated when older version of Dataset
    is no longer supported."""
    if not hasattr(index, "_namespace"):
        index._namespace = ""
    return index
