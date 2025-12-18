import copy
from dataclasses import dataclass
from typing import Optional, Any
from pathlib import Path
import h5py
import numpy as np
import torch

from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix
from temporaldata import Data, Interval


@dataclass
class DatasetIndex:
    r"""The dataset can be indexed by specifying a recording id and a start and end time."""

    recording_id: str
    start: float
    end: float
    _namespace: str = ""


class Dataset(torch.utils.data.Dataset):
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
        self._recording_ids = np.sort(np.array(recording_ids))

        self._filepaths = [dataset_dir / f"{rid}.h5" for rid in self._recording_ids]
        if keep_files_open:
            self._data_objects = [Data.from_hdf5(h5py.File(x)) for x in self._filepaths]

        self.transform = transform
        self.namespace_attributes = namespace_attributes

    @property
    def recording_ids(self) -> list[str]:
        return self._recording_ids.tolist()

    def get_recording(self, recording_id: str, _namespace: str = "") -> Data:
        idx = np.searchsorted(self._recording_ids, recording_id)
        if self._recording_ids[idx] != recording_id:
            raise ValueError(f"Recording id '{recording_id}' not found in dataset.")

        if hasattr(self, "_data_objects"):
            data = copy.deepcopy(self._data_objects[idx])
        else:
            file = h5py.File(self._filepaths[idx], "r")
            data = Data.from_hdf5(file, lazy=True)

        self.get_recording_hook(data)
        if _namespace:
            self.apply_namespace(data, _namespace + "/")
        return data

    def __getitem__(self, index: DatasetIndex) -> Data:
        index = _ensure_index_has_namespace(index)

        data = self.get_recording(index.recording_id, index._namespace)
        sample = data.slice(index.start, index.end)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(self, *args, **kwargs) -> dict[str, Interval]:
        return {rid: self.get_recording(rid).domain for rid in self._recording_ids}

    def apply_namespace(self, data: Data, namespace: str) -> Data:
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
            set_nested_attribute(data, attrib, value)

        return data

    def get_recording_hook(self, data: Data) -> None:
        pass

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        n_rec = len(self._recording_ids)
        attrs = []
        if self.transform is not None:
            attrs.append("transform=...")
        return f"<{cls}(n_recordings={n_rec}{', ' if attrs else ''}{', '.join(attrs)})>"


def set_nested_attribute(data: Data, path: str, value: Any) -> Data:
    # Split key by dots, resolve using getattr
    components = path.split(".")
    obj = data
    for c in components[:-1]:
        try:
            obj = getattr(obj, c)
        except AttributeError:
            raise AttributeError(
                f"Could not resolve {path} in data (specifically, at level {c}))"
            )

    setattr(obj, components[-1], value)
    return data


def _ensure_index_has_namespace(index: DatasetIndex) -> DatasetIndex:
    r"""Older DatasetIndex objects did not have a namespace attribute,
    so we add it if it is not present to support backwards compatibility."""
    if not hasattr(index, "_namespace"):
        index._namespace = ""
    return index
