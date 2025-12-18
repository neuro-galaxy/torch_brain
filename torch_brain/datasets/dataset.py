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


class NestedDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset] | dict[str, Dataset],
        transform: Optional[TransformType] = None,
    ):

        if _is_dict_like(datasets):
            # this check supports Python dicts, OmegaConf.DictConfig, etc.
            # Why do we want to support OmegaConf.DictConfig? It is received when using
            # hydra.utils.instantiate() to instantiate the NestedDataset
            dataset_dict = datasets
        elif isinstance(datasets, (list, tuple)):
            dataset_names = [x.__class__.__name__ for x in datasets]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    "Duplicate dataset class names found in provided datasets."
                    " Please use a dictionary instead to specify dataset names explicitly."
                )
            dataset_dict = {name: x for name, x in zip(dataset_names, datasets)}
        else:
            raise TypeError(
                f"datasets must be a list/tuple or a dict-like object"
                f" (got {type(datasets)})"
            )

        self._datasets = dataset_dict
        rec_ids = []
        for name, dataset in self._datasets.items():
            rec_ids.extend(np_string_prefix(name + "/", dataset.recording_ids))
        self._recording_ids = np.sort(rec_ids)

        self.transform = transform

    @property
    def datasets(self) -> dict[str, Dataset]:
        return self._datasets

    def get_recording(self, recording_id: str, _namespace: str = "") -> Data:
        if "/" not in recording_id:
            raise ValueError(
                f"recording_id '{recording_id}' missing dataset prefix. "
                f"Expected format: 'dataset_name/recording_id'."
            )
        dataset_name, recording_id = recording_id.split("/", 1)
        _namespace = _namespace_join(_namespace, dataset_name)
        data = self.datasets[dataset_name].get_recording(recording_id, _namespace)
        self.get_recording_hook(data)
        return data

    def __getitem__(self, index: DatasetIndex) -> Data:
        index = _ensure_index_has_namespace(index)

        if "/" not in index.recording_id:
            raise ValueError(
                f"recording_id '{index.recording_id}' missing dataset prefix. "
                f"Expected format: 'dataset_name/recording_id'."
            )
        dataset_name, recording_id = index.recording_id.split("/", 1)
        new_index = DatasetIndex(
            recording_id=recording_id,
            start=index.start,
            end=index.end,
            _namespace=_namespace_join(index._namespace, dataset_name),
        )
        sample = self.datasets[dataset_name][new_index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(self, *args, **kwargs) -> dict[str, Interval]:
        ans = {}
        for dataset_name, dataset in self.datasets.items():
            samp_intervals = dataset.get_sampling_intervals(*args, **kwargs)
            for rid, interval in samp_intervals.items():
                ans[dataset_name + "/" + rid] = interval
        return ans


class SpikingDatasetMixin:
    def get_unit_ids(self) -> list[str]:
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()


class NestedSpikingDataset(SpikingDatasetMixin, NestedDataset): ...


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


def _is_dict_like(obj) -> bool:
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _namespace_join(a: str, b: str) -> str:
    return a + "/" + b if a else b


def _ensure_index_has_namespace(index: DatasetIndex) -> DatasetIndex:
    r"""Older DatasetIndex objects did not have a namespace attribute,
    so we add it if it is not present to support backwards compatibility."""
    if not hasattr(index, "_namespace"):
        index._namespace = ""
    return index
