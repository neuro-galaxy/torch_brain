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
        self._dataset_dir = dataset_dir

        if recording_ids is None:
            recording_ids = [x.stem for x in dataset_dir.glob("*.h5")]
        self._recording_ids = np.sort(np.array(recording_ids))

        if keep_files_open:
            self._filepaths = [dataset_dir / f"{rid}.h5" for rid in self._recording_ids]
            self._data_objects = [Data.from_hdf5(h5py.File(x)) for x in self._filepaths]

        self.transform = transform

        self._namespace = ""
        self.namespace_attributes = namespace_attributes

    @property
    def recording_ids(self) -> list[str]:
        return self._recording_ids.tolist()

    def get_recording(self, recording_id: str) -> Data:
        idx = np.searchsorted(self._recording_ids, recording_id)
        if self._recording_ids[idx] != recording_id:
            raise ValueError(f"Recording id '{recording_id}' not found in dataset.")

        if hasattr(self, "_data_objects"):
            data = copy.deepcopy(self._data_objects[idx])
        else:
            file = h5py.File(self._filepaths[idx], "r")
            data = Data.from_hdf5(file, lazy=True)

        self.get_recording_hook(data)
        self.apply_namespace(self._namespace, data)
        return data

    def get_slice(self, recording_id: str, start: float, end: float) -> Data:
        data = self.get_recording(recording_id)
        sample = data.slice(start, end)
        self.get_slice_hook(sample)
        return sample

    def __getitem__(self, index: DatasetIndex) -> Data:
        sample = self.get_slice(index.recording_id, index.start, index.end)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def set_namespace(self, name: str):
        self._namespace = name + "/" + self._namespace
        self._recording_ids = np_string_prefix(name + "/", self._recording_ids)

    def apply_namespace(self, namespace: str, data: Data) -> Data:
        if not namespace:
            return data

        for attrib in self.namespace_attributes:
            value = data.get_nested_attribute(attrib)
            if isinstance(value, str):
                value = namespace + value
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                value = np_string_prefix(namespace, value.astype(str))
            else:
                raise TypeError(
                    f"Attribute '{attrib}' is of unsupported type: {type(value)}. "
                    "Expected str or np.ndarray."
                )
            set_nested_attribute(data, attrib, value)

        return data

    def get_recording_hook(self, data: Data) -> None:
        pass

    def get_slice_hook(self, data_slice: Data) -> None:
        pass

    def get_sampling_intervals(self, *args, **kwargs) -> dict[str, Interval]:
        return {rid: self.get_recording(rid).domain for rid in self._recording_ids}

    def get_subject_ids(self) -> list[str]:
        ids = [self.get_recording(rid).subject.id for rid in self._recording_ids]
        return np.sort(np.unique(ids)).tolist()


class MultiDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset] | dict[str, Dataset],
        transform: Optional[TransformType] = None,
    ):

        if _is_dict_like(datasets):
            # To support Python dicts, OmegaConf.DictConfig, etc.
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
        for name, dataset in self._datasets.items():
            dataset.set_namespace(name)

        rec_ids = sum([ds.recording_ids for ds in self.datasets.values()], [])
        self._recording_ids = np.sort(rec_ids)

        self.transform = transform

    @property
    def datasets(self) -> dict[str, Dataset]:
        return self._datasets

    def get_recording(self, recording_id: str) -> Data:
        dataset_name, _ = recording_id.split("/", 1)
        data = self.datasets[dataset_name].get_recording(recording_id)
        self.get_recording_hook(data)
        return data

    def __getitem__(self, index: DatasetIndex) -> Data:
        dataset_name, _ = index.recording_id.split("/", 1)
        sample = self.datasets[dataset_name][index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SpikingDatasetMixin:
    def get_unit_ids(self) -> list[str]:
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans)).tolist()


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
