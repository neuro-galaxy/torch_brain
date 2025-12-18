from .dataset import Dataset, DatasetIndex, _ensure_index_has_namespace
from typing import Optional
import numpy as np
from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix
from temporaldata import Data, Interval

from .mixins import SpikingDatasetMixin


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


class NestedSpikingDataset(SpikingDatasetMixin, NestedDataset): ...


def _is_dict_like(obj) -> bool:
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _namespace_join(a: str, b: str) -> str:
    return a + "/" + b if a else b
