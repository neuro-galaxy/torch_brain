import copy
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import h5py
import numpy as np
import torch

from torch_brain.transforms import TransformType
from torch_brain.utils import numpy_string_prefix
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
    ):

        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)
        self._dataset_dir = dataset_dir

        if recording_ids is None:
            recording_ids = [x.stem for x in dataset_dir.glob("*.h5")]
        self._recording_ids = np.sort(np.array(recording_ids))

        self.id_namespace = ""

        if keep_files_open:
            self._data_objects = {
                rid: Data.from_hdf5(h5py.File(dataset_dir / f"{rid}.h5"), lazy=True)
                for rid in recording_ids
            }

        self.transform = transform

    @property
    def recording_ids(self) -> np.ndarray:
        return self._recording_ids

    def get_recording(self, recording_id: str) -> Data:
        if hasattr(self, "_data_objects"):
            data = copy.deepcopy(self._data_objects[recording_id])
        else:
            file = h5py.File(self._dataset_dir / recording_id, "r")
            data = Data.from_hdf5(file, lazy=True)

        self.prefix_session_id(data)
        self.prefix_subject_id(data)
        self.get_recording_hook(data)
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

    def prefix_session_id(self, data: Data):
        if (
            hasattr(data, "session")
            and hasattr(data.session, "id")
            and isinstance(data.session.id, str)
        ):
            data.session.id = f"{self.id_namespace}{data.session.id}"

    def prefix_subject_id(self, data: Data):
        if (
            hasattr(data, "subject")
            and hasattr(data.subject, "id")
            and isinstance(data.subject.id, str)
        ):
            data.subject.id = f"{self.id_namespace}{data.subject.id}"

    def get_recording_hook(self, data: Data) -> None:
        pass

    def get_slice_hook(self, data_slice: Data) -> None:
        pass

    def get_sampling_intervals(self) -> dict[str, Interval]:
        return {rid: self.get_recording(rid).domain for rid in self.recording_ids}

    def get_subject_ids(self) -> np.ndarray:
        ids = [self.get_recording(rid).subject.id for rid in self.recording_ids]
        return np.sort(np.unique(ids))


class MultiDataset(Dataset):
    def __init__(
        self,
        datasets: list[Dataset] | dict[str, Dataset],
        transform: Optional[TransformType] = None,
    ):
        if isinstance(datasets, dict):
            dataset_dict = datasets
        else:
            dataset_names = [x.__class__.__name__ for x in datasets]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    "Duplicate dataset class names found in provided datasets."
                    " Please use a dictionary instead to specify dataset names explicitly."
                )
            dataset_dict = {name: x for name, x in zip(dataset_names, datasets)}

        self._datasets = dataset_dict
        for name, dataset in self._datasets.items():
            dataset.id_namespace = f"{name}/"

        self.transform = transform

        rec_ids = []
        for name, dataset in self.datasets.items():
            rec_ids.append(numpy_string_prefix(f"{name}/", dataset.recording_ids))
        self._recording_ids = np.sort(np.concat(rec_ids))

    @property
    def datasets(self) -> dict[str, Dataset]:
        return self._datasets

    def get_recording(self, recording_id: str) -> Data:
        dataset_name, recording_id = recording_id.split("/", 1)
        return self.datasets[dataset_name].get_recording(recording_id)


class SpikingDatasetMixin:
    def get_recording_hook(self, data: Data):
        if len(self.id_namespace) > 0:
            # All spiking datasets have units, which need to be prefixed appropriately
            data.units.id = numpy_string_prefix(
                self.id_namespace,
                data.units.id.astype(str),
            )

    def get_unit_ids(self):
        ans = [self.get_recording(rid).units.id for rid in self.recording_ids]
        return np.sort(np.concatenate(ans))
