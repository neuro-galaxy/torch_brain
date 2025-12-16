import copy
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import h5py
import numpy as np
import torch

from torch_brain.transforms import TransformType
from temporaldata import Data, Interval


@dataclass
class Timeslice:
    r"""The dataset can be indexed by specifying a recording id and a start and end time."""

    recording_id: str
    start: float
    end: float


class TemporalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[TransformType] = None,
        keep_files_open: bool = True,
        autoprefix_ids: bool = True,
    ):

        if not isinstance(dataset_dir, Path):
            dataset_dir = Path(dataset_dir)
        self._dataset_dir = dataset_dir

        if recording_ids is None:
            recording_ids = [x.stem for x in dataset_dir.glob("*.h5")]
        self._recording_ids = np.sort(np.array(recording_ids))

        self.autoprefix_ids = autoprefix_ids

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

        if self.autoprefix_ids:
            data.subject.id = f"{data.brainset.id}/{data.subject.id}"

        self.get_recording_hook(data)
        return data

    def get_slice(self, recording_id: str, start: float, end: float) -> Data:
        data = self.get_recording(recording_id)
        sample = data.slice(start, end)
        self.get_slice_hook(sample)
        return sample

    def __getitem__(self, index: Timeslice) -> Data:
        sample = self.get_slice(index.recording_id, index.start, index.end)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_recording_hook(self, data: Data) -> None:
        pass

    def get_slice_hook(self, data_slice: Data) -> None:
        pass

    def get_sampling_intervals(self) -> dict[str, Interval]:
        return {rid: self.get_recording(rid).domain for rid in self.recording_ids}


class SpikingDatasetMixin:
    def get_unit_ids(self):
        return np.sort(
            np.concatenate(
                [self.get_recording(rid).units.id for rid in self.recording_ids]
            )
        )

    def get_recording_hook(self, data: Data):
        if self.autoprefix_ids:
            unit_prefix_str = f"{data.brainset.id}/{data.session.id}/"
            if np.__version__ >= "2.0":
                data.units.id = np.strings.add(
                    unit_prefix_str,
                    data.units.id.astype(str),
                )
            else:
                data.units.id = np.core.defchararray.add(
                    unit_prefix_str,
                    data.units.id.astype(str),
                )
