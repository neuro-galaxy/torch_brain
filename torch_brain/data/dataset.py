from __future__ import annotations
import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import copy
import numpy as np
import omegaconf


import h5py
import numpy as np
import torch

from temporaldata import Data, Interval
from torch_brain.transforms import TransformType


@dataclass
class DatasetIndex:
    r"""The dataset can be indexed by specifying a recording id and a start and end time."""

    recording_id: str
    start: float
    end: float


class Dataset(torch.utils.data.Dataset):
    r"""This class abstracts a collection of lazily-loaded Data objects. Each data object
    corresponds to a full recording. It is never fully loaded into memory, but rather
    lazy-loaded on-the-fly from disk.

    The dataset can be indexed by a recording id and a start and end time using the `get`
    method, or by a DatasetIndex object. This definition is a deviation from the standard
    PyTorch Dataset definition, which generally presents the dataset directly as samples.
    In this case, the Dataset by itself does not provide you with samples, but rather the
    means to flexibly work and access complete sessions.
    Within this framework, it is the job of the sampler to provide a list of
    DatasetIndex objects that are used to slice the dataset into samples (see
    `torch_brain.data.sampler`).

    The lazy loading is done both in:
    - time: only the requested time interval is loaded, without having to load the entire
      recording into memory, and
    - attributes: attributes are not loaded until they are requested, this is useful when
      only a small subset of the attributes are actually needed.

    References to the underlying hdf5 files will be opened, and will only be closed when
    the Dataset object is destroyed.

    Args:
        root: The root directory of the dataset.
        config: The configuration file specifying the sessions to include.
        brainset: The brainset to include. This is used to specify a single brainset,
            and can only be used if config is not provided.
        session: The session to include. This is used to specify a single session, and
            can only be used if config is not provided.
        split: The split of the dataset. This is used to determine the sampling intervals
            for each session. The split is optional, and is used to load a subset of the data
            in a session based on a predefined split.
        transform: A transform to apply to the data. This transform should be a callable
            that takes a Data object and returns a Data object.
        unit_id_prefix_fn: Optional[Callable[[Data], str]]
            Function to generate prefix strings for unit IDs to ensure uniqueness across the dataset.
            Takes a Data object as input and returns a string prefix.
            By default (when None), uses: "<brainset-id>/<session-id>/"
            Example:
                >>> unit_id_prefix_fn = lambda data: f"{data.brainset.id}/{data.session.id}/"
    """

    split: Optional[str] = None
    transform: Optional[TransformType] = None
    unit_id_prefix_fn: Optional[Callable[[Data], str]] = (None,)
    _check_for_data_leakage_flag: bool = True
    _open_files: Optional[Dict[str, h5py.File]] = None
    _data_objects: Optional[Dict[str, Data]] = None

    @classmethod
    def from_config(
        cls,
        root: str,
        config: str,
        split: Optional[str] = None,
        transform: Optional[TransformType] = None,
        unit_id_prefix_fn: Optional[Callable[[Data], str]] = None,
    ) -> Dataset:
        """Creates a Dataset from a configuration file.

        Args:
            root: The root directory of the dataset.
            config: The configuration file specifying the sessions to include.
        """
        # Parse config
        if isinstance(config, omegaconf.listconfig.ListConfig):
            config = omegaconf.OmegaConf.to_container(config)
        elif Path(config).is_file():
            config = omegaconf.OmegaConf.load(config)
        else:
            raise ValueError(f"Could not open configuration file: '{config}'")

        dataset = super().__new__(cls)
        dataset.root = root
        dataset.config = config
        dataset.split = split
        dataset.transform = transform
        dataset.unit_id_prefix_fn = unit_id_prefix_fn
        dataset.recording_dict = cls._look_for_files(root, config)
        dataset._open_files_and_data_objects()

        return dataset

    @classmethod
    def from_data_files(
        cls,
        filepath_dict: Dict[str, Path],
        split: Optional[str] = None,
        transform: Optional[TransformType] = None,
        unit_id_prefix_fn: Optional[Callable[[Data], str]] = None,
    ) -> Dataset:
        r"""Create a dataset from Data h5 files.

        Args:
            filepath_dict: A dictionary mapping recording IDs (strings) to file paths containing `temporaldata.Data` objects.
        """

        dataset = super().__new__(cls)
        dataset.split = split
        dataset.transform = transform
        dataset.unit_id_prefix_fn = unit_id_prefix_fn
        dataset.recording_dict = {
            recording_id: {"filename": filepath, "config": None}
            for recording_id, filepath in filepath_dict.items()
        }
        dataset._open_files_and_data_objects()

        return dataset

    @classmethod
    def from_data_objects(
        cls,
        data_dict: Dict[str, Data],
        split: Optional[str] = None,
        transform: Optional[TransformType] = None,
        unit_id_prefix_fn: Optional[Callable[[Data], str]] = None,
    ) -> Dataset:

        dataset = super().__new__(cls)
        dataset.split = split
        dataset.transform = transform
        dataset.unit_id_prefix_fn = unit_id_prefix_fn
        dataset._data_objects = data_dict
        dataset.recording_dict = {
            recording_id: {"filename": None, "config": None}
            for recording_id, _ in data_dict.items()
        }

        return dataset

    def _open_files_and_data_objects(self):
        self._open_files = {
            recording_id: h5py.File(recording_info["filename"], "r")
            for recording_id, recording_info in self.recording_dict.items()
        }

        self._data_objects = {
            recording_id: Data.from_hdf5(f, lazy=True)
            for recording_id, f in self._open_files.items()
        }

    def _close_open_files(self):
        """Closes the open files and deletes open data objects.
        This is useful when you are done with the dataset.
        """
        if self._open_files is not None:
            for f in self._open_files.values():
                f.close()
            self._open_files = None

        self._data_objects = None

    def __del__(self):
        self._close_open_files()

    @classmethod
    def _look_for_files(
        cls, root: str | Path, config: omegaconf.DictConfig
    ) -> Dict[str, Dict]:
        recording_dict = {}

        for i, selection_list in enumerate(config):
            selection = selection_list["selection"]

            # parse selection
            if len(selection) == 0:
                raise ValueError(
                    f"Selection {i} is empty. Please at least specify a brainset."
                )

            for subselection in selection:
                if subselection.get("brainset", "") == "":
                    raise ValueError(f"Please specify a brainset to include.")

                # Get a list of all the potentially chunks in this dataset.
                brainset_dir = Path(root) / subselection["brainset"]
                files = list(brainset_dir.glob("*.h5"))
                session_ids = sorted([f.stem for f in files])

                if len(session_ids) == 0:
                    raise ValueError(
                        f"No files found in {brainset_dir}. This is either a problem "
                        "with the provided path or the dataset has not been downloaded "
                        "and/or processed. For supported brainsets, please refer to "
                        "https://github.com/neuro-galaxy/brainsets"
                    )

                # Perform selection. Right now, we are limiting ourselves to session,
                # and subject filters, but we could make selection more flexible in the
                # future.
                sel_session = subselection.get("session", None)
                sel_sessions = subselection.get("sessions", None)
                sel_subject = subselection.get("subject", None)
                sel_subjects = subselection.get("subjects", None)
                # exclude_sessions allows you to exclude some sessions from the selection.
                # example use: you want to train on the complete brainset, but leave out
                # a few sessions for evaluating transfer performance.
                sel_exclude_sessions = subselection.get("exclude_sessions", None)

                # if subject is used for selection, we need to load all the files and
                # extract the subjects ids
                if sel_subject is not None or sel_subjects is not None:
                    all_session_subjects = []
                    for session_id in session_ids:
                        with h5py.File(brainset_dir / (session_id + ".h5"), "r") as f:
                            session_data = Data.from_hdf5(f, lazy=True)
                            all_session_subjects.append(session_data.subject.id)

                filtered = False
                if sel_session is not None:
                    assert (
                        sel_session in session_ids
                    ), f"Session {sel_session} not found in brainset {subselection['brainset']}"
                    session_ids = [sel_session]
                    filtered = True

                if sel_sessions is not None:
                    assert (
                        not filtered
                    ), "Cannot specify session AND sessions in selection"

                    # Check that all sortsets are in the brainset.
                    for session in sel_sessions:
                        assert (
                            session in session_ids
                        ), f"Session {session} not found in brainset {subselection['brainset']}"

                    session_ids = sorted(sel_sessions)
                    filtered = True

                if sel_subject is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subject AND session(s) in selection"

                    assert (
                        sel_subject in all_session_subjects
                    ), f"Could not find subject {sel_subject} in brainset {subselection['brainset']}"

                    session_ids = [
                        session
                        for i, session in enumerate(session_ids)
                        if all_session_subjects[i] == sel_subject
                    ]
                    filtered = True

                if sel_subjects is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subjects AND subject/session(s) in selection"

                    # Make sure all subjects asked for are in the brainset
                    sel_subjects = set(sel_subjects)
                    assert sel_subjects.issubset(all_session_subjects), (
                        f"Could not find subject(s) {sel_subjects - all_session_subjects} "
                        f" in brainset {subselection['brainset']}"
                    )

                    session_ids = [
                        session
                        for i, session in enumerate(session_ids)
                        if all_session_subjects[i] in sel_subjects
                    ]
                    filtered = True

                # Exclude sortsets if asked.
                if sel_exclude_sessions is not None:
                    session_ids = [
                        session
                        for session in session_ids
                        if session not in sel_exclude_sessions
                    ]

                assert (
                    len(session_ids) > 0
                ), f"No sessions left after filtering for selection {subselection['brainset']}"

                # Now we get the session-level information
                config = selection_list.get("config", {})

                for session_id in session_ids:
                    recording_id = subselection["brainset"] + "/" + session_id

                    if recording_id in recording_dict:
                        raise ValueError(
                            f"Recording {recording_id} is already included in the dataset."
                            "Please verify that it is only selected once."
                        )

                    recording_dict[recording_id] = dict(
                        filename=(Path(root) / (recording_id + ".h5")),
                        config=config,
                    )

        return recording_dict

    def get(self, recording_id: str, start: float, end: float):
        r"""This is the main method to extract a slice from a recording. It returns a
        Data object that contains all data for recording :obj:`recording_id` between
        times :obj:`start` and :obj:`end`.

        Args:
            recording_id: The recording id of the slice. This is usually
                <brainset_id>/<session_id>
            start: The start time of the slice.
            end: The end time of the slice.
        """
        data = copy.copy(self._data_objects[recording_id])
        # TODO: add more tests to make sure that slice does not modify the original data object
        # note there should be no issues as long as the self._data_objects stay lazy
        sample = data.slice(start, end)
        sample.units.id = self._get_unit_ids_with_prefix(sample)

        if self._check_for_data_leakage_flag and self.split is not None:
            sample._check_for_data_leakage(self.split)

        sample.recording_id = recording_id
        sample.config = self.recording_dict[recording_id]["config"]
        return sample

    def get_recording_data(self, recording_id: str):
        r"""Returns the data object corresponding to the recording :obj:`recording_id`.
        If the split is not :obj:`None`, the data object is sliced to the allowed sampling
        intervals for the split, to avoid any data leakage. :obj:`RegularTimeSeries`
        objects are converted to :obj:`IrregularTimeSeries` objects, since they are
        most likely no longer contiguous.

        .. warning::
            This method might load the full data object in memory, avoid multiple calls
            to this method if possible.
        """
        data = copy.copy(self._data_objects[recording_id])

        # get allowed sampling intervals
        if self.split is not None:
            sampling_intervals = self.get_sampling_intervals()[recording_id]
            data = data.select_by_interval(sampling_intervals)
            if self._check_for_data_leakage_flag:
                data._check_for_data_leakage(self.split)
        else:
            data = copy.deepcopy(data)

        data.units.id = self._get_unit_ids_with_prefix(data)
        return data

    def get_sampling_intervals(self):
        r"""Returns a dictionary of sampling intervals for each session.
        This represents the intervals that can be sampled from each session.

        Note that these intervals will change depending on the split. If no split is
        provided, the full domain of the data is used.
        """
        sampling_intervals_dict = {}
        for recording_id in self.recording_dict.keys():
            sampling_domain = (
                f"{self.split}_domain" if self.split is not None else "domain"
            )
            sampling_intervals = getattr(
                self._data_objects[recording_id], sampling_domain
            )
            sampling_intervals_modifier_code = self.recording_dict[recording_id][
                "config"
            ].get("sampling_intervals_modifier", None)
            if sampling_intervals_modifier_code is not None:
                local_vars = {
                    "data": copy.deepcopy(self._data_objects[recording_id]),
                    "sampling_intervals": sampling_intervals,
                    "split": self.split,
                }
                try:
                    exec(sampling_intervals_modifier_code, {}, local_vars)
                except NameError as e:
                    error_message = (
                        f"{e}. Variables that are passed to the sampling_intervals_modifier "
                        f"are: {list(local_vars.keys())}"
                    )
                    raise NameError(error_message) from e
                except Exception as e:
                    error_message = (
                        f"Error while executing sampling_intervals_modifier defined in "
                        f"the config file for session {recording_id}: {e}"
                    )
                    raise type(e)(error_message) from e

                sampling_intervals = local_vars.get("sampling_intervals")
            sampling_intervals_dict[recording_id] = sampling_intervals
        return sampling_intervals_dict

    def get_recording_config_dict(self):
        r"""Returns configs for each session in the dataset as a dictionary."""
        ans = {}
        for recording_id in self.recording_dict.keys():
            ans[recording_id] = self.recording_dict[recording_id]["config"]
        return ans

    def get_recording_ids(self):
        r"""Returns the session ids of the dataset."""
        return sorted(list(self.recording_dict.keys()))

    def _get_unit_ids_with_prefix(self, data: Data) -> np.ndarray:
        r"""Add prefix string to data.units.id and return a new numpy string array.

        If `unit_id_prefix_fn` is set (not None), then this function is used
        to decide the prefix string. Otherwise, the default value is:
        `<data.brainset.id>/<data.session.id>/`
        """
        if self.unit_id_prefix_fn is not None:
            prefix_str = self.unit_id_prefix_fn(data)
        else:
            prefix_str = f"{data.brainset.id}/{data.session.id}/"

        unit_ids = data.units.id
        return np.core.defchararray.add(prefix_str, unit_ids.astype(str))

    def get_unit_ids(self):
        r"""Returns all unit ids in the dataset."""
        unit_ids_list = []
        for recording_id in self.recording_dict.keys():
            data = self._data_objects[recording_id]
            unit_ids = self._get_unit_ids_with_prefix(data)
            unit_ids_list.extend(unit_ids)
        return unit_ids_list

    def get_subject_ids(self):
        r"""Returns all subject ids in the dataset."""
        subject_ids = []
        for recording_id in self.recording_dict.keys():
            data = self._data_objects[recording_id]
            subject_ids.append(f"{data.brainset.id}/{data.subject.id}")
        return sorted(list(set(subject_ids)))

    def disable_data_leakage_check(self):
        r"""Disables the data leakage check.

        .. warning::
            Only do this you are absolutely sure that there is no leakage between the
            current split and other splits (eg. the test split).
        """
        self._check_for_data_leakage_flag = False
        logging.warn(
            f"Data leakage check is disabled. Please be absolutely sure that there is "
            f"no leakage between {self.split} and other splits."
        )

    def __getitem__(self, index: DatasetIndex):
        sample = self.get(index.recording_id, index.start, index.end)

        # apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        raise NotImplementedError("Length of dataset is not defined")

    def __iter__(self):
        raise NotImplementedError("Iteration over dataset is not defined")

    def __repr__(self):
        return f"Dataset(root={self.root}, config={self.config}, split={self.split})"
