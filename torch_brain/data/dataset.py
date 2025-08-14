import copy
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import h5py
import numpy as np
import omegaconf
import torch
from temporaldata import Data, Interval
from tqdm import tqdm


@dataclass
class DatasetIndex:
    r"""The dataset can be indexed by specifying a recording id and a start and end time."""

    recording_id: str
    start: float
    end: float


default_session_id_prefix_fn = lambda data: f"{data.brainset.id}/"
default_unit_id_prefix_fn = lambda data: f"{data.brainset.id}/{data.session.id}/"
default_subject_id_prefix_fn = lambda data: f"{data.brainset.id}/"


class Dataset(torch.utils.data.Dataset):
    r"""This class abstracts a collection of lazily-loaded Data objects. Each data object
    corresponds to a full recording. It is never fully loaded into memory, but rather
    lazy-loaded on-the-fly from disk.

    The dataset can be indexed by a recording id and a start and end times using the `get`
    method. This definition is a deviation from the standard PyTorch Dataset definition,
    which generally presents the dataset directly as samples.
    In this case, the Dataset by itself does not provide you with samples, but rather the
    means to flexibly work and access complete recordings.

    Within this framework, it is the job of the sampler to provide a list of
    :class:`DatasetIndex` objects that are used to slice the dataset into samples (see
    Samplers).

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
        unit_id_prefix_fn:
            A function to generate prefix strings for unit IDs to ensure uniqueness across
            the dataset. It takes a Data object as input and returns a string that would be
            prefixed to all unit ids in that Data object.
            Default corresponds to the function `lambda data: f"{data.brainset.id}/{data.session.id}/"`
        session_id_prefix_fn: Same as unit_id_prefix_fn but for session ids.
            Default corresponds to the function `lambda data: f"{data.brainset.id}/"`
        subject_id_prefix_fn: Same as unit_id_prefix_fn but for subject ids.
            Default corresponds to the function `lambda data: f"{data.brainset.id}/"`
        keep_files_open: Whether to keep the hdf5 files open. This only keeps references
            to the files, and does not load the data into memory.
    """

    _check_for_data_leakage_flag: bool = True
    _open_files: Optional[Dict[str, h5py.File]] = None
    _data_objects: Optional[Dict[str, Data]] = None

    def __init__(
        self,
        root: str,
        *,
        config: Optional[str] = None,
        recording_id: Optional[str] = None,
        split: Optional[str] = None,
        transform: Optional[Callable[[Data], Any]] = None,
        unit_id_prefix_fn: Callable[[Data], str] = default_unit_id_prefix_fn,
        session_id_prefix_fn: Callable[[Data], str] = default_session_id_prefix_fn,
        subject_id_prefix_fn: Callable[[Data], str] = default_subject_id_prefix_fn,
        keep_files_open: bool = True,
    ):
        super().__init__()
        self.root = root
        self.config = config
        self.split = split
        self.transform = transform
        self.unit_id_prefix_fn = unit_id_prefix_fn
        self.session_id_prefix_fn = session_id_prefix_fn
        self.subject_id_prefix_fn = subject_id_prefix_fn
        self.keep_files_open = keep_files_open

        if config is not None:
            assert (
                recording_id is None
            ), "Cannot specify recording_id when using config."

            if isinstance(config, omegaconf.listconfig.ListConfig):
                config = omegaconf.OmegaConf.to_container(config)
            elif Path(config).is_file():
                config = omegaconf.OmegaConf.load(config)
            else:
                raise ValueError(f"Could not open configuration file: '{config}'")

            self.recording_dict = self._look_for_files(config)

        elif recording_id is not None:
            self.recording_dict = {
                recording_id: {
                    "filename": Path(self.root) / (recording_id + ".h5"),
                    "config": {},
                }
            }
        else:
            raise ValueError("Please either specify a config file or a recording_id.")

        if self.keep_files_open:
            # open files lazily
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

    def _look_for_files(self, config: omegaconf.DictConfig) -> Dict[str, Dict]:
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
                brainset_dir = Path(self.root) / subselection["brainset"]
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
                        filename=(Path(self.root) / (recording_id + ".h5")),
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
        data = self._get_data_object(recording_id)
        sample = data.slice(start, end)

        if self._check_for_data_leakage_flag and self.split is not None:
            sample._check_for_data_leakage(self.split)

        self._update_data_with_prefixed_ids(sample)
        sample.config = self.recording_dict[recording_id]["config"]

        return sample

    def _get_data_object(self, recording_id: str):
        if self.keep_files_open:
            return copy.copy(self._data_objects[recording_id])
        else:
            file = h5py.File(self.recording_dict[recording_id]["filename"], "r")
            return Data.from_hdf5(file, lazy=True)

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
        data = self._get_data_object(recording_id)

        # get allowed sampling intervals
        if self.split is not None:
            sampling_intervals = self.get_sampling_intervals()[recording_id]
            data = data.select_by_interval(sampling_intervals)
            if self._check_for_data_leakage_flag:
                data._check_for_data_leakage(self.split)
        else:
            data = copy.deepcopy(data)

        self._update_data_with_prefixed_ids(data)
        return data

    def get_sampling_intervals(self):
        r"""Returns a dictionary of sampling intervals for each session.
        This represents the intervals that can be sampled from each session.

        Note that these intervals will change depending on the split. If no split is
        provided, the full domain of the data is used.
        """
        sampling_intervals = self.get_all_ids()['sampling_intervals']

        sampling_intervals = {key: Interval(np.array(start), np.array(end)) for key, (start, end) in sampling_intervals.items()}
        
        return sampling_intervals
        sampling_intervals_dict = {}
        for recording_id in tqdm(self.recording_dict.keys(), desc="Getting sampling intervals"):
            sampling_domain = (
                f"{self.split}_domain" if self.split is not None else "domain"
            )
            sampling_intervals = getattr(
                self._get_data_object(recording_id), sampling_domain
            )
            sampling_intervals_modifier_code = self.recording_dict[recording_id][
                "config"
            ].get("sampling_intervals_modifier", None)
            if sampling_intervals_modifier_code is not None:
                local_vars = {
                    "data": self._get_data_object(recording_id),
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

    def _get_unit_ids_with_prefix(self, data: Data) -> np.ndarray:
        r"""Return unit ids with prefix applied"""
        prefix_str = self.unit_id_prefix_fn(data)
        return np.core.defchararray.add(prefix_str, data.units.id.astype(str))

    def _get_session_id_with_prefix(self, data: Data) -> str:
        r"""Return session id with prefix applied"""
        return f"{self.session_id_prefix_fn(data)}{data.session.id}"

    def _get_subject_id_with_prefix(self, data: Data) -> str:
        r"""Return subject id with prefix applied"""
        return f"{self.subject_id_prefix_fn(data)}{data.subject.id}"

    def _update_data_with_prefixed_ids(self, data: Data):
        r"""Inplace add prefixes to unit ids, session id, and subect id"""
        if hasattr(data, "units"):
            data.units.id = self._get_unit_ids_with_prefix(data)

        if hasattr(data, "session"):
            data.session.id = self._get_session_id_with_prefix(data)

        if hasattr(data, "subject"):
            data.subject.id = self._get_subject_id_with_prefix(data)

    def _get_cache_directory(self):
        r"""Returns the cache directory path for this dataset."""
        cache_dir = Path(self.root) / ".torch_brain_cache" / "dataset_ids"
        print(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_dataset_hash(self):
        r"""Generate a unique hash for this dataset configuration to use as cache key."""
        # Create a hash based on dataset configuration that would affect IDs
        hash_data = {
            'root': str(self.root),
            'config': self.config if isinstance(self.config, (str, type(None))) else str(self.config),
            'split': self.split,
            # 'recording_dict_keys': sorted(self.recording_dict.keys()) if self.recording_dict else [],
            # # Include modification times of recording files to detect changes
            # 'file_mtimes': {
            #     recording_id: os.path.getmtime(info['filename'])
            #     for recording_id, info in self.recording_dict.items()
            #     if os.path.exists(info['filename'])
            # } if self.recording_dict else {}
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def _get_cache_file_path(self):
        r"""Returns the full path to the cache file for this dataset."""
        dataset_hash = self._get_dataset_hash()
        cache_dir = self._get_cache_directory()
        return cache_dir / f"ids_cache_{dataset_hash}.json"

    def _load_ids_from_disk_cache(self):
        r"""Load cached IDs from disk if available and valid."""
        cache_file = self._get_cache_file_path()
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            # Convert lists back to numpy arrays where needed
            if 'unit_ids' in cached_data:
                cached_data['unit_ids'] = [str(uid) for uid in cached_data['unit_ids']]
                
            return cached_data
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logging.warning(f"Failed to load IDs cache from {cache_file}: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except OSError:
                pass
            return None

    def _save_ids_to_disk_cache(self, ids_data):
        r"""Save IDs data to disk cache."""
        cache_file = self._get_cache_file_path()
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, value in ids_data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                elif isinstance(value, list):
                    # Convert any numpy elements in lists to regular Python types
                    serializable_data[key] = [
                        item.tolist() if isinstance(item, np.ndarray) else str(item)
                        for item in value
                    ]
                else:
                    serializable_data[key] = value
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            logging.info(f"Saved IDs cache to {cache_file}")
        except (IOError, TypeError) as e:
            logging.warning(f"Failed to save IDs cache to {cache_file}: {e}")

    def get_all_ids(self, use_cache=True):
        r"""Returns all types of IDs in the dataset in a single pass through recordings.
        
        This is an optimized version that collects all ID types (unit, session, 
        subject, brainset) in one iteration instead of multiple separate iterations.
        
        The function uses a two-level caching system:
        1. Memory cache: Cached in instance variable for fastest access
        2. Disk cache: Persistent cache in ~/.torch_brain_cache/dataset_ids/
        
        Args:
            use_cache: If True, use both memory and disk cache to avoid re-computation.
            
        Returns:
            dict: Dictionary containing all ID types:
                - 'unit_ids': List of all unit IDs 
                - 'session_ids': List of all unique session IDs
                - 'subject_ids': List of all unique subject IDs
                - 'brainset_ids': List of all unique brainset IDs
        """
        # Check memory cache first (fastest)
        if use_cache and hasattr(self, '_cached_all_ids'):
            return self._cached_all_ids
        
        # Check disk cache second (fast, persistent)
        if use_cache:
            disk_cached_result = self._load_ids_from_disk_cache()
            if disk_cached_result is not None:
                # Store in memory cache for even faster subsequent access
                self._cached_all_ids = disk_cached_result
                logging.info("Loaded IDs from disk cache")
                return disk_cached_result
            
        unit_ids_list = []
        session_ids_set = set()
        subject_ids_set = set()
        brainset_ids_set = set()
        sampling_intervals_dict = {}
        
        for recording_id in tqdm(
            self.recording_dict.keys(), desc="Collecting all IDs"
        ):
            file = h5py.File(self.recording_dict[recording_id]["filename"], "r")
            data = Data.from_hdf5(file, lazy=True)
            
            unit_ids = self._get_unit_ids_with_prefix(data)
            unit_ids_list.extend(unit_ids)
            
            session_id = self._get_session_id_with_prefix(data)
            session_ids_set.add(session_id)
        
            subject_id = self._get_subject_id_with_prefix(data)
            subject_ids_set.add(subject_id)
        
            brainset_ids_set.add(data.brainset.id)

            sampling_domain = (
                f"{self.split}_domain" if self.split is not None else "domain"
            )
            sampling_intervals = getattr(data, sampling_domain)
            sampling_intervals_dict[recording_id] = (list(sampling_intervals.start), list(sampling_intervals.end))
            # del data
            file.close()

        result = {
            'unit_ids': unit_ids_list,
            'session_ids': sorted(list(session_ids_set)),
            'subject_ids': sorted(list(subject_ids_set)),
            'brainset_ids': sorted(list(brainset_ids_set)),
            'sampling_intervals': sampling_intervals_dict,
        }
        
        # Cache the results if requested
        if use_cache:
            # Store in memory cache
            self._cached_all_ids = result
            # Save to disk cache for persistence across sessions
            self._save_ids_to_disk_cache(result)
            
        return result

    def get_unit_ids(self):
        r"""Returns all unit ids in the dataset."""
        return self.get_all_ids()['unit_ids']

    def get_session_ids(self):
        r"""Returns the session ids of the dataset."""
        return self.get_all_ids()['session_ids']

    def get_subject_ids(self):
        r"""Returns all subject ids in the dataset."""
        return self.get_all_ids()['subject_ids']

    def get_brainset_ids(self):
        r"""Returns all brainset ids in the dataset."""
        return self.get_all_ids()['brainset_ids']

    def clear_ids_cache(self):
        r"""Clears both memory and disk cached IDs. Useful when the dataset has been modified."""
        # Clear memory cache
        if hasattr(self, '_cached_all_ids'):
            delattr(self, '_cached_all_ids')
        
        # Clear disk cache
        cache_file = self._get_cache_file_path()
        if cache_file.exists():
            try:
                cache_file.unlink()
                logging.info(f"Cleared disk cache: {cache_file}")
            except OSError as e:
                logging.warning(f"Failed to clear disk cache {cache_file}: {e}")

    def clear_all_ids_cache_files(self):
        r"""Clears all IDs cache files from the cache directory. Useful for cleanup."""
        cache_dir = self._get_cache_directory()
        try:
            cache_files = list(cache_dir.glob("ids_cache_*.json"))
            for cache_file in cache_files:
                cache_file.unlink()
            logging.info(f"Cleared {len(cache_files)} cache files from {cache_dir}")
        except OSError as e:
            logging.warning(f"Failed to clear cache files from {cache_dir}: {e}")

    def get_number_of_tokens(self):
        r"""Returns the number of tokens in the dataset."""
        tokens_total = 0
        for recording_id in tqdm(
            self.recording_dict.keys(), desc="Collecting number of tokens"
        ):
            data = self._get_data_object(recording_id)
            if self.split == "train":
                tokens_total += sum(data.eeg.train_mask) * data.eeg.signal.shape[1]
            elif self.split == "val":
                tokens_total += sum(data.eeg.val_mask) * data.eeg.signal.shape[1]
            elif self.split == "test":
                tokens_total += sum(data.eeg.test_mask) * data.eeg.signal.shape[1]
            else:
                raise ValueError(f"Invalid split: {self.split}")
        return tokens_total

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
