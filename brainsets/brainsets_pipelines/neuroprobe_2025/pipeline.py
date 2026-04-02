# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "neuroprobe==0.1.7",
#   "numpy==2.2.6",
# ]
# ///

from itertools import product
import os
import time
import h5py
import logging
import shutil
import urllib.request
import zipfile
import torch
from torch.utils.data import Subset
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Literal, Tuple, Optional, NamedTuple, get_args
from tqdm import tqdm

from brainsets.pipeline import BrainsetPipeline
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries
from brainsets.descriptions import (
    BrainsetDescription,
    SubjectDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from brainsets import serialize_fn_map

logging.basicConfig(level=logging.INFO)


BASE_URL = "https://braintreebank.dev"
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_MAX_RETRIES = 2
DOWNLOAD_RETRY_BACKOFF_SECONDS = 1.0
COMMON_ASSETS = [
    "data/localization.zip",
    "data/subject_timings.zip",
    "data/subject_metadata.zip",
    "data/electrode_labels.zip",
    "data/speaker_annotations.zip",
    "data/scene_annotations.zip",
    "data/transcripts.zip",
    "data/trees.zip",
    "data/movie_frames.zip",
    "data/corrupted_elec.json",
]

FILENAME_MAP = lambda sub_id, trial_id: f"sub_{sub_id}_trial{trial_id:03d}"
ASSET_PATH_MAP = (
    lambda sub_id, trial_id: f"data/subject_data/sub_{sub_id}/trial{trial_id:03d}/{FILENAME_MAP(sub_id, trial_id)}.h5.zip"
)

SUBSET_TIER_KEY_MAP = lambda lite, nano: (
    "lite" if lite else ("nano" if nano else "full")
)
LABEL_MODE_KEY_MAP = lambda binary_tasks: "binary" if binary_tasks else "multiclass"
EvalSettingOption = Literal["within_session", "cross_x"]
ALL_EVAL_SETTINGS = {
    "lite": [True, False],
    "nano": [True, False],
    "binary_tasks": [True, False],
    "eval_setting": get_args(EvalSettingOption),
}


class DownloadedAsset(NamedTuple):
    path: Path
    subject_id: int
    trial_id: int


def split_selector_key(
    *,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    eval_setting: str,
    eval_name: str,
    fold_idx: int,
    split_type: str,
) -> str:
    # Shared selector key used for both splits.<key> and channels.<key>.
    return (
        f"{SUBSET_TIER_KEY_MAP(lite, nano)}${LABEL_MODE_KEY_MAP(binary_tasks)}$"
        f"{eval_setting}${eval_name}$fold{fold_idx}${split_type}"
    )


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--no_splits",
    action="store_true",
    help="Skip split extraction; write only processed data.",
)


class Pipeline(BrainsetPipeline):
    brainset_id = "neuroprobe_2025"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Optional[Namespace],
    ) -> pd.DataFrame:
        raw_dir.mkdir(exist_ok=True, parents=True)
        _prepare_neuroprobe_lib(raw_dir)

        # Ensure shared metadata/assets are present once before parallel workers run.
        logging.info(
            "Downloading common assets for neuroprobe_2025; this may take several minutes."
        )
        _ensure_common_assets(
            raw_dir,
            COMMON_ASSETS,
            overwrite=bool(args and args.redownload),
        )

        manifest_list = [
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "filename": FILENAME_MAP(subject_id, trial_id),
            }
            for subject_id, trial_id in neuroprobe_config.NEUROPROBE_FULL_SUBJECT_TRIALS
        ]

        # trials (manifest items) are indexed by filename
        manifest = pd.DataFrame(manifest_list).set_index("filename")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        extracted_path = _download_and_extract(
            self.raw_dir,
            ASSET_PATH_MAP(manifest_item.subject_id, manifest_item.trial_id),
            overwrite=bool(self.args and self.args.redownload),
        )
        return DownloadedAsset(
            path=extracted_path,
            subject_id=manifest_item.subject_id,
            trial_id=manifest_item.trial_id,
        )

    def process(self, download_output):
        _prepare_neuroprobe_lib(self.raw_dir)
        self.update_status("Processing")
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        output_path = self.processed_dir / download_output.path.name
        if output_path.exists() and not (self.args and self.args.reprocess):
            logging.info(f"Skipping processing for {output_path} because it exists")
            self.update_status("Skipped Processing")
            return

        brainset_description = get_brainset_description()
        subject_id = download_output.subject_id
        trial_id = download_output.trial_id

        logging.info(
            f"Processing {download_output.path} to {self.processed_dir}\n"
            f"  subject_id: {subject_id}\n"
            f"  trial_id: {trial_id}\n"
            f"  no_splits: {self.args.no_splits}"
        )

        # subject metadata
        self.update_status("Extracting subject metadata")
        subject = _get_subject_metadata(subject_id)

        # extract channel data & splits (if not disabled)
        if self.args.no_splits:
            self.update_status("Extracting channel data")
            subject_obj = neuroprobe.BrainTreebankSubject(
                subject_id=subject_id,
                allow_corrupted=False,
                cache=False,
                dtype=torch.float32,
                coordinates_type="lpi",
            )
            channels = _extract_channel_data(subject_obj)
            split_indices = {}
            split_channel_masks = {}
            logging.info("Skipping split extraction (--no_splits)")
        else:
            split_indices, split_channel_masks = self.iterate_extract_splits(
                subject_id, trial_id
            )
            channels = self.all_channels[subject_id]
            logging.info(f"Extracted {len(split_indices)} splits")

        # extract neural data
        self.update_status("Extracting neural data")
        seeg_data = _extract_neural_data(download_output.path, channels)
        logging.info(
            f"Loaded and registered {len(seeg_data)} samples of neural data with {len(channels)} channels"
        )

        self.update_status("Registering session")
        recording_id = FILENAME_MAP(subject_id, trial_id)
        data = Data(
            brainset=brainset_description,
            subject=subject,
            # Keep session.id canonical and globally unique.
            session=Data(id=recording_id),
            # neural activity
            seeg_data=seeg_data,
            channels=channels,
            # domain
            domain=seeg_data.domain,
        )

        # Register all split intervals under data.splits using shared selector keys.
        if not self.args.no_splits:
            splits = Data(domain=seeg_data.domain)
            self.update_status("Registering splits")
            for split_key, intervals in split_indices.items():
                setattr(splits, split_key, intervals)
            data.splits = splits
            # Store split-specific channel masks separately from base channel metadata.
            channel_splits = Data()
            for split_key, split_mask in split_channel_masks.items():
                setattr(channel_splits, split_key, split_mask)
            data.channel_splits = channel_splits

        # save data to disk
        self.update_status("Storing")
        path = self.processed_dir / download_output.path.name
        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
        logging.info(f"Saved data to {path}")

    def iterate_extract_splits(
        self, subject_id: int, trial_id: int
    ) -> Tuple[Dict[str, Interval], Dict[str, np.ndarray]]:
        _prepare_neuroprobe_lib(self.raw_dir)
        if not hasattr(self, "all_subjects"):
            # load all subjects and channels once
            # channels will be populated with subsets for each fold
            unique_subject_ids = sorted(
                {sid for sid, _ in neuroprobe_config.NEUROPROBE_FULL_SUBJECT_TRIALS}
            )
            self.all_subjects = {
                subject_id: neuroprobe.BrainTreebankSubject(
                    subject_id=subject_id,
                    allow_corrupted=False,
                    cache=False,
                    dtype=torch.float32,
                    coordinates_type="lpi",
                )
                for subject_id in unique_subject_ids
            }
            self.all_channels = {
                subject_id: _extract_channel_data(self.all_subjects[subject_id])
                for subject_id in self.all_subjects
            }

        all_combinations = product(*ALL_EVAL_SETTINGS.values())
        split_indices: Dict[str, Interval] = {}
        split_channel_masks: Dict[str, np.ndarray] = {}
        for setting_combination in all_combinations:
            lite, nano, binary_tasks, eval_setting = setting_combination
            if lite and nano:  # lite and nano cannot be True at the same time
                continue
            if (
                eval_setting == "cross_x" and nano
            ):  # keep benchmark parity with cross-session support
                continue
            if (
                lite
                and (subject_id, trial_id)
                not in neuroprobe_config.NEUROPROBE_LITE_SUBJECT_TRIALS
            ):
                continue
            if (
                nano
                and (subject_id, trial_id)
                not in neuroprobe_config.NEUROPROBE_NANO_SUBJECT_TRIALS
            ):
                continue

            self.update_status(
                f"Extracting splits (lite={lite}, nano={nano}, binary_tasks={binary_tasks}, eval_setting={eval_setting})"
            )
            _split_indices, _split_channel_masks = _extract_and_structure_splits(
                all_subjects=self.all_subjects,
                all_channels=self.all_channels,
                subject_id=subject_id,
                trial_id=trial_id,
                lite=lite,
                nano=nano,
                binary_tasks=binary_tasks,
                eval_setting=eval_setting,
            )
            for key, value in _split_indices.items():
                split_indices[key] = value
            for key, value in _split_channel_masks.items():
                split_channel_masks[key] = value

        return split_indices, split_channel_masks


def get_brainset_description() -> BrainsetDescription:
    return BrainsetDescription(
        id="neuroprobe_2025",
        origin_version="dataset=0.0.0; neuroprobe=0.1.7",
        derived_version="1.0.0",
        source="https://neuroprobe.dev/",
        description="High-resolution neural datasets enable foundation models for the next generation of "
        "brain-computer interfaces and neurological treatments. The community requires rigorous benchmarks "
        "to discriminate between competing modeling approaches, yet no standardized evaluation frameworks "
        "exist for intracranial EEG (iEEG) recordings. To address this gap, we present Neuroprobe: a suite "
        "of decoding tasks for studying multi-modal language processing in the brain. Unlike scalp EEG, "
        "intracranial EEG requires invasive surgery to implant electrodes that record neural activity directly "
        "from the brain with minimal signal distortion. Neuroprobe is built on the BrainTreebank dataset, which "
        "consists of 40 hours of iEEG recordings from 10 human subjects performing a naturalistic movie viewing "
        "task. Neuroprobe serves two critical functions. First, it is a mine from which neuroscience insights "
        "can be drawn. The high temporal and spatial resolution of the labeled iEEG allows researchers to "
        "systematically determine when and where computations for each aspect of language processing occur "
        "in the brain by measuring the decodability of each feature across time and all electrode locations. "
        "Using Neuroprobe, we visualize how information flows from key language and audio processing sites in "
        "the superior temporal gyrus to sites in the prefrontal cortex. We also demonstrate the progression "
        "from processing simple auditory features (e.g., pitch and volume) to more complex language features "
        "(part of speech and word position in the sentence tree) in a purely data-driven manner. Second, as "
        "the field moves toward neural foundation models trained on large-scale datasets, Neuroprobe provides "
        "a rigorous framework for comparing competing architectures and training protocols. We found that the "
        "linear baseline on spectrogram inputs is surprisingly strong, beating frontier foundation models on "
        "many tasks. Neuroprobe is designed with computational efficiency and ease of use in mind. We make "
        "the code for Neuroprobe openly available and will maintain a public leaderboard of evaluation submissions, "
        "aiming to enable measurable progress in the field of iEEG foundation models.",
    )


def _get_subject_metadata(subject_id: int) -> SubjectDescription:
    return SubjectDescription(
        id=str(subject_id),
        species=Species.HOMO_SAPIENS,
        sex=Sex.UNKNOWN,
    )


def _prepare_neuroprobe_lib(raw_dir: Path) -> None:
    # neuroprobe requires the raw data to be set as an environment variable
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(raw_dir)
    global neuroprobe, neuroprobe_config, neuroprobe_train_test_splits
    import neuroprobe
    import neuroprobe.config as neuroprobe_config
    import neuroprobe.train_test_splits as neuroprobe_train_test_splits


# subject is neuroprobe.BrainTreebankSubject object
def _extract_channel_data(subject) -> ArrayDict:
    channel_name_basis = np.array(
        list(subject.h5_neural_data_keys.keys()), dtype=np.str_
    )
    aligned_localization = subject.localization_data.set_index("Electrode").reindex(
        channel_name_basis
    )
    channels = ArrayDict(
        id=np.arange(len(channel_name_basis)),
        name=channel_name_basis,  # e.g. T1bIc1
        h5_label=np.array(  # e.g. electrode_76
            [subject.h5_neural_data_keys[name] for name in channel_name_basis]
        ),
        included=np.isin(channel_name_basis, subject.electrode_labels).astype(
            np.bool_
        ),  # excludes corrupted and trigger electrodes
        type=np.ones(len(channel_name_basis)) * int(RecordingTech.STEREO_EEG),
    )
    # register localization data for each channel
    for col in aligned_localization.columns:
        loc_series = aligned_localization[col]
        # not all channels have localization data
        if pd.api.types.is_string_dtype(loc_series):
            full_column = loc_series.fillna("").to_numpy().astype(np.str_)
        elif pd.api.types.is_numeric_dtype(loc_series):
            full_column = loc_series.fillna(np.nan).to_numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {loc_series.dtype}")
        setattr(channels, f"localization_{col}", full_column)
    return channels


def _extract_and_structure_splits(
    all_subjects: Dict[int, object],
    all_channels: Dict[int, ArrayDict],
    subject_id: int,
    trial_id: int,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    eval_setting: EvalSettingOption,
) -> Tuple[Dict[str, Interval], Dict[str, np.ndarray]]:
    split_indices: Dict[str, Interval] = {}
    split_channel_masks: Dict[str, np.ndarray] = {}

    assert (
        len(neuroprobe_config.NEUROPROBE_TASKS_MAPPING) > 0
    ), "No tasks to extract splits for"
    for eval_name in neuroprobe_config.NEUROPROBE_TASKS_MAPPING:
        # load splits via neuroprobe API
        folds = _extract_splits(
            all_subjects=all_subjects,
            subject_id=subject_id,
            trial_id=trial_id,
            lite=lite,
            nano=nano,
            binary_tasks=binary_tasks,
            eval_name=eval_name,
            eval_setting=eval_setting,
        )

        # load channels for each fold
        assert len(folds) > 0, "No folds to extract splits for"
        channels = all_channels[subject_id]
        for fold_idx, fold in enumerate(folds):
            for split_type, dataset_key in (
                ("train", "train_dataset"),
                ("val", "val_dataset"),
                ("test", "test_dataset"),
            ):
                selector_key = split_selector_key(
                    lite=lite,
                    nano=nano,
                    binary_tasks=binary_tasks,
                    eval_setting=eval_setting,
                    eval_name=eval_name,
                    fold_idx=fold_idx,
                    split_type=split_type,
                )
                # Register split-specific channel mask for this fold.
                split_channel_masks[selector_key] = np.isin(
                    channels.name, _get_electrode_labels(fold[dataset_key])
                ).astype(bool)

                # Store split interval indices using the same selector key semantics.
                split_indices[selector_key] = _intervals_from_dataset(fold[dataset_key])

    return split_indices, split_channel_masks


def _extract_splits(
    all_subjects: Dict[int, object],
    subject_id: int,
    trial_id: int,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    eval_name: str,
    eval_setting: EvalSettingOption,
):
    dtype = torch.float32
    max_samples = None
    start_neural_data_before_word_onset = 0
    end_neural_data_after_word_onset = neuroprobe_config.SAMPLING_RATE * 1

    if eval_setting == "within_session":
        folds = neuroprobe_train_test_splits.generate_splits_within_session(
            test_subject=all_subjects[subject_id],
            test_trial_id=trial_id,
            eval_name=eval_name,
            dtype=dtype,
            lite=lite,
            nano=nano,
            binary_tasks=binary_tasks,
            output_indices=True,
            output_dict=True,
            start_neural_data_before_word_onset=start_neural_data_before_word_onset,
            end_neural_data_after_word_onset=end_neural_data_after_word_onset,
            max_samples=max_samples,
        )
        return folds
    elif eval_setting == "cross_x":
        return _extract_local_role_splits(
            subject=all_subjects[subject_id],
            trial_id=trial_id,
            eval_name=eval_name,
            dtype=dtype,
            lite=lite,
            nano=nano,  # NOTE: We add nano here for completeness
            binary_tasks=binary_tasks,
            output_indices=True,
            output_dict=True,
            start_neural_data_before_word_onset=start_neural_data_before_word_onset,
            end_neural_data_after_word_onset=end_neural_data_after_word_onset,
            max_samples=max_samples,
        )
    raise ValueError(f"Unsupported eval_setting: {eval_setting}")


def _extract_local_role_splits(
    subject: object,
    trial_id: int,
    eval_name: str,
    dtype: torch.dtype,
    lite: bool,
    nano: bool,
    binary_tasks: bool,
    output_indices: bool,
    output_dict: bool,
    start_neural_data_before_word_onset: int,
    end_neural_data_after_word_onset: int,
    max_samples: Optional[int],
) -> List[Dict[str, object]]:
    local_dataset = neuroprobe.BrainTreebankSubjectTrialBenchmarkDataset(
        subject=subject,
        trial_id=trial_id,
        dtype=dtype,
        eval_name=eval_name,
        binary_tasks=binary_tasks,
        output_indices=output_indices,
        output_dict=output_dict,
        start_neural_data_before_word_onset=start_neural_data_before_word_onset,
        end_neural_data_after_word_onset=end_neural_data_after_word_onset,
        lite=lite,
        nano=nano,
        max_samples=max_samples,
    )

    local_size = len(local_dataset)
    val_size = local_size // 2
    val_indices = list(range(val_size))
    test_indices = list(range(val_size, local_size))
    val_dataset = Subset(local_dataset, val_indices)
    test_dataset = Subset(local_dataset, test_indices)

    # Mirror Neuroprobe split metadata expectations on Subset wrappers.
    val_dataset.electrode_coordinates = local_dataset.electrode_coordinates
    val_dataset.electrode_labels = local_dataset.electrode_labels
    test_dataset.electrode_coordinates = local_dataset.electrode_coordinates
    test_dataset.electrode_labels = local_dataset.electrode_labels

    return [
        {
            # Intentional: when this split is consumed as training, we use the
            # entire local sample set for training. cross_x uses different
            # subject/session roles for train vs test upstream, so this does
            # not leak evaluation windows into training.
            "train_dataset": local_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
        }
    ]


def _intervals_from_dataset(dataset) -> Interval:
    items = [_unpack_dataset_item(item) for item in dataset]
    if len(items) == 0:
        return Interval(
            start=np.array([], dtype=np.float64),
            end=np.array([], dtype=np.float64),
            label=np.array([], dtype=np.int64),
        )

    windows, labels = zip(*items)
    window_array = np.array(windows)
    label_array = np.array(labels)

    return Interval(
        start=window_array[:, 0].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
        end=window_array[:, 1].astype(np.float64) / neuroprobe_config.SAMPLING_RATE,
        label=label_array,
    )


def _extract_neural_data(input_file: Path, channels: ArrayDict) -> RegularTimeSeries:
    with h5py.File(input_file, "r") as f:
        data = None
        # read channels in same order as in channels object
        for c, key in enumerate(channels.h5_label):
            if data is None:
                data = np.zeros(
                    (f["data"][key].shape[0], len(channels)), dtype=np.float32
                )
            data[:, c] = f["data"][key][:]

    seeg_data = RegularTimeSeries(
        data=data,
        sampling_rate=float(neuroprobe_config.SAMPLING_RATE),
        domain="auto",
    )

    return seeg_data


def _download_file(url: str, dest: Path, *, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")

    total_attempts = DOWNLOAD_MAX_RETRIES + 1
    for attempt in range(1, total_attempts + 1):
        try:
            with (
                urllib.request.urlopen(
                    url, timeout=DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                tmp_dest.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            tmp_dest.replace(dest)
            return
        except Exception:
            if tmp_dest.exists():
                tmp_dest.unlink()
            if attempt == total_attempts:
                raise
            logging.warning(
                "Download attempt %d/%d failed for %s; retrying in %.1fs.",
                attempt,
                total_attempts,
                url,
                DOWNLOAD_RETRY_BACKOFF_SECONDS,
            )
            time.sleep(DOWNLOAD_RETRY_BACKOFF_SECONDS)


def _download_and_extract(raw_dir: Path, href: str, *, overwrite: bool) -> Path:
    url = f"{BASE_URL}/{href}"
    basename = Path(href).name
    zip_path = raw_dir / basename

    if href.endswith(".zip"):
        extracted_path = raw_dir / Path(href).stem
        if extracted_path.exists() and not overwrite:
            return extracted_path

        _download_file(url, zip_path, overwrite=overwrite)
        with zipfile.ZipFile(zip_path, "r") as zip_handle:
            zip_handle.extractall(raw_dir)
        zip_path.unlink()
        return extracted_path

    extracted_path = raw_dir / basename
    _download_file(url, extracted_path, overwrite=overwrite)
    return extracted_path


def _ensure_common_assets(raw_dir: Path, assets: list[str], *, overwrite: bool) -> None:
    for asset in tqdm(assets, desc="Downloading neuroprobe common assets"):
        _download_and_extract(raw_dir, asset, overwrite=overwrite)


def _get_electrode_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "electrode_labels"):
        return dataset.electrode_labels
    if hasattr(dataset, "dataset"):
        return _get_electrode_labels(dataset.dataset)
    raise AttributeError(
        f"{type(dataset).__name__} has no electrode_labels and no nested dataset"
    )


def _unpack_dataset_item(item) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        return item["data"], item["label"]
    return item[0], item[1]
