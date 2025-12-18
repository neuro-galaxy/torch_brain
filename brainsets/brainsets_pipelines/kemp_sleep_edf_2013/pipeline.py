# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "boto3~=1.41.0",
#   "scikit-learn==1.7.2",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import h5py
import logging
import mne
import numpy as np
import pandas as pd

from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from brainsets.taxonomy import RecordingTech, Species, Sex
from brainsets.pipeline import BrainsetPipeline
from brainsets.utils.split import (
    chop_intervals,
    generate_stratified_folds,
)
from brainsets.utils.s3_utils import get_s3_client_for_download
from temporaldata import Data, Interval, RegularTimeSeries, ArrayDict


logging.basicConfig(level=logging.INFO)


parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--study_type",
    type=str,
    choices=["sc", "st", "both"],
    default="both",
    help="Which study to download: 'sc' (Sleep Cassette), 'st' (Sleep Telemetry), or 'both'",
)


class Pipeline(BrainsetPipeline):
    brainset_id = "kemp_sleep_edf_2013"
    bucket = "physionet-open"
    prefix = "sleep-edfx/1.0.0"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        s3 = get_s3_client_for_download()

        prefixes = []
        if args.study_type in ["sc", "both"]:
            prefixes.append(f"{cls.prefix}/sleep-cassette/")
        if args.study_type in ["st", "both"]:
            prefixes.append(f"{cls.prefix}/sleep-telemetry/")

        def find_hypnogram_key(s3, psg_key: str) -> Optional[str]:
            """Find the hypnogram file corresponding to a PSG file."""
            psg_path = Path(psg_key)
            base_name = psg_path.stem

            for prefix_len in [7, 6]:
                prefix = base_name[:prefix_len]
                search_prefix = str(psg_path.parent / prefix)
                response = s3.list_objects_v2(Bucket=cls.bucket, Prefix=search_prefix)

                for obj in response.get("Contents", []):
                    if "Hypnogram" in obj["Key"] and obj["Key"].endswith(".edf"):
                        return obj["Key"]

            return None

        manifest_rows = []

        for prefix in prefixes:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=cls.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("PSG.edf"):
                        filename = Path(key).name
                        base_name = filename.replace(".edf", "")

                        hypnogram_key = find_hypnogram_key(s3, key)

                        if "sleep-cassette" in key:
                            subject_id = base_name[3:5]
                            study_type = "sleep_cassette"
                        elif "sleep-telemetry" in key:
                            subject_id = base_name[3:5]
                            study_type = "sleep_telemetry"
                        else:
                            subject_id = base_name
                            study_type = "unknown"

                        session_id = f"{study_type}_{base_name}"

                        manifest_rows.append(
                            {
                                "session_id": session_id,
                                "psg_s3_key": key,
                                "hypnogram_s3_key": hypnogram_key,
                                "study_type": study_type,
                                "subject_id": subject_id,
                            }
                        )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> Tuple[Path, Path]:
        self.update_status("DOWNLOADING")
        s3 = get_s3_client_for_download()

        psg_key = manifest_item.psg_s3_key
        hypnogram_key = manifest_item.hypnogram_s3_key

        if not hypnogram_key:
            raise ValueError(f"No hypnogram found for PSG file: {psg_key}")

        psg_local = self.raw_dir / Path(psg_key).relative_to(self.prefix)
        psg_local.parent.mkdir(parents=True, exist_ok=True)

        if not psg_local.exists() or self.args.redownload:
            logging.info(f"Downloading PSG: {Path(psg_key).name}")
            s3.download_file(self.bucket, psg_key, str(psg_local))
        else:
            logging.info(f"Skipping PSG download, file exists: {psg_local}")

        hypnogram_local = self.raw_dir / Path(hypnogram_key).relative_to(self.prefix)
        hypnogram_local.parent.mkdir(parents=True, exist_ok=True)

        if not hypnogram_local.exists() or self.args.redownload:
            logging.info(f"Downloading Hypnogram: {Path(hypnogram_key).name}")
            s3.download_file(self.bucket, hypnogram_key, str(hypnogram_local))
        else:
            logging.info(f"Skipping Hypnogram download, file exists: {hypnogram_local}")

        return psg_local, hypnogram_local

    def process(self, download_output: Tuple[Path, Path]) -> None:
        psg_path, hypnogram_path = download_output

        self.update_status("PROCESSING")

        base_name = psg_path.stem

        output_path = self.processed_dir / f"{base_name}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            logging.info(f"Skipping processing, file exists: {output_path}")
            return

        brainset_description = BrainsetDescription(
            id="kemp_sleep_edf_2013",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://www.physionet.org/content/sleep-edfx/1.0.0/",
            description="Sleep-EDF Database Expanded containing 197 whole-night "
            "polysomnographic sleep recordings with EEG, EOG, EMG, and sleep stage annotations.",
        )

        logging.info(f"Processing file: {psg_path}")

        self.update_status("Loading EDF")
        raw_psg = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)

        self.update_status("Extracting Metadata")
        age, sex = parse_subject_metadata(raw_psg)

        if "SC4" in base_name:
            subject_id = base_name[3:5]
            study_type = "sleep_cassette"
        elif "ST7" in base_name:
            subject_id = base_name[3:5]
            study_type = "sleep_telemetry"
        else:
            subject_id = base_name
            study_type = "unknown"

        subject = SubjectDescription(
            id=f"{study_type}_{subject_id}",
            species=Species.HOMO_SAPIENS,
            age=age,
            sex=sex,
        )

        recording_date = raw_psg.info.get("meas_date")
        if recording_date is not None:
            recording_date = recording_date.strftime("%Y-%m-%d")

        session_description = SessionDescription(
            id=base_name,
            recording_date=recording_date,
        )

        device_description = DeviceDescription(
            id=base_name,
            recording_tech=RecordingTech.POLYSOMNOGRAPHY,
        )

        self.update_status("Extracting Signals")
        signals, units = extract_signals(raw_psg)

        self.update_status("Extracting Sleep Stages")
        stages = extract_sleep_stages(str(hypnogram_path))

        self.update_status("Creating Splits")
        splits = create_splits(stages, n_folds=3, seed=42)

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            eeg=signals,
            units=units,
            stages=stages,
            splits=splits,
            domain=signals.domain,
        )

        self.update_status("Storing")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logging.info(f"Saved processed data to: {output_path}")


def parse_subject_metadata(raw: mne.io.Raw) -> Tuple[Optional[int], Sex]:
    """Extract subject metadata from EDF header."""
    info = raw.info
    subject_info = info.get("subject_info", {})

    # In this dataset, the age is stored in the last_name field, in format "Xyr".
    try:
        age_str = subject_info.get("last_name")
        if age_str is not None:
            age = int(age_str.replace("yr", ""))
    except (ValueError, AttributeError) as e:
        logging.warning(f"Could not parse age from last_name: {age_str}, setting to 0")
        age = 0

    sex_str = subject_info.get("sex")

    if sex_str is not None:
        sex = Sex.MALE if sex_str == 1 else Sex.FEMALE if sex_str == 2 else Sex.UNKNOWN
    else:
        sex = Sex.UNKNOWN

    return age, sex


def extract_signals(raw_psg: mne.io.Raw) -> Tuple[RegularTimeSeries, ArrayDict]:
    """Extract physiological signals from PSG EDF file as a RegularTimeSeries."""
    data, times = raw_psg.get_data(return_times=True)
    ch_names = raw_psg.ch_names

    signal_list = []
    unit_meta = []

    for idx, ch_name in enumerate(ch_names):
        ch_name_lower = ch_name.lower()
        signal_data = data[idx, :]

        modality = None
        if (
            "eeg" in ch_name_lower
            or "fpz-cz" in ch_name_lower
            or "pz-oz" in ch_name_lower
        ):
            modality = "EEG"
        elif "eog" in ch_name_lower:
            modality = "EOG"
        elif "emg" in ch_name_lower:
            modality = "EMG"
        elif "resp" in ch_name_lower:
            modality = "RESP"
        elif "temp" in ch_name_lower:
            modality = "TEMP"
        else:
            continue

        signal_list.append(signal_data)

        unit_meta.append(
            {
                "id": str(ch_name),
                "modality": modality,
            }
        )

    if not signal_list:
        raise ValueError("No signals extracted from PSG file")

    stacked_signals = np.stack(signal_list, axis=1)

    signals = RegularTimeSeries(
        signal=stacked_signals,
        sampling_rate=raw_psg.info["sfreq"],
        domain=Interval(start=times[0], end=times[-1]),
    )

    units_df = pd.DataFrame(unit_meta)
    units = ArrayDict.from_dataframe(units_df)

    return signals, units


def extract_sleep_stages(hypnogram_file: str) -> Interval:
    """Extract sleep stage annotations from hypnogram EDF+ file as an Interval object."""
    annotations = mne.read_annotations(hypnogram_file)

    sleep_stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
        "Sleep stage ?": 6,
        "Movement time": 7,
    }

    starts = []
    ends = []
    stage_names = []
    stage_ids = []

    for annot_onset, annot_duration, annot_description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        starts.append(annot_onset)
        ends.append(annot_onset + annot_duration)
        stage_names.append(annot_description)
        stage_ids.append(sleep_stage_map[annot_description])

    if len(starts) == 0:
        raise ValueError(
            f"No sleep stage annotations found in hypnogram: {hypnogram_file}"
        )

    return Interval(
        start=np.array(starts),
        end=np.array(ends),
        names=np.array(stage_names),
        id=np.array(stage_ids, dtype=np.int64),
    )


def create_splits(
    stages: Interval, epoch_duration: float = 30.0, n_folds: int = 5, seed: int = 42
) -> Data:
    """Generate train/valid/test splits from sleep stage intervals.

    The Sleep-EDF dataset does not provide predefined splits. We generate standardized
    splits to enable reproducible cross-validation and ensure the research community
    can share and compare results using the same data partitions.

    The unknown sleep stage (stage 6) is the only one removed from the splits to maintain flexibility.
    Users can still access all stages in the raw data and choose the number of sleep stages relevant
    to their research.
    """
    if len(stages) == 0:
        raise ValueError("No stages provided for splitting")

    chopped = chop_intervals(stages, duration=epoch_duration, check_no_overlap=True)
    logging.info(f"Chopped {len(stages)} stages into {len(chopped)} epochs")

    UNKNOWN_STAGE_ID = 6
    mask = ~np.isin(chopped.id, [UNKNOWN_STAGE_ID])
    filtered = chopped.select_by_mask(mask)
    logging.info(f"Filtered out unknown stages, {len(filtered)} epochs remaining")

    for stage_id, count in zip(*np.unique(filtered.id, return_counts=True)):
        if count < n_folds:
            mask = ~np.isin(filtered.id, [stage_id])
            filtered = filtered.select_by_mask(mask)
            logging.info(
                f"Filtered out stage {stage_id}, {len(filtered)} epochs remaining"
            )

    if len(filtered) == 0:
        raise ValueError("No valid epochs remaining after filtering")

    folds = generate_stratified_folds(
        filtered,
        stratify_by="id",
        n_folds=n_folds,
        val_ratio=0.2,
        seed=seed,
    )
    logging.info(f"Generated {n_folds} stratified folds")

    folds_dict = {f"fold_{i}": fold for i, fold in enumerate(folds)}
    splits = Data(**folds_dict, domain=filtered)

    return splits
