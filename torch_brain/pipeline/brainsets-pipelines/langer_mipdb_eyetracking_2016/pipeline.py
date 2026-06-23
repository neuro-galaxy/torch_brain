# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "boto3~=1.41.0",
#   "mne~=1.11.0",
#   "scikit-learn~=1.6.0",
# ]
# ///

import json
import logging
import re
import ssl
import sys
from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen

import h5py
import mne
import numpy as np
import pandas as pd

from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    SessionDescription,
    SubjectDescription,
    serialize_fn_map,
)
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.mne import (
    extract_channels,
    extract_measurement_date,
    extract_signal,
)
from torch_brain.utils.s3 import get_cached_s3_client, get_object_list
from torch_brain.utils.split import (
    generate_stratified_folds,
    generate_string_kfold_assignment,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eyetracking  # noqa: E402
import paradigm  # noqa: E402

# Metadata files are not hosted in the same s3 bucket as the public data
# Public data on the 126 subjects
SUBJECTS_METADATA_URL = (
    "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/MIPDB_PublicFile.csv"
)
# EEG Channel location file
CHANNEL_LOCATION_URL = (
    "https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/_static/GSN_HydroCel_129.sfp"
)
EPOCH_DURATION = 30.0  # seconds
SPLIT_N_FOLDS = 3
SPLIT_VAL_RATIO = 0.2
SEED = 42

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "cmi_mipdb_2016"
    bucket = "fcp-indi"
    prefix = "data/Projects/EEG_Eyetracking_CMI_data/"
    parser = parser

    @staticmethod
    def extract_annotations(raw: "mne.io.BaseRaw") -> Interval:
        raise NotImplementedError("extract_annotations is not yet implemented.")

    @staticmethod
    def generate_string_binary_assignment(
        string_id: str,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> str:
        raise NotImplementedError(
            "generate_string_binary_assignment is not yet implemented."
        )

    def _download_subject_metadata(self) -> Path:
        """Download the MIPDB public metadata CSV if not already cached locally.

        Returns the local path to the downloaded CSV file.
        """
        local_path = self.raw_dir / "MIPDB_PublicFile.csv"

        if not local_path.exists() or self.args.redownload:
            self.update_status("Downloading subject metadata CSV")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            # NITRC's certificate has a hostname mismatch; skip verification
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(SUBJECTS_METADATA_URL, context=ctx) as response:
                local_path.write_bytes(response.read())
        else:
            self.update_status("Subject metadata CSV already cached")

        return local_path

    def _download_channel_location(self) -> Path:
        """Download the GSN HydroCel 129 channel location SFP file.

        Returns the local path to the downloaded file.
        """
        local_path = self.raw_dir / "GSN_HydroCel_129.sfp"

        if not local_path.exists() or self.args.redownload:
            self.update_status("Downloading channel location file")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(CHANNEL_LOCATION_URL, context=ctx) as response:
                local_path.write_bytes(response.read())
        else:
            self.update_status("Channel location file already cached")

        return local_path

    @classmethod
    def get_manifest(cls, raw_dir: Path, args) -> pd.DataFrame:
        s3 = get_cached_s3_client()

        manifest_rows = []
        et_keys_by_subject: dict[str, list[str]] = {}

        rel_keys = get_object_list(bucket=cls.bucket, prefix=cls.prefix, s3_client=s3)

        for rel_key in rel_keys:
            rel_path = Path(rel_key)
            filename = rel_path.name

            if filename.startswith("."):
                continue

            rel_parts = rel_path.parts

            # EEG: subject_id/EEG/raw/raw_format/*.raw
            if (
                len(rel_parts) >= 4
                and rel_parts[1] == "EEG"
                and rel_parts[2] == "raw"
                and rel_parts[3] == "raw_format"
            ) and filename.endswith(".raw"):
                subject_id = rel_parts[0]
                session_id = f"{Path(filename).stem}"

                manifest_rows.append(
                    {
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "s3_key": cls.prefix + rel_key,
                    }
                )

            # ET: subject_id/Eyetracking/txt/*.txt
            if (
                len(rel_parts) >= 3
                and rel_parts[1] == "Eyetracking"
                and rel_parts[2] == "txt"
                and filename.endswith(".txt")
            ):
                subject_id = rel_parts[0]
                et_keys_by_subject.setdefault(subject_id, []).append(
                    cls.prefix + rel_key
                )

        for row in manifest_rows:
            et_keys = et_keys_by_subject.get(row["subject_id"], [])
            row["et_s3_keys"] = json.dumps(et_keys)

        manifest = pd.DataFrame(
            manifest_rows,
            columns=["session_id", "subject_id", "s3_key", "et_s3_keys"],
        ).set_index("session_id")
        return manifest

    def download(self, manifest_item) -> dict:
        self.update_status("DOWNLOADING")
        s3 = get_cached_s3_client()

        # Ensure dataset-level metadata is cached (downloaded once per dataset)
        self._download_subject_metadata()
        self._download_channel_location()

        # --- EEG file ---
        s3_key = manifest_item.s3_key
        eeg_path = self.raw_dir / Path(s3_key).relative_to(self.prefix)
        eeg_path.parent.mkdir(parents=True, exist_ok=True)

        if not eeg_path.exists() or self.args.redownload:
            self.update_status(f"Downloading EEG: {Path(s3_key).name}")
            s3.download_file(self.bucket, s3_key, str(eeg_path))
        else:
            self.update_status(f"Skipping EEG download, exists: {eeg_path}")

        # --- ET files (shared across sessions for a subject, cached) ---
        et_dir = None
        et_s3_keys = json.loads(manifest_item.et_s3_keys)
        if et_s3_keys:
            subject_id = manifest_item.subject_id
            et_dir = self.raw_dir / subject_id / "Eyetracking" / "txt"
            for et_key in et_s3_keys:
                et_path = self.raw_dir / Path(et_key).relative_to(self.prefix)
                et_path.parent.mkdir(parents=True, exist_ok=True)
                if not et_path.exists() or self.args.redownload:
                    self.update_status(f"Downloading ET: {Path(et_key).name}")
                    s3.download_file(self.bucket, et_key, str(et_path))
        else:
            logging.warning(
                f"No ET files available for subject {manifest_item.subject_id}"
            )

        return {"eeg": eeg_path, "et_dir": et_dir}

    def process(self, download_output: dict) -> None:
        raw_path = download_output["eeg"]
        et_dir = download_output.get("et_dir")

        self.update_status("PROCESSING")

        recording_id = raw_path.stem

        output_path = self.processed_dir / f"{recording_id}.h5"
        if output_path.exists() and not self.args.reprocess:
            self.update_status(f"Skipping processing, file exists: {output_path}")
            return

        brainset_description = BrainsetDescription(
            id="cmi_mipdb_2016",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://fcon_1000.projects.nitrc.org/indi/cmi_eeg/",
            description="The Child Mind Institute - MIPDB provides EEG,"
            "eye-tracking, and behavioral data across multiple paradigms from"
            "126 psychiatric and healthy subjects aged 6 - 44 years old.",
        )

        self.update_status("Loading EEG file")
        raw = mne.io.read_raw_egi(str(raw_path), preload=True, verbose=True)
        meas_date = extract_measurement_date(raw)

        session_description = SessionDescription(
            id=recording_id, recording_date=meas_date
        )

        subject_id = recording_id[:9]
        validate_subject_id(subject_id)
        device_description = DeviceDescription(
            id=f"GSN_HydroCel_129_{subject_id}",
            recording_tech="SCALP_EEG",
        )

        csv_path = self.raw_dir / "MIPDB_PublicFile.csv"
        subject_info = parse_subject_metadata(csv_path, subject_id)

        subject_description = SubjectDescription(
            id=subject_id,
            species="HUMAN",
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        self.update_status("Extracting EEG signal and channels")
        eeg_signal = extract_signal(raw)
        channels = extract_channels(raw)
        sfp_path = self.raw_dir / "GSN_HydroCel_129.sfp"
        if not hasattr(channels, "pos"):
            channels.pos = extract_channel_locations(sfp_path, channels)

        self.update_status("Extracting annotations")
        annotations = self.extract_annotations(raw)
        paradigm_intervals = paradigm.get_all_paradigm_intervals(annotations)
        paradigm_events = paradigm.get_paradigm_events(annotations, paradigm_intervals)

        # --- Eyetracking  ---
        et_data = None
        if et_dir is not None and len(paradigm_intervals) > 0:
            self.update_status("Processing eyetracking")
            et_data = eyetracking.process_session(
                et_dir,
                paradigm_intervals,
                paradigm_events,
                float(eeg_signal.domain.start[0]),
                float(eeg_signal.domain.end[0]),
            )
            if et_data is None:
                logging.warning(f"No ET data aligned for session {recording_id}")
        elif et_dir is None:
            logging.warning(f"No ET directory for session {recording_id}")

        self.update_status("Creating splits")
        splits = create_splits(
            eeg_domain=eeg_signal.domain,
            subject_id=subject_description.id,
            session_id=session_description.id,
            paradigm_intervals=paradigm_intervals,
        )

        self.update_status("Creating Data Object")
        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject_description,
            "session": session_description,
            "device": device_description,
            "eeg": eeg_signal,
            "channels": channels,
            "domain": eeg_signal.domain,
            "splits": splits,
        }
        if len(annotations) > 0:
            data_kwargs["annotations"] = annotations
        if len(paradigm_intervals) > 0:
            data_kwargs["paradigms"] = paradigm_intervals
        if len(paradigm_events) > 0:
            data_kwargs["paradigm_events"] = paradigm_events
        if et_data is not None:
            data_kwargs["eyetracking"] = et_data

        data = Data(**data_kwargs)

        self.update_status("Storing processed data to disk")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def validate_subject_id(subject_id: str) -> None:
    """Validate that subject_id matches the CMI MIPDB convention: A000xxxxx.

    Raises:
        ValueError: If subject_id does not match A000xxxxx format.
    """
    SUBJECT_ID_PATTERN = re.compile(r"^A000\d{5}$")
    if not SUBJECT_ID_PATTERN.match(subject_id):
        raise ValueError(
            f"Invalid subject_id for CMI MIPDB: {subject_id!r}. "
            "Expected format: A000xxxxx where x are digits (e.g. A00054400)."
        )


def parse_subject_metadata(csv_path: Path, subject_id: str) -> dict:
    """Look up subject metadata (age, sex) from the MIPDB public metadata.

    Args:
        csv_path: Path to the MIPDB public metadata CSV file.
        subject_id: Subject identifier (e.g., "A00054400").

    Returns:
        Dict with 'age' and 'sex' keys.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Subject metadata not found at {csv_path}. Run download() first."
        )
    df = pd.read_csv(csv_path, index_col="ID")

    if subject_id not in df.index:
        logging.warning("No metadata found for subject %s", subject_id)
        return {"age": 0.0, "sex": 0}

    subject = df.loc[subject_id]

    age = subject.get("Age", None)
    sex = subject.get("Sex", None)
    sex = int(sex) if pd.notna(sex) else 0

    return {"age": age, "sex": sex}


def extract_channel_locations(sfp_path: Path, channels: ArrayDict) -> np.ndarray:
    """Build 3D EEG electrode positions aligned with the given channel list.

    Reads the GSN HydroCel 129 SFP file (label, X, Y, Z in Cartesian
    head coordinates) and returns an ``(N, 3)`` float array of positions
    in the same order as ``channels.id``. Only channels whose type is
    ``"eeg"`` are looked up in the SFP file; all others receive NaN
    coordinates.

    Args:
        sfp_path: Path to the GSN HydroCel 129 SFP file.
        channels: ArrayDict with ``id`` and ``type``.

    Returns:
        Array of shape ``(N, 3)`` with XYZ coordinates in head space.
        Non-EEG channels and missing labels are filled with NaN.

    Raises:
        FileNotFoundError: If ``sfp_path`` does not exist.
        ValueError: If the SFP file does not have the expected format
            (four whitespace-separated columns: label, x, y, z).
    """
    if not sfp_path.exists():
        raise FileNotFoundError(
            f"Channel location file not found at {sfp_path}. Run download() first."
        )
    try:
        loc_df = pd.read_csv(
            sfp_path,
            sep=r"\s+",
            header=None,
            names=["label", "x", "y", "z"],
        )
    except pd.errors.ParserError as e:
        raise ValueError(
            f"Failed to parse channel location file at {sfp_path}: {e}"
        ) from e

    try:
        loc_df[["x", "y", "z"]] = loc_df[["x", "y", "z"]].astype(float)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Channel location file at {sfp_path} must have four "
            f"whitespace-separated columns (label, x, y, z) with numeric "
            f"coordinates: {e}"
        ) from e

    loc_map = {row.label: (row.x, row.y, row.z) for _, row in loc_df.iterrows()}

    locations = np.full((len(channels.id), 3), np.nan)
    for i, (ch, ch_type) in enumerate(zip(channels.id, channels.type, strict=True)):
        if ch_type == "eeg" and ch in loc_map:
            locations[i] = loc_map[ch]

    return locations


def create_splits(
    eeg_domain: Interval,
    subject_id: str,
    session_id: str,
    paradigm_intervals: Interval | None = None,
    epoch_duration: float = EPOCH_DURATION,
) -> Data:
    """Generate behavior-agnostic and behavior-relevant splits for one recording.

    Behavior-agnostic splits (no folds, train/valid only):
        - Intrasession: causal train/valid split on full domain.
        - Intersubject: deterministic subject-level train/valid assignment.
        - Intersession: deterministic session-level train/valid assignment.

    Behavior-relevant splits (only when paradigm annotations exist, with folds,
    train/valid/test):
        - Intrasession: stratified k-fold by task with train/valid/test.
        - Intersubject: per-fold subject-level train/valid/test assignment.
        - Intersession: per-fold session-level train/valid/test assignment.

    Args:
        eeg_domain: Full time domain of the EEG recording.
        subject_id: Subject identifier for cross-subject splits.
        session_id: Session identifier for cross-session splits.
        paradigm_intervals: Optional paradigm intervals with a ``task`` attribute.
        epoch_duration: Duration of each epoch in seconds.

    Returns:
        Data object with nested ``behavior_agnostic`` and optionally
        ``behavior_relevant`` sub-objects.
    """
    # ---- Behavior-agnostic splits (always present) ----
    all_epochs = eeg_domain.subdivide(step=epoch_duration, drop_short=True)
    logging.info(f"Behavior-agnostic: {len(all_epochs)} epochs of {epoch_duration}s")

    if len(all_epochs) < 2:  # train/valid splits only
        raise ValueError(
            f"Not enough epochs ({len(all_epochs)}) for behavior-agnostic "
            f"intrasession split. Need at least 2."
        )

    # Causal intrasession:
    n_train = max(1, int(len(all_epochs) * (1 - SPLIT_VAL_RATIO)))
    train_mask = np.zeros(len(all_epochs), dtype=bool)
    train_mask[:n_train] = True
    valid_mask = ~train_mask

    agnostic_intrasession = Data(
        train=all_epochs.select_by_mask(train_mask),
        valid=all_epochs.select_by_mask(valid_mask),
        domain=all_epochs,
    )

    intersubject_assignment = Pipeline.generate_string_binary_assignment(
        subject_id,
        val_ratio=SPLIT_VAL_RATIO,
        seed=SEED,
    )
    intersession_assignment = Pipeline.generate_string_binary_assignment(
        f"{subject_id}_{session_id}",
        val_ratio=SPLIT_VAL_RATIO,
        seed=SEED,
    )

    behavior_agnostic = Data(
        intrasession=agnostic_intrasession,
        intersubject_assignment=intersubject_assignment,
        intersession_assignment=intersession_assignment,
        domain=all_epochs,
    )

    splits_kwargs = {"behavior_agnostic": behavior_agnostic}

    # ---- Behavior-relevant splits (only when paradigms exist) ----
    if paradigm_intervals is not None and len(paradigm_intervals) > 0:
        paradigm_epochs = paradigm_intervals.subdivide(
            step=epoch_duration,
            drop_short=True,
        )
        logging.info(
            f"Behavior-relevant: {len(paradigm_epochs)} paradigm epochs "
            f"of {epoch_duration}s"
        )

        if len(paradigm_epochs) < SPLIT_N_FOLDS:
            logging.warning(
                f"Not enough paradigm epochs ({len(paradigm_epochs)}) for "
                f"{SPLIT_N_FOLDS} folds. Skipping behavior-relevant splits."
            )
        else:
            stratified_folds = generate_stratified_folds(
                paradigm_epochs,
                stratify_by="task",
                n_folds=SPLIT_N_FOLDS,
                val_ratio=SPLIT_VAL_RATIO,
                seed=SEED,
            )

            intrasession_kwargs = {"domain": paradigm_epochs}
            for i, fold in enumerate(stratified_folds):
                intrasession_kwargs[f"fold_{i}"] = fold

            behavior_relevant_kwargs: dict = {
                "intrasession": Data(**intrasession_kwargs),
                "domain": paradigm_epochs,
            }

            subject_assignments = generate_string_kfold_assignment(
                string_id=subject_id,
                n_folds=SPLIT_N_FOLDS,
                val_ratio=SPLIT_VAL_RATIO,
                seed=SEED,
            )
            session_assignments = generate_string_kfold_assignment(
                string_id=f"{subject_id}_{session_id}",
                n_folds=SPLIT_N_FOLDS,
                val_ratio=SPLIT_VAL_RATIO,
                seed=SEED,
            )

            for fold_idx, assignment in enumerate(subject_assignments):
                behavior_relevant_kwargs[f"intersubject_fold_{fold_idx}_assignment"] = (
                    assignment
                )
            for fold_idx, assignment in enumerate(session_assignments):
                behavior_relevant_kwargs[f"intersession_fold_{fold_idx}_assignment"] = (
                    assignment
                )

            splits_kwargs["behavior_relevant"] = Data(**behavior_relevant_kwargs)

    return Data(**splits_kwargs, domain=eeg_domain)
