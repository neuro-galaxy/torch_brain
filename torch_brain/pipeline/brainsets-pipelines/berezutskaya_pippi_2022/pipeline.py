# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne~=1.11.0",
#   "boto3~=1.41.0",
# ]
# ///

import datetime
import hashlib
import io
import json
import logging
import os
import re
import unicodedata
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd

from torch_brain.data import (
    ArrayDict,
    BrainsetDescription,
    Data,
    DeviceDescription,
    Interval,
    RegularTimeSeries,
    SessionDescription,
    SubjectDescription,
)
from torch_brain.data.pippi import (
    PIPPI_SUBSET_TIERS,
    pippi_subset_tiers_for_subject,
)
from torch_brain.pipeline import BrainsetPipeline

# MNE tries to create ~/.mne config state on import. In this sandbox that home
# location may be read-only, so force MNE to use a temporary writable home.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

from torch_brain.utils.mne import extract_measurement_date
from torch_brain.utils.s3 import get_cached_s3_client, get_object_list

try:
    import mne
except ImportError:  # pragma: no cover - guarded at runtime
    mne = None


logging.basicConfig(level=logging.INFO)

PIPELINE_DIR = Path(__file__).resolve().parent
BRAINSET_ID = "berezutskaya_pippi_2022"
DERIVED_VERSION = "1.1.1"
OPENNEURO_DATASET_ID = "ds003688"
OPENNEURO_VERSION = "1.0.7"
OPENNEURO_BUCKET = "openneuro.org"
OPENNEURO_PREFIX = f"{OPENNEURO_DATASET_ID}/"
WITHIN_SESSION_KEY = "within_session"
BRAIN_AREA_LABEL_COLUMNS = (
    "label_dkt",
    "label_destrieux",
)

VALID_TASKS = (
    "delta_volume",
    "face_num",
    "frame_brightness",
    "global_flow",
    "gpt2_surprisal",
    "local_flow",
    "onset",
    "pitch",
    "speech",
    "volume",
    "word_gap",
    "word_head_pos",
    "word_index",
    "word_length",
    "word_part_speech",
)

_RECORDING_ID_RE = re.compile(
    r"^(sub-(?P<subject>\d+)_ses-iemu_task-film_acq-(?P<acq>[A-Za-z0-9]+)_run-(?P<run>\d+))$"
)
_CHANNELS_REL_RE = re.compile(
    r"^(sub-\d+)/ses-iemu/ieeg/"
    r"(sub-\d+_ses-iemu_task-film_acq-[A-Za-z0-9]+_run-\d+)_channels\.tsv$"
)


class DownloadedAsset(NamedTuple):
    vhdr_path: Path
    recording_id: str
    acquisition: str
    subject_number: int
    run: int


parser = ArgumentParser(
    description="Prepare the Pippi movie iEEG dataset into processed brainsets files."
)
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--labels-dir",
    default=None,
    help=(
        "Directory containing label CSVs. Defaults to the labels packaged with "
        "this pipeline."
    ),
)
parser.add_argument(
    "--labels",
    default=None,
    help=(
        "Comma-separated list of label CSV filenames. If omitted, all CSVs in labels-dir are used."
    ),
)


def _resolve_labels_dir(labels_dir: str | Path | None) -> Path:
    """Resolve an explicit label directory or the pipeline's packaged labels."""
    using_packaged_default = labels_dir is None
    path = PIPELINE_DIR / "labels" if using_packaged_default else Path(labels_dir)
    source = "Packaged label directory" if using_packaged_default else "Label directory"
    if not path.is_dir():
        raise FileNotFoundError(f"{source} not found: '{path}'.")
    if not any(path.glob("*.csv")):
        raise FileNotFoundError(f"{source} contains no CSV files: '{path}'.")
    return path


parser.add_argument(
    "--pre-offset-s",
    type=float,
    default=0.0,
    help="Seconds before each label timestamp for the interval start.",
)
parser.add_argument(
    "--post-offset-s",
    type=float,
    default=1.0,
    help="Seconds after each label timestamp for the interval end.",
)
parser.add_argument(
    "--no-splits",
    action="store_true",
    help="Skip split generation; write only processed data.",
)
parser.add_argument(
    "--no-balance-splits",
    dest="balance_splits",
    action="store_false",
    default=True,
    help="Disable split-wise class balancing.",
)
parser.add_argument(
    "--balance-seed",
    type=int,
    default=0,
    help="Random seed used when split balancing is enabled.",
)


def _parse_recording_id(recording_id: str) -> tuple[str, int, int]:
    match = _RECORDING_ID_RE.fullmatch(recording_id)
    if match is None:
        raise ValueError(
            "Invalid Pippi recording_id "
            f"'{recording_id}'. Expected "
            "'sub-<subject>_ses-iemu_task-film_acq-<acq>_run-<run>'."
        )
    return match.group("acq"), int(match.group("subject")), int(match.group("run"))


def _build_recording_id(subject: int, acquisition: str, run: int) -> str:
    if (
        isinstance(subject, bool)
        or not isinstance(subject, int)
        or subject < 0
        or isinstance(run, bool)
        or not isinstance(run, int)
        or run < 1
        or not isinstance(acquisition, str)
        or not re.fullmatch(r"[A-Za-z0-9]+", acquisition)
    ):
        raise ValueError(
            "Invalid Pippi recording-id components: "
            f"subject={subject!r}, acquisition={acquisition!r}, run={run!r}."
        )
    return f"sub-{subject:02d}_ses-iemu_task-film_acq-{acquisition}_run-{run}"


def split_selector_key(
    *,
    subset_tier: str,
    label_mode: str,
    task_name: str,
    fold_idx: int,
    split_name: str,
) -> str:
    if subset_tier not in PIPPI_SUBSET_TIERS:
        raise ValueError(
            f"Invalid Pippi subset_tier '{subset_tier}'. Must be one of {PIPPI_SUBSET_TIERS}."
        )
    return (
        f"{subset_tier}${label_mode}${WITHIN_SESSION_KEY}${task_name}$"
        f"fold{fold_idx}${split_name}"
    )


def _sanitize_attr_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _normalize_ascii_string(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    # Common iEEG metadata uses micro-sign units like "µV", which temporaldata
    # cannot serialize as fixed-length ASCII strings.
    text = text.replace("µ", "u").replace("μ", "u")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _parse_task_and_label_mode(label_file: str) -> tuple[str, str]:
    name = Path(label_file).name
    match = re.fullmatch(r"(.+)_(binary|multiclass)_labels\.csv", name)
    if match is None:
        raise ValueError(
            "Label file naming must encode mode as "
            "'<task>_binary_labels.csv' or '<task>_multiclass_labels.csv'. "
            f"Got: {label_file}"
        )
    task_name, mode = match.group(1), match.group(2)
    if task_name not in VALID_TASKS:
        raise ValueError(f"Unsupported task '{task_name}' from {label_file}.")
    return task_name, mode


def _load_label_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "start_time_s" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'start_time_s' and 'label' in {csv_path}")
    return (
        df["start_time_s"].astype(float).to_numpy(),
        pd.to_numeric(df["label"], errors="raise").to_numpy(),
    )


def _labels_for_mode(labels: np.ndarray, *, mode: str, csv_path: Path) -> np.ndarray:
    labels_int = labels.astype(np.int64)
    if mode == "binary":
        unique = set(int(x) for x in np.unique(labels_int))
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Binary label file contains non-binary labels {sorted(unique)}: {csv_path}"
            )
    return labels_int


def _build_windows(
    times: np.ndarray,
    labels: np.ndarray,
    *,
    pre_offset_s: float,
    post_offset_s: float,
    domain_start: float,
    domain_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = times - pre_offset_s
    end = times + post_offset_s
    valid = np.logical_and(start >= domain_start, end <= domain_end)
    return (
        start[valid].astype(np.float64),
        end[valid].astype(np.float64),
        labels[valid],
    )


def _balanced_binary_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.arange(len(labels))
    n_keep = min(len(pos_idx), len(neg_idx))
    pos_keep = rng.choice(pos_idx, size=n_keep, replace=False)
    neg_keep = rng.choice(neg_idx, size=n_keep, replace=False)
    keep = np.concatenate([pos_keep, neg_keep])
    rng.shuffle(keep)
    return keep


def _balanced_multiclass_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) <= 1:
        return np.arange(len(labels))
    class_indices = [np.where(labels == class_id)[0] for class_id in classes]
    n_keep = min(len(indices) for indices in class_indices)
    keep = np.concatenate(
        [rng.choice(indices, size=n_keep, replace=False) for indices in class_indices]
    )
    rng.shuffle(keep)
    return keep


def _balance_interval(interval: Interval, rng: np.random.Generator) -> Interval:
    labels = np.asarray(interval.label)
    if np.all(np.isin(np.unique(labels), [0, 1])):
        keep = _balanced_binary_indices(labels, rng)
    else:
        keep = _balanced_multiclass_indices(labels, rng)
    return Interval(
        start=interval.start[keep],
        end=interval.end[keep],
        label=labels[keep],
    )


def _stable_balance_rng(
    *,
    balance_seed: int,
    recording_id: str,
    task_name: str,
    label_mode: str,
    fold_idx: int,
    split_name: str,
) -> np.random.Generator:
    key = "|".join(
        [
            "berezutskaya_pippi_2022",
            str(int(balance_seed)),
            recording_id,
            task_name,
            label_mode,
            str(int(fold_idx)),
            split_name,
        ]
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _split_twofold(
    start: np.ndarray,
    end: np.ndarray,
    labels: np.ndarray,
    *,
    balance: bool = False,
    rng: np.random.Generator | None = None,
    rng_factory: Callable[[int, str], np.random.Generator] | None = None,
) -> dict[int, dict[str, Interval]]:
    order = np.argsort(start)
    start = start[order]
    end = end[order]
    labels = labels[order]

    n = len(start)
    mid = n // 2
    folds: dict[int, dict[str, Interval]] = {}
    for fold_idx, (t0, t1) in enumerate(((0, mid), (mid, n))):
        test_indices = np.arange(t0, t1)
        train_indices = np.setdiff1d(np.arange(n), test_indices)
        val_split = t0 + (len(test_indices) // 2)
        val_indices = np.arange(t0, val_split)
        final_test_indices = np.arange(val_split, t1)

        fold = {
            "train": Interval(
                start=start[train_indices],
                end=end[train_indices],
                label=labels[train_indices],
            ),
            "val": Interval(
                start=start[val_indices],
                end=end[val_indices],
                label=labels[val_indices],
            ),
            "test": Interval(
                start=start[final_test_indices],
                end=end[final_test_indices],
                label=labels[final_test_indices],
            ),
        }
        if balance:
            balanced_fold = {}
            for split_name, split_interval in fold.items():
                split_rng = (
                    rng_factory(fold_idx, split_name)
                    if rng_factory is not None
                    else rng
                )
                if split_rng is None:
                    raise ValueError("balance=True requires an RNG or rng_factory")
                balanced_fold[split_name] = _balance_interval(split_interval, split_rng)
            fold = balanced_fold
        folds[fold_idx] = fold
    return folds


def _build_label_maps(
    label_tasks: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, str]]]:
    maps: dict[str, dict[str, dict[str, str]]] = {"multiclass": {}}
    for task_name, labels in label_tasks.items():
        unique = sorted(set(int(x) for x in np.unique(labels)))
        maps["multiclass"][task_name] = {str(v): str(v) for v in unique}
    return maps


def _recording_relative_paths(
    recording_id: str,
    *,
    acquisition: str | None = None,
    subject_number: int | None = None,
    run: int | None = None,
) -> dict[str, str]:
    if acquisition is None or subject_number is None or run is None:
        acquisition, subject_number, run = _parse_recording_id(recording_id)
    subject_id = f"sub-{subject_number:02d}"
    base_dir = f"{subject_id}/ses-iemu/ieeg"
    base_name = f"{recording_id}_ieeg"
    event_name = f"{subject_id}_ses-iemu_task-film_run-{run}_events.tsv"
    prefix = f"{subject_id}_ses-iemu_acq-{acquisition}"
    return {
        "participants": "participants.tsv",
        "channels": f"{base_dir}/{recording_id}_channels.tsv",
        "vhdr": f"{base_dir}/{base_name}.vhdr",
        "vmrk": f"{base_dir}/{base_name}.vmrk",
        "eeg": f"{base_dir}/{base_name}.eeg",
        "ieeg_json": f"{base_dir}/{base_name}.json",
        "events": f"{base_dir}/{event_name}",
        "electrodes": f"{base_dir}/{prefix}_electrodes.tsv",
        "coordsystem": f"{base_dir}/{prefix}_coordsystem.json",
    }


def _build_manifest_row(
    recording_id: str, *, available_relpaths: set[str]
) -> dict[str, object]:
    relpaths = _recording_relative_paths(recording_id)
    # participants.tsv is optional; subject metadata builder already tolerates
    # missing participants information.
    missing = sorted(
        path
        for name, path in relpaths.items()
        if name != "participants" and path not in available_relpaths
    )
    if missing:
        raise ValueError(f"Missing required sidecars for {recording_id}: {missing}")
    acquisition, subject_number, run = _parse_recording_id(recording_id)
    row: dict[str, object] = {
        "recording_id": recording_id,
        "subject_id": f"sub-{subject_number:02d}",
        "test_subject": subject_number,
        "test_run": run,
        "acquisition": acquisition,
    }
    for name, relpath in relpaths.items():
        if name == "participants" and relpath not in available_relpaths:
            row[f"{name}_relpath"] = None
            continue
        row[f"{name}_relpath"] = relpath
    return row


def _read_remote_tsv(s3_client, relpath: str) -> pd.DataFrame:
    response = s3_client.get_object(
        Bucket=OPENNEURO_BUCKET,
        Key=f"{OPENNEURO_PREFIX}{relpath}",
    )
    payload = response["Body"].read()
    return pd.read_csv(io.BytesIO(payload), sep="\t")


def _channel_table_has_seeg(channel_table: pd.DataFrame) -> bool:
    if "type" not in channel_table.columns:
        raise ValueError("Channel table missing required 'type' column.")
    return channel_table["type"].astype(str).str.upper().eq("SEEG").any()


def _discover_local_manifest_rows(raw_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    available = {
        path.relative_to(raw_dir).as_posix()
        for path in raw_dir.rglob("*")
        if path.is_file()
    }
    for channel_path in sorted(
        raw_dir.glob("sub-*/ses-iemu/ieeg/*task-film*_channels.tsv")
    ):
        channel_table = pd.read_csv(channel_path, sep="\t")
        if not _channel_table_has_seeg(channel_table):
            continue
        recording_id = channel_path.name.removesuffix("_channels.tsv")
        rows.append(_build_manifest_row(recording_id, available_relpaths=available))
    return rows


def _discover_remote_manifest_rows(raw_dir: Path) -> list[dict[str, object]]:
    s3_client = get_cached_s3_client()
    available = set(
        get_object_list(OPENNEURO_BUCKET, OPENNEURO_PREFIX, s3_client=s3_client)
    )
    rows: list[dict[str, object]] = []
    for relpath in sorted(available):
        match = _CHANNELS_REL_RE.fullmatch(relpath)
        if match is None:
            continue
        channel_table = _read_remote_tsv(s3_client, relpath)
        if not _channel_table_has_seeg(channel_table):
            continue
        recording_id = match.group(2)
        rows.append(_build_manifest_row(recording_id, available_relpaths=available))
    return rows


def _download_relpath(
    s3_client,
    *,
    relpath: str,
    target_root: Path,
    overwrite: bool,
) -> Path:
    local_path = target_root / relpath
    if local_path.exists() and not overwrite:
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(
        OPENNEURO_BUCKET,
        f"{OPENNEURO_PREFIX}{relpath}",
        str(local_path),
    )
    return local_path


def _extract_task_bounds(events_df: pd.DataFrame) -> tuple[float, float]:
    if "trial_type" not in events_df.columns or "onset" not in events_df.columns:
        raise ValueError("Events table must contain 'trial_type' and 'onset'.")
    start_rows = events_df.loc[
        events_df["trial_type"].astype(str) == "start task", "onset"
    ]
    end_rows = events_df.loc[events_df["trial_type"].astype(str) == "end task", "onset"]
    if len(start_rows) != 1 or len(end_rows) != 1:
        raise ValueError("Expected exactly one 'start task' and one 'end task' event.")
    task_start = float(start_rows.iloc[0])
    task_end = float(end_rows.iloc[0])
    if (
        not np.isfinite(task_start)
        or not np.isfinite(task_end)
        or task_end <= task_start
    ):
        raise ValueError(f"Invalid task bounds: start={task_start}, end={task_end}.")
    return task_start, task_end


def _read_raw_brainvision(vhdr_path: Path):
    if mne is None:
        raise ImportError(
            "The Pippi movie pipeline requires mne. Install it or run through "
            "brainsets prepare so inline dependencies are honored."
        )
    return mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")


def _load_seeg_signal(
    vhdr_path: Path,
    channel_table: pd.DataFrame,
    *,
    task_start_s: float,
    task_end_s: float,
) -> tuple[RegularTimeSeries, list[str]]:
    raw = _read_raw_brainvision(vhdr_path)
    seeg_rows = channel_table.loc[
        channel_table["type"].astype(str).str.upper() == "SEEG"
    ].copy()
    if seeg_rows.empty:
        raise ValueError(f"No SEEG channels found for {vhdr_path}.")

    seeg_names = seeg_rows["name"].astype(str).tolist()
    missing = [name for name in seeg_names if name not in raw.ch_names]
    if missing:
        raise ValueError(
            f"SEEG channels missing from BrainVision data for {vhdr_path.name}: {missing}"
        )

    signal = raw.get_data(picks=seeg_names).T
    timestamps = raw.times.astype(np.float64)
    keep = np.logical_and(timestamps >= task_start_s, timestamps <= task_end_s)
    if not np.any(keep):
        raise ValueError(
            f"Task crop produced zero samples for {vhdr_path.name}: "
            f"task_start_s={task_start_s}, task_end_s={task_end_s}."
        )

    movie_timestamps = timestamps[keep] - task_start_s
    seeg_data = RegularTimeSeries(
        data=signal[keep].astype(np.float32),
        sampling_rate=float(raw.info["sfreq"]),
        domain="auto",
        domain_start=float(movie_timestamps[0]),
    )
    return seeg_data, seeg_names


def _load_brain_area_labels() -> pd.DataFrame:
    path = PIPELINE_DIR / "brain_areas" / "brain_area_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required brain area label CSV: {path}")

    try:
        labels = pd.read_csv(path, dtype={"subject": str, "electrode": str})
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Brain area label CSV is empty: {path}") from exc
    required = {"subject", "electrode", *BRAIN_AREA_LABEL_COLUMNS}
    missing_columns = sorted(required.difference(labels.columns))
    if missing_columns:
        raise ValueError(
            f"Brain area label CSV missing required columns: {missing_columns}"
        )
    if labels.empty:
        raise ValueError(f"Brain area label CSV contains no rows: {path}")

    labels = labels.copy()
    labels["subject"] = labels["subject"].astype(str).str.zfill(2)
    labels["electrode"] = labels["electrode"].astype(str)
    duplicates = labels.duplicated(["subject", "electrode"], keep=False)
    if duplicates.any():
        duplicate_keys = (
            labels.loc[duplicates, ["subject", "electrode"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate brain area labels for {duplicate_keys}")
    return labels


def _add_brain_area_labels(
    merged: pd.DataFrame,
    *,
    subject_number: int | None,
) -> pd.DataFrame:
    if subject_number is None:
        return merged

    labels = _load_brain_area_labels()

    subject = f"{subject_number:02d}"
    subject_labels = labels.loc[labels["subject"] == subject].copy()
    enriched = merged.copy()
    if subject_labels.empty:
        for column in BRAIN_AREA_LABEL_COLUMNS:
            enriched[column] = ""
        return enriched

    enriched["__channel_order"] = np.arange(len(enriched), dtype=int)
    subject_labels = subject_labels.rename(columns={"electrode": "name"})
    enriched = enriched.merge(
        subject_labels[["name", *BRAIN_AREA_LABEL_COLUMNS]],
        on="name",
        how="left",
        sort=False,
    ).sort_values("__channel_order")
    enriched = enriched.drop(columns="__channel_order").reset_index(drop=True)

    if (
        len(enriched) != len(merged)
        or enriched["name"].tolist() != merged["name"].tolist()
    ):
        raise ValueError(
            "Brain area label merge changed channel order or channel count."
        )

    missing = enriched[list(BRAIN_AREA_LABEL_COLUMNS)].isna().any(axis=1)
    if missing.any():
        missing_names = enriched.loc[missing, "name"].astype(str).tolist()
        raise ValueError(
            f"Missing brain area labels for subject {subject}: {missing_names}"
        )
    return enriched


def _build_channels(
    channel_table: pd.DataFrame,
    electrodes_table: pd.DataFrame,
    *,
    seeg_names: list[str],
    subject_number: int | None = None,
) -> ArrayDict:
    seeg_table = channel_table.loc[
        channel_table["name"].astype(str).isin(seeg_names)
    ].copy()
    seeg_table["name"] = seeg_table["name"].astype(str)
    electrodes_table = electrodes_table.copy()
    electrodes_table["name"] = electrodes_table["name"].astype(str)

    merged = seeg_table.merge(
        electrodes_table,
        on="name",
        how="left",
        suffixes=("", "_electrode"),
        sort=False,
    )
    if merged[["x", "y", "z"]].isna().any().any():
        raise ValueError(
            "Missing x/y/z electrode coordinates for one or more SEEG channels."
        )
    merged = _add_brain_area_labels(merged, subject_number=subject_number)

    included = np.logical_not(
        merged.get("status", pd.Series(["good"] * len(merged)))
        .astype(str)
        .str.lower()
        .eq("bad")
    )
    x = merged["x"].to_numpy(dtype=np.float32)
    y = merged["y"].to_numpy(dtype=np.float32)
    z = merged["z"].to_numpy(dtype=np.float32)
    channels = ArrayDict(
        id=np.arange(len(merged), dtype=int),
        name=merged["name"].to_numpy(dtype="U"),
        included=included.to_numpy(dtype=bool),
        type=np.ones(len(merged), dtype=int) * 31,
        x=x,
        y=y,
        z=z,
        coord_acpc_x=x.copy(),
        coord_acpc_y=y.copy(),
        coord_acpc_z=z.copy(),
    )

    reserved = {
        "id",
        "name",
        "included",
        "type",
        "x",
        "y",
        "z",
        "coord_acpc_x",
        "coord_acpc_y",
        "coord_acpc_z",
    }
    for column in merged.columns:
        safe_name = _sanitize_attr_name(column)
        if safe_name in reserved:
            continue
        values = merged[column]
        if pd.api.types.is_numeric_dtype(values):
            setattr(channels, safe_name, values.to_numpy(dtype=np.float32))
        else:
            setattr(
                channels,
                safe_name,
                values.map(_normalize_ascii_string).to_numpy(dtype="U"),
            )
    return channels


def _build_subject_description(raw_root: Path, subject_id: str) -> SubjectDescription:
    participants_path = raw_root / "participants.tsv"
    if not participants_path.exists():
        return SubjectDescription(id=subject_id, species="HOMO_SAPIENS")

    participants = pd.read_csv(participants_path, sep="\t")
    rows = participants.loc[participants["participant_id"].astype(str) == subject_id]
    if rows.empty:
        return SubjectDescription(id=subject_id, species="HOMO_SAPIENS")

    row = rows.iloc[0]
    age = None
    if "age" in row and pd.notna(row["age"]):
        age = float(row["age"])
    sex = None
    if "sex" in row and pd.notna(row["sex"]):
        sex = str(row["sex"])
    return SubjectDescription(
        id=subject_id,
        species="HOMO_SAPIENS",
        age=age,
        sex=sex,
    )


def _build_session_description(
    *,
    raw,
    recording_id: str,
) -> SessionDescription:
    try:
        recording_date = extract_measurement_date(raw)
    except Exception:
        recording_date = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return SessionDescription(
        id=recording_id,
        recording_date=recording_date,
        task=7,
    )


def _build_device_description(
    ieeg_metadata: dict,
    *,
    acquisition: str,
) -> DeviceDescription:
    manufacturer = ieeg_metadata.get("Manufacturer")
    device_id = f"{manufacturer}_{acquisition}" if manufacturer else acquisition
    return DeviceDescription(
        id=device_id,
        recording_tech="STEREO_EEG",
        processing="brainvision_raw_task_crop",
    )


def _get_brainset_description() -> BrainsetDescription:
    return BrainsetDescription(
        id=BRAINSET_ID,
        origin_version=f"{OPENNEURO_DATASET_ID}:{OPENNEURO_VERSION}",
        derived_version=DERIVED_VERSION,
        source=(
            f"https://openneuro.org/datasets/{OPENNEURO_DATASET_ID}/versions/{OPENNEURO_VERSION}"
        ),
        description=(
            "Human intracranial movie-watching recordings from the Pippi film dataset, "
            "prepared as SEEG-only continuous recordings with BYD-style label intervals."
        ),
    )


def process_file(
    vhdr_path: Path,
    output_dir: Path,
    *,
    labels_dir: Path,
    label_files: Iterable[str],
    pre_offset_s: float,
    post_offset_s: float,
    no_splits: bool = False,
    balance_splits: bool = False,
    balance_seed: int = 0,
    recording_id: str | None = None,
    acquisition: str | None = None,
    subject_number: int | None = None,
    run: int | None = None,
) -> Path:
    if recording_id is None:
        recording_id = vhdr_path.name.removesuffix("_ieeg.vhdr")
    if acquisition is None or subject_number is None or run is None:
        parsed_acquisition, parsed_subject_number, parsed_run = _parse_recording_id(
            recording_id
        )
        if acquisition is None:
            acquisition = parsed_acquisition
        if subject_number is None:
            subject_number = parsed_subject_number
        if run is None:
            run = parsed_run
    raw_root = vhdr_path.parents[3]
    relpaths = _recording_relative_paths(
        recording_id,
        acquisition=acquisition,
        subject_number=subject_number,
        run=run,
    )

    channel_table = pd.read_csv(raw_root / relpaths["channels"], sep="\t")
    events_df = pd.read_csv(raw_root / relpaths["events"], sep="\t")
    electrodes_table = pd.read_csv(raw_root / relpaths["electrodes"], sep="\t")
    with (raw_root / relpaths["ieeg_json"]).open("r", encoding="utf-8") as handle:
        ieeg_metadata = json.load(handle)

    task_start_s, task_end_s = _extract_task_bounds(events_df)
    seeg_data, seeg_names = _load_seeg_signal(
        vhdr_path,
        channel_table,
        task_start_s=task_start_s,
        task_end_s=task_end_s,
    )
    channels = _build_channels(
        channel_table,
        electrodes_table,
        seeg_names=seeg_names,
        subject_number=subject_number,
    )

    raw = _read_raw_brainvision(vhdr_path)
    subject_id = f"sub-{subject_number:02d}"
    data = Data(
        brainset=_get_brainset_description(),
        subject=_build_subject_description(raw_root, subject_id),
        session=_build_session_description(raw=raw, recording_id=recording_id),
        device=_build_device_description(ieeg_metadata, acquisition=acquisition),
        seeg_data=seeg_data,
        channels=channels,
        domain=seeg_data.domain,
    )

    data.alignment_version = "1.0.0"
    data.alignment_reference = "events.start task/end task"
    data.alignment_method = "crop_to_film_interval_then_zero_to_movie_time"
    data.alignment_applied_at_prepare = True
    data.alignment_parameters_json = json.dumps(
        {
            "task_start_s": float(task_start_s),
            "task_end_s": float(task_end_s),
            "test_run": run,
        },
        sort_keys=True,
    )

    if not no_splits:
        label_names = (
            list(label_files)
            if label_files
            else sorted(p.name for p in labels_dir.glob("*.csv"))
        )
        if not label_names:
            raise ValueError(f"No label CSVs found in labels-dir '{labels_dir}'.")

        domain_start = float(seeg_data.domain.start[0])
        domain_end = float(seeg_data.domain.end[0])
        data.splits = Data(domain=seeg_data.domain)
        data.channel_splits = Data()
        label_tasks_for_maps: dict[str, np.ndarray] = {}
        seen_mode_task_pairs: set[tuple[str, str]] = set()

        for label_name in label_names:
            task_name, label_mode = _parse_task_and_label_mode(label_name)
            mode_task = (label_mode, task_name)
            if mode_task in seen_mode_task_pairs:
                raise ValueError(
                    "Duplicate task/mode label files would overwrite split attributes: "
                    f"mode={label_mode}, task={task_name}"
                )
            seen_mode_task_pairs.add(mode_task)

            label_times, labels = _load_label_csv(labels_dir / label_name)
            start, end, labels = _build_windows(
                label_times,
                labels,
                pre_offset_s=pre_offset_s,
                post_offset_s=post_offset_s,
                domain_start=domain_start,
                domain_end=domain_end,
            )
            labels_for_mode = _labels_for_mode(
                labels,
                mode=label_mode,
                csv_path=labels_dir / label_name,
            )
            rng_factory = None
            if balance_splits:

                def rng_factory(
                    fold_idx: int,
                    split_name: str,
                    *,
                    task_name: str = task_name,
                    label_mode: str = label_mode,
                ) -> np.random.Generator:
                    return _stable_balance_rng(
                        balance_seed=balance_seed,
                        recording_id=recording_id,
                        task_name=task_name,
                        label_mode=label_mode,
                        fold_idx=fold_idx,
                        split_name=split_name,
                    )

            folds = _split_twofold(
                start,
                end,
                labels_for_mode,
                balance=balance_splits,
                rng_factory=rng_factory,
            )
            # Each recording always participates in the full benchmark tier and
            # optionally in one of the coverage-based subject subsets.
            subset_tiers = pippi_subset_tiers_for_subject(subject_number)
            for fold_idx, fold in folds.items():
                for subset_tier in subset_tiers:
                    for split_name in ("train", "val", "test"):
                        selector_key = split_selector_key(
                            subset_tier=subset_tier,
                            label_mode=label_mode,
                            task_name=task_name,
                            fold_idx=fold_idx,
                            split_name=split_name,
                        )
                        setattr(data.splits, selector_key, fold[split_name])
                        setattr(
                            data.channel_splits,
                            selector_key,
                            channels.included.astype(bool),
                        )

            if label_mode == "multiclass":
                label_tasks_for_maps[task_name] = labels_for_mode

        if label_tasks_for_maps:
            data.label_maps_json = json.dumps(
                _build_label_maps(label_tasks_for_maps),
                sort_keys=True,
            )

    output_path = output_dir / f"{recording_id}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with h5py.File(tmp_output_path, "w") as handle:
            data.to_hdf5(handle)
            handle["seeg_data"].attrs["unit"] = "V"
            handle["seeg_data"].attrs["scale_to_uV"] = 1e6
        tmp_output_path.replace(output_path)
    except Exception:
        tmp_output_path.unlink(missing_ok=True)
        raise
    return output_path


class Pipeline(BrainsetPipeline):
    brainset_id = BRAINSET_ID
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Namespace | None,
    ) -> pd.DataFrame:
        raw_dir.mkdir(exist_ok=True, parents=True)
        local_rows = _discover_local_manifest_rows(raw_dir)
        rows_by_id: dict[str, dict[str, object]] = {
            str(row["recording_id"]): row for row in local_rows
        }

        remote_rows: list[dict[str, object]] = []
        try:
            remote_rows = _discover_remote_manifest_rows(raw_dir)
        except Exception as exc:
            # If local discovery already found eligible recordings, keep moving.
            if not rows_by_id:
                raise
            logging.warning(
                "Remote manifest discovery failed; using local rows only: %s",
                exc,
            )

        for row in remote_rows:
            # Prefer local rows when recording ids overlap, but include any
            # additional eligible remote recordings not present locally.
            rows_by_id.setdefault(str(row["recording_id"]), row)

        rows = list(rows_by_id.values())
        if not rows:
            raise ValueError(
                f"No eligible Pippi SEEG film recordings found under {raw_dir} "
                "or from the configured OpenNeuro source."
            )
        manifest = pd.DataFrame(rows).set_index("recording_id").sort_index()
        return manifest

    def download(self, manifest_item: NamedTuple) -> DownloadedAsset:
        self.update_status("DOWNLOADING")
        vhdr_path = self.raw_dir / manifest_item.vhdr_relpath
        recording_id = getattr(
            manifest_item,
            "recording_id",
            getattr(manifest_item, "Index", vhdr_path.name.removesuffix("_ieeg.vhdr")),
        )
        download_output = DownloadedAsset(
            vhdr_path=vhdr_path,
            recording_id=recording_id,
            acquisition=str(manifest_item.acquisition),
            subject_number=int(manifest_item.test_subject),
            run=int(manifest_item.test_run),
        )
        if vhdr_path.exists() and not (self.args and self.args.redownload):
            return download_output

        s3_client = get_cached_s3_client()
        overwrite = bool(self.args and self.args.redownload)
        for field_name in (
            "participants_relpath",
            "channels_relpath",
            "vhdr_relpath",
            "vmrk_relpath",
            "eeg_relpath",
            "ieeg_json_relpath",
            "events_relpath",
            "electrodes_relpath",
            "coordsystem_relpath",
        ):
            relpath = getattr(manifest_item, field_name, None)
            if relpath is None or pd.isna(relpath):
                if field_name == "participants_relpath":
                    continue
                raise ValueError(
                    f"Manifest item missing required field '{field_name}' for "
                    f"recording '{recording_id}'."
                )
            _download_relpath(
                s3_client,
                relpath=str(relpath),
                target_root=self.raw_dir,
                overwrite=overwrite,
            )
        return download_output

    def process(self, download_output: DownloadedAsset | Path):
        self.update_status("PROCESSING")
        if isinstance(download_output, Path):
            vhdr_path = download_output
            recording_id = vhdr_path.name.removesuffix("_ieeg.vhdr")
            acquisition, subject_number, run = _parse_recording_id(recording_id)
        else:
            vhdr_path = download_output.vhdr_path
            recording_id = download_output.recording_id
            acquisition = download_output.acquisition
            subject_number = download_output.subject_number
            run = download_output.run

        output_path = self.processed_dir / f"{recording_id}.h5"
        if output_path.exists() and not (self.args and self.args.reprocess):
            logging.info(f"Skipping processing for {output_path} because it exists")
            self.update_status("Skipped Processing")
            return

        label_files = None
        if self.args and self.args.labels:
            label_files = [x.strip() for x in self.args.labels.split(",") if x.strip()]

        labels_dir = _resolve_labels_dir(self.args.labels_dir if self.args else None)
        process_file(
            vhdr_path=vhdr_path,
            output_dir=self.processed_dir,
            labels_dir=labels_dir,
            label_files=label_files or (),
            pre_offset_s=self.args.pre_offset_s if self.args else 0.0,
            post_offset_s=self.args.post_offset_s if self.args else 1.0,
            no_splits=bool(self.args and self.args.no_splits),
            balance_splits=self.args.balance_splits if self.args else True,
            balance_seed=self.args.balance_seed if self.args else 0,
            recording_id=recording_id,
            acquisition=acquisition,
            subject_number=subject_number,
            run=run,
        )
