# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "dandi==0.74.3",
#   "pynwb==2.6.0",
#   "hdmf==3.12.2",
#   "h5py==3.11.0",
#   "numpy==1.26.4",
#   "pandas==2.2.2",
# ]
# ///

import datetime
import hashlib
import json
import logging
import re
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

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
from torch_brain.pipeline import BrainsetPipeline
from torch_brain.utils.dandi import (
    download_file,
    extract_subject_from_nwb,
    get_nwb_asset_list,
)

logging.basicConfig(level=logging.INFO)

# Context margin retained around encoding after alignment.
# This controls how much extra signal to keep before/after the task interval.
EDGE_OFFSET_S = 10.0
# Timebase correction (BYD-specific): raw SEEG timestamps include ~10s pre-movie pad.
# Subtract this to map neural timestamps onto movie/label time.
# Keep separate from EDGE_OFFSET_S: same numeric default today, different purpose.
LFP_MOVIE_PAD_S = 10.0

KEEP_CHANNEL_AREAS = {"AMY", "HIP", "ACC", "SMA", "OFC"}

ALIGNMENT_VERSION = "2.0.0"
ALIGNMENT_REFERENCE = "nwb.trials.stim_phase=encoding"
ALIGNMENT_METHOD = "fixed_pad_shift_then_encoding_window_crop"

SUBSET_TIER = "full"
WITHIN_SESSION_KEY = "within_session"
_RAW_RECORDING_STEM_RE = re.compile(
    r"^(sub-CS(?P<subject>\d+)_ses-P(?P<p_subject>\d+)CSR(?P<session>\d+))"
    r"(?:_behavior\+ecephys)?$"
)


DANDISET_ID = "DANDI:000623"
PIPELINE_DIR = Path(__file__).resolve().parent
BRAIN_AREA_LABEL_COLUMNS = (
    "label_dkt",
    "label_destrieux",
)
BRAIN_AREA_CONTEXT_COLUMNS = (
    "provided_location",
    "trajectory_role",
    "coordinate_assumption",
    "label_status",
    "qc_status",
)
BRAIN_AREA_MERGE_COLUMNS = BRAIN_AREA_LABEL_COLUMNS + BRAIN_AREA_CONTEXT_COLUMNS
_BRAIN_AREA_COORD_KEY_COLUMNS = ("__mni_x_key", "__mni_y_key", "__mni_z_key")

parser = ArgumentParser(
    description=(
        "Prepare BYD NWB sessions into processed brainsets files. "
        "Alignment contract: generated split timestamps are already aligned "
        "to the stored SEEG timeline."
    ),
    epilog=(
        "For aligned BYD processed outputs, downstream "
        "'processed_split_time_shift_seconds' is deprecated."
    ),
)
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--hrefs-file",
    default=None,
    help=(
        "Optional href list to filter manifest. If relative, resolved against the pipeline dir."
    ),
)
parser.add_argument(
    "--labels-dir",
    required=True,
    help="Directory containing label CSVs.",
)
parser.add_argument(
    "--labels",
    default=None,
    help=(
        "Comma-separated list of label CSV filenames. If omitted, all CSVs in labels-dir are used."
    ),
)
parser.add_argument(
    "--pre-offset-s",
    type=float,
    default=0.0,
    help=(
        "Seconds before aligned label time for window start. "
        "Label/movie time is aligned to SEEG during prepare."
    ),
)
parser.add_argument(
    "--post-offset-s",
    type=float,
    default=1.0,
    help=(
        "Seconds after aligned label time for window end. "
        "Label/movie time is aligned to SEEG during prepare."
    ),
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


def _read_hrefs(hrefs_path: Path) -> set[str]:
    hrefs = set()
    with hrefs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            href = line.strip()
            if not href or href.startswith("#"):
                continue
            hrefs.add(href)
    return hrefs


class Pipeline(BrainsetPipeline):
    """Brainsets pipeline for the Keles BYD 2024 NWB dataset."""

    brainset_id = "keles_byd_2024"
    parser = parser

    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Namespace | None,
    ) -> pd.DataFrame:
        """Enumerate all NWB assets from DANDI and build the manifest table."""
        raw_dir.mkdir(exist_ok=True, parents=True)

        hrefs_filter = None
        if args is not None and args.hrefs_file:
            hrefs_path = Path(args.hrefs_file)
            if not hrefs_path.is_absolute():
                hrefs_path = PIPELINE_DIR / hrefs_path
            hrefs_filter = _read_hrefs(hrefs_path)

        asset_list = get_nwb_asset_list(DANDISET_ID)
        manifest_list = []
        for asset in asset_list:
            if hrefs_filter is not None and asset.path not in hrefs_filter:
                continue
            nwb_path = Path(asset.path)
            stem = nwb_path.stem
            # Example stem: sub-CS48_ses-P48CSR1_behavior+ecephys
            session_id = stem.split("_behavior")[0] if "_behavior" in stem else stem
            subject_id = session_id.split("_")[0]
            manifest_list.append(
                {
                    "path": asset.path,
                    "url": asset.download_url,
                    "session_id": session_id,
                    "subject_id": subject_id,
                    "stem": stem,
                }
            )

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        """Download a single NWB asset from DANDI into the raw directory."""
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        # DANDI download, skip if present unless --redownload
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=bool(self.args and self.args.redownload),
        )
        return fpath

    def process(self, fpath):
        """Process a single NWB file into a standardized brainsets HDF5 output."""
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        recording_id, _, _ = _parse_recording_id_from_input_file(str(fpath))
        output_path = self.processed_dir / f"{recording_id}.h5"
        if output_path.exists() and not (self.args and self.args.reprocess):
            logging.info(f"Skipping processing for {output_path} because it exists")
            self.update_status("Skipped Processing")
            return

        logging.info(f"Processing {fpath} to {self.processed_dir}")

        label_files = None
        if self.args and self.args.labels:
            label_files = [x.strip() for x in self.args.labels.split(",") if x.strip()]

        process_file(
            str(fpath),
            str(self.processed_dir),
            labels_dir=str(self.args.labels_dir) if self.args else None,
            label_files=label_files,
            pre_offset_s=self.args.pre_offset_s if self.args else 0.0,
            post_offset_s=self.args.post_offset_s if self.args else 1.0,
            no_splits=bool(self.args and self.args.no_splits),
            balance_splits=self.args.balance_splits if self.args else True,
            balance_seed=self.args.balance_seed if self.args else 0,
        )


def _compute_included_mask(electrode_df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask based on origchannel_name region codes (BYD convention)."""
    if "origchannel_name" not in electrode_df.columns:
        raise ValueError("Electrode table missing origchannel_name for inclusion mask.")

    names = electrode_df["origchannel_name"].astype(str)
    # Match prep_filterLFP.py: use substring [1:4] for region code
    areas = names.str.slice(1, 4).str.upper()
    return areas.isin(KEEP_CHANNEL_AREAS).to_numpy(dtype=bool)


def _sanitize_attr_name(name: str) -> str:
    """Normalize metadata column names into safe attribute identifiers."""
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _pick_channel_name(electrode_df: pd.DataFrame) -> np.ndarray:
    """Select a column to use as the channel name array, with fallbacks."""
    return electrode_df["origchannel_name"].astype(str).to_numpy()


def _subject_number_from_brain_area_subject(value: object) -> int | None:
    match = re.fullmatch(r"sub-p(?P<subject>\d+)cs", str(value).strip().lower())
    if match is None:
        return None
    return int(match.group("subject"))


def _coordinate_key(values: pd.Series, *, errors: str = "raise") -> pd.Series:
    return pd.to_numeric(values, errors=errors).round(3)


def _finite_xyz_mask(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce")
    return np.isfinite(coords.to_numpy()).all(axis=1)


def _add_coordinate_keys(
    df: pd.DataFrame,
    source_columns: tuple[str, str, str],
    *,
    errors: str = "raise",
) -> pd.DataFrame:
    for source_column, key_column in zip(
        source_columns, _BRAIN_AREA_COORD_KEY_COLUMNS, strict=True
    ):
        df[key_column] = _coordinate_key(df[source_column], errors=errors)
    return df


def _deduplicate_brain_area_labels(labels: pd.DataFrame) -> pd.DataFrame:
    duplicate_key_columns = [
        "subject_number",
        "electrode",
        *_BRAIN_AREA_COORD_KEY_COLUMNS,
    ]
    # The BYD export may include one row per session. Session rows are equivalent
    # for channel labeling when subject/electrode/coordinates and label context
    # are identical, so collapse them without making session part of the key.
    labels = labels.drop_duplicates(
        subset=[*duplicate_key_columns, *BRAIN_AREA_MERGE_COLUMNS],
    )
    duplicates = labels.duplicated(duplicate_key_columns, keep=False)
    if duplicates.any():
        duplicate_keys = (
            labels.loc[duplicates, duplicate_key_columns]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate brain area labels for {duplicate_keys}")
    return labels


def _mark_missing_coordinate_contacts(
    enriched: pd.DataFrame,
    missing_coordinate_mask: np.ndarray,
) -> pd.DataFrame:
    if not missing_coordinate_mask.any():
        return enriched

    enriched = enriched.copy()
    enriched.loc[missing_coordinate_mask, "label_dkt"] = ""
    enriched.loc[missing_coordinate_mask, "label_destrieux"] = ""
    if "location" in enriched.columns:
        enriched.loc[missing_coordinate_mask, "provided_location"] = (
            enriched.loc[missing_coordinate_mask, "location"].fillna("").astype(str)
        )
    else:
        enriched.loc[missing_coordinate_mask, "provided_location"] = ""
    enriched.loc[missing_coordinate_mask, "trajectory_role"] = ""
    enriched.loc[missing_coordinate_mask, "coordinate_assumption"] = "nwb_xyz_missing"
    enriched.loc[missing_coordinate_mask, "label_status"] = (
        "unlabeled_missing_coordinates"
    )
    enriched.loc[missing_coordinate_mask, "qc_status"] = (
        "not_labelable_missing_coordinates"
    )
    return enriched


def _load_brain_area_labels() -> pd.DataFrame:
    path = PIPELINE_DIR / "brain_areas" / "brain_area_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required brain area label CSV: {path}")

    try:
        labels = pd.read_csv(path, dtype={"subject": str, "electrode": str})
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Brain area label CSV is empty: {path}") from exc

    required = {
        "subject",
        "electrode",
        "mni_x",
        "mni_y",
        "mni_z",
        *BRAIN_AREA_MERGE_COLUMNS,
    }
    missing_columns = sorted(required.difference(labels.columns))
    if missing_columns:
        raise ValueError(
            f"Brain area label CSV missing required columns: {missing_columns}"
        )
    if labels.empty:
        raise ValueError(f"Brain area label CSV contains no rows: {path}")

    labels["subject_number"] = labels["subject"].map(
        _subject_number_from_brain_area_subject
    )
    if labels["subject_number"].isna().any():
        bad_subjects = sorted(
            labels.loc[labels["subject_number"].isna(), "subject"].unique()
        )
        raise ValueError(
            f"Could not parse BYD brain area label subjects: {bad_subjects}"
        )
    labels["electrode"] = labels["electrode"].astype(str)
    label_coords = labels[["mni_x", "mni_y", "mni_z"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(label_coords.to_numpy()).all():
        raise ValueError("Brain area label CSV contains non-finite MNI coordinates.")
    labels[["mni_x", "mni_y", "mni_z"]] = label_coords
    labels = _add_coordinate_keys(labels, ("mni_x", "mni_y", "mni_z"))
    return _deduplicate_brain_area_labels(labels)


def _add_brain_area_labels(
    electrode_df: pd.DataFrame,
    *,
    subject_number: int | None,
) -> pd.DataFrame:
    if subject_number is None:
        return electrode_df

    required_electrode_columns = {"origchannel_name", "x", "y", "z"}
    missing_electrode_columns = sorted(
        required_electrode_columns.difference(electrode_df.columns)
    )
    if missing_electrode_columns:
        raise ValueError(
            "Electrode table missing required columns for BYD brain area labels: "
            f"{missing_electrode_columns}"
        )

    labels = _load_brain_area_labels()
    subject_labels = labels.loc[labels["subject_number"] == subject_number].copy()
    enriched = electrode_df.copy()
    if subject_labels.empty:
        raise ValueError(
            f"No brain area labels found for BYD subject {subject_number}."
        )

    enriched["__channel_order"] = np.arange(len(enriched), dtype=int)
    enriched["__brain_area_electrode"] = enriched["origchannel_name"].astype(str)
    enriched = _add_coordinate_keys(enriched, ("x", "y", "z"), errors="coerce")

    subject_labels = subject_labels.rename(
        columns={"electrode": "__brain_area_electrode"}
    )
    merge_columns = [
        "__brain_area_electrode",
        *_BRAIN_AREA_COORD_KEY_COLUMNS,
    ]
    enriched = enriched.merge(
        subject_labels[merge_columns + list(BRAIN_AREA_MERGE_COLUMNS)],
        on=merge_columns,
        how="left",
        sort=False,
    ).sort_values("__channel_order")

    drop_columns = [
        "__channel_order",
        "__brain_area_electrode",
        *_BRAIN_AREA_COORD_KEY_COLUMNS,
    ]
    enriched = enriched.drop(columns=drop_columns).reset_index(drop=True)
    finite_coordinate_mask = _finite_xyz_mask(enriched)
    missing_coordinate_mask = ~finite_coordinate_mask

    if (
        len(enriched) != len(electrode_df)
        or _pick_channel_name(enriched).tolist()
        != _pick_channel_name(electrode_df).tolist()
    ):
        raise ValueError(
            "Brain area label merge changed channel order or channel count."
        )

    enriched = _mark_missing_coordinate_contacts(enriched, missing_coordinate_mask)

    missing = (
        finite_coordinate_mask
        & enriched[list(BRAIN_AREA_MERGE_COLUMNS)].isna().any(axis=1).to_numpy()
    )
    if missing.any():
        missing_rows = enriched.loc[missing, ["origchannel_name", "x", "y", "z"]]
        raise ValueError(
            f"Missing brain area labels for subject {subject_number}: "
            f"{missing_rows.to_dict('records')}"
        )
    return enriched


def _get_processing_interface(processing_module, name: str):
    """Safely fetch a processing data interface by name."""
    try:
        return processing_module.data_interfaces[name]
    except Exception:
        return None


def _get_electrical_series(container):
    """Return the ElectricalSeries from an LFP container, if present."""
    # LFP in pynwb exposes electrical_series as a dict-like container
    if hasattr(container, "electrical_series"):
        try:
            es_map = container.electrical_series
            if isinstance(es_map, dict) and len(es_map) > 0:
                # prefer named ElectricalSeries if present
                if "ElectricalSeries" in es_map:
                    return es_map["ElectricalSeries"]
                return next(iter(es_map.values()))
        except Exception:
            pass

    # Prefer direct lookup by name on data_interfaces
    try:
        series = container.data_interfaces.get("ElectricalSeries")
    except Exception:
        series = None
    if series is not None:
        return series

    # Fallback: find the first ElectricalSeries by type
    try:
        for value in container.data_interfaces.values():
            if value.__class__.__name__ == "ElectricalSeries":
                return value
    except Exception:
        pass

    return None


def _extract_lfp(nwbfile) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """Extract LFP data, timestamps, electrode metadata, and sampling rate."""
    if "ecephys" not in nwbfile.processing:
        raise ValueError("NWB file missing processing/ecephys")
    ecephys = nwbfile.processing["ecephys"]

    lfp_container = _get_processing_interface(ecephys, "LFP_macro")

    if lfp_container is None:
        available = list(getattr(ecephys, "data_interfaces", {}).keys())
        raise ValueError(
            f"NWB file missing LFP container in processing/ecephys. Available: {available}"
        )

    lfp_series = _get_electrical_series(lfp_container)
    if lfp_series is None:
        available = list(getattr(lfp_container, "data_interfaces", {}).keys())
        raise ValueError(
            "LFP container missing ElectricalSeries. "
            f"Available data_interfaces: {available}"
        )

    data = lfp_series.data[:]
    n_channels = len(lfp_series.electrodes)
    if data.ndim != 2:
        raise ValueError(f"Unexpected LFP data shape: {data.shape}")

    # Ensure shape is (time, channels) for RegularTimeSeries
    if data.shape[1] == n_channels:
        data_time_first = data
    elif data.shape[0] == n_channels:
        data_time_first = data.T
    else:
        raise ValueError(
            f"Cannot align LFP data with electrodes. "
            f"data.shape={data.shape}, n_channels={n_channels}"
        )

    # Build time axis and sampling rate following the upstream BYD scripts:
    # - prefer series.rate when available
    # - otherwise estimate from timestamp differences
    if lfp_series.rate is None:
        timestamps = lfp_series.timestamps[:]
        if len(timestamps) < 2:
            raise ValueError(
                "Cannot infer sampling rate from fewer than two LFP timestamps."
            )
        sampling_rate_hz = 1.0 / float(np.diff(timestamps).mean())
    else:
        sampling_rate_hz = float(lfp_series.rate)
        timestamps = np.arange(data_time_first.shape[0]) / float(
            lfp_series.rate
        ) + float(lfp_series.starting_time)

    if data_time_first.shape[0] != len(timestamps):
        raise ValueError(
            "LFP data/time mismatch: "
            f"data_time_first.shape[0]={data_time_first.shape[0]} vs "
            f"len(timestamps)={len(timestamps)}"
        )
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError(
            f"Invalid LFP sampling rate inferred from NWB: {sampling_rate_hz}"
        )

    electrode_df = lfp_series.electrodes.to_dataframe()
    return data_time_first, timestamps, electrode_df, float(sampling_rate_hz)


def _extract_encoding_interval(
    trials_df: pd.DataFrame | None,
) -> tuple[float, float]:
    """Return encoding start/stop timestamps from NWB trials metadata."""
    if trials_df is None:
        raise ValueError("NWB file missing trials table; expected encoding phase.")
    if "stim_phase" not in trials_df.columns:
        raise ValueError(
            "Trials table missing stim_phase column; expected encoding phase."
        )
    if "start_time" not in trials_df.columns or "stop_time" not in trials_df.columns:
        raise ValueError(
            "Trials table missing start_time/stop_time columns; expected encoding phase."
        )

    enc_df = trials_df.loc[trials_df["stim_phase"] == "encoding"]
    if enc_df.empty:
        raise ValueError("No encoding phase found in trials table.")

    # Deterministic anchor choice: earliest encoding interval by start_time.
    enc_df = enc_df.sort_values("start_time", kind="mergesort")
    enc_start = float(enc_df["start_time"].iloc[0])
    enc_stop = float(enc_df["stop_time"].iloc[0])

    if not np.isfinite(enc_start) or not np.isfinite(enc_stop):
        raise ValueError(
            f"Invalid encoding interval values: start={enc_start}, stop={enc_stop}"
        )
    if enc_stop <= enc_start:
        raise ValueError(
            f"Invalid encoding interval ordering: start={enc_start}, stop={enc_stop}"
        )

    return enc_start, enc_stop


def _apply_task_window(
    lfp_data: np.ndarray,
    lfp_time: np.ndarray,
    trials_df: pd.DataFrame | None,
    edge_offset_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fixed movie<->SEEG shift and crop to encoding +/- context.

    Alignment contract (BYD, hard-coded):
    - Correct SEEG timestamps by subtracting ``LFP_MOVIE_PAD_S``.
      This maps movie-time labels to the corrected SEEG timeline.
    - Keep only the encoding interval expanded by ``edge_offset_s`` on both sides.
    """
    enc_start, enc_stop = _extract_encoding_interval(trials_df)

    # 1) Clock correction: raw SEEG time -> movie/label time.
    lfp_time_aligned = lfp_time.astype(np.float64) - float(LFP_MOVIE_PAD_S)

    # 2) Context crop: keep encoding interval with symmetric edge margin.
    keep_start = float(enc_start - edge_offset_s)
    keep_stop = float(enc_stop + edge_offset_s)
    mask = np.logical_and(lfp_time_aligned >= keep_start, lfp_time_aligned <= keep_stop)

    if not np.any(mask):
        raise ValueError(
            "Encoding/task crop produced zero samples after alignment. "
            f"enc_start={enc_start}, enc_stop={enc_stop}, "
            f"edge_offset_s={edge_offset_s}, lfp_movie_pad_s={LFP_MOVIE_PAD_S}"
        )

    lfp_time_use = lfp_time_aligned[mask]
    # lfp_data is (time, channels), so mask the first dimension
    lfp_data_use = lfp_data[mask, :]
    return lfp_data_use, lfp_time_use


def _build_channels(
    electrode_df: pd.DataFrame,
    *,
    subject_number: int | None,
) -> ArrayDict:
    """Build a channel metadata ArrayDict including electrode table columns."""
    electrode_df = _add_brain_area_labels(
        electrode_df,
        subject_number=subject_number,
    )
    channel_names = _pick_channel_name(electrode_df)
    channels = ArrayDict(
        id=np.arange(len(channel_names)),
        name=channel_names,
        included=np.ones(len(channel_names), dtype=bool),
        type=np.ones(len(channel_names), dtype=int) * 31,
    )

    # Attach electrode metadata columns for reproducibility
    reserved = {"id", "name", "included", "type"}
    drop_cols = {"origchannel", "pairwise_distances", "origchannel_name"}
    used = set(reserved)
    for col in electrode_df.columns:
        safe_name = _sanitize_attr_name(col)
        if safe_name in drop_cols:
            continue
        if safe_name in reserved:
            raise ValueError(
                f"Electrode column name '{col}' sanitizes to reserved field '{safe_name}'."
            )
        if safe_name in used:
            raise ValueError(
                f"Duplicate sanitized electrode column name '{safe_name}' from '{col}'."
            )
        used.add(safe_name)

        values = electrode_df[col]
        if pd.api.types.is_string_dtype(values):
            values = values.fillna("").astype(str).to_numpy()
        elif pd.api.types.is_numeric_dtype(values):
            values = values.to_numpy(dtype=np.float32)
        else:
            # Fallback: stringify unknown dtype
            values = values.astype(str).to_numpy()
        setattr(channels, safe_name, values)

    # The released NWB x/y/z columns define BYD's public MNI152 RAS frame.
    # Coerce missing or non-finite source values to NaN so unlabelable contacts
    # retain serialized coordinate aliases without changing label semantics.
    coordinate_aliases = {
        "coord_byd_mni152_r": "x",
        "coord_byd_mni152_a": "y",
        "coord_byd_mni152_s": "z",
    }
    for alias, source in coordinate_aliases.items():
        values = pd.to_numeric(electrode_df[source], errors="coerce")
        values = values.mask(~np.isfinite(values), np.nan)
        setattr(channels, alias, values.to_numpy(dtype=np.float32).copy())

    return channels


def get_brainset_description() -> BrainsetDescription:
    """Return the dataset-level metadata description for this brainset."""
    return BrainsetDescription(
        id="keles_byd_2024",
        origin_version="DANDI:000623",
        derived_version="1.1.1",
        source="https://dandiarchive.org/dandiset/000623",
        description=(
            "Multimodal intracranial recordings during movie watching in human patients. "
            "NWB files include LFP and behavioral data recorded during the 'Bang! You're Dead' "
            "movie task and a recognition task."
        ),
    )


def _infer_subject_metadata(nwbfile) -> SubjectDescription:
    """Extract subject metadata from NWB, raising if missing or malformed."""
    try:
        return extract_subject_from_nwb(nwbfile)
    except Exception as exc:
        raise ValueError("Failed to extract subject metadata from NWB.") from exc


def _infer_session_metadata(nwbfile, session_id: str) -> SessionDescription:
    """Extract session metadata from NWB, with a fallback date."""
    recording_date = (
        nwbfile.session_start_time
        if hasattr(nwbfile, "session_start_time") and nwbfile.session_start_time
        else datetime.datetime(1970, 1, 1)
    )
    return SessionDescription(
        id=session_id,
        recording_date=recording_date,
        task=7,
    )


def _parse_recording_id_from_input_file(input_file: str) -> tuple[str, int, int]:
    """Parse canonical BYD recording id + subject/session integers from raw filename."""
    stem = Path(input_file).stem
    match = _RAW_RECORDING_STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(
            "Input filename must match "
            "'sub-CS<subject>_ses-P<subject>CSR<session>_behavior+ecephys.nwb' "
            "or canonical stem without suffix. "
            f"Got stem='{stem}'."
        )
    subject_id = int(match.group("subject"))
    p_subject_id = int(match.group("p_subject"))
    session_id = int(match.group("session"))
    if subject_id != p_subject_id:
        raise ValueError(
            f"Malformed BYD filename stem '{stem}': CS subject {subject_id} "
            f"does not match P subject {p_subject_id}."
        )
    recording_id = match.group(1)
    return recording_id, subject_id, session_id


def split_selector_key(
    *,
    label_mode: str,
    task_name: str,
    fold_idx: int,
    split_name: str,
) -> str:
    """Return shared selector key used for both splits.<key> and channel_splits.<key>."""
    return (
        f"{SUBSET_TIER}${label_mode}${WITHIN_SESSION_KEY}${task_name}$"
        f"fold{fold_idx}${split_name}"
    )


def _infer_device_metadata(nwbfile) -> DeviceDescription:
    """Infer device metadata from NWB devices, defaulting to depth electrodes."""
    device_id = None
    if hasattr(nwbfile, "devices") and len(nwbfile.devices) > 0:
        device_id = list(nwbfile.devices.keys())[0]
    return DeviceDescription(
        id=device_id or "depth_electrodes",
        recording_tech="STEREO_EEG",
        processing="raw_lfp",
    )


def _load_label_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a label CSV with `start_time_s` and `label` columns."""
    df = pd.read_csv(csv_path)
    if "start_time_s" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'start_time_s' and 'label' in {csv_path}")
    times = df["start_time_s"].astype(float).to_numpy()
    labels = pd.to_numeric(df["label"], errors="raise").to_numpy()
    return times, labels


def _parse_task_and_label_mode(label_file: str) -> tuple[str, str]:
    """Parse task name and label mode from a CSV filename.

    Required filename patterns:
    - <task>_binary_labels.csv
    - <task>_multiclass_labels.csv
    """
    name = Path(label_file).name
    match = re.fullmatch(r"(.+)_(binary|multiclass)_labels\.csv", name)
    if match is None:
        raise ValueError(
            "Label file naming must encode mode as "
            "'<task>_binary_labels.csv' or '<task>_multiclass_labels.csv'. "
            f"Got: {label_file}"
        )

    task_name, mode = match.group(1), match.group(2)
    if not task_name:
        raise ValueError(f"Could not parse task name from label filename: {label_file}")
    return task_name, mode


def _labels_for_mode(labels: np.ndarray, *, mode: str, csv_path: Path) -> np.ndarray:
    """Validate and coerce labels for a filename-selected mode."""
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
    """Build window start/end times around label timestamps, clipped to domain."""
    start = times - pre_offset_s
    end = times + post_offset_s

    valid = np.logical_and(start >= domain_start, end <= domain_end)
    start = start[valid]
    end = end[valid]
    labels = labels[valid]

    return start.astype(np.float64), end.astype(np.float64), labels


def _balanced_binary_indices(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return balanced indices for binary labels, preserving old BYD balancing behavior."""
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
    """Return balanced indices for multiclass labels by downsampling to the smallest class."""
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
    """Subsample an Interval to balanced classes for binary or multiclass labels."""
    labels = np.asarray(interval.label)
    # Keep 0/1 balancing behavior unchanged for binary labels.
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
            "keles_byd_2024",
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
    """Create two within-session folds with val as first half of test."""
    order = np.argsort(start)
    start = start[order]
    end = end[order]
    labels = labels[order]

    n = len(start)
    mid = n // 2
    folds = {}

    for fold_idx, (t0, t1) in enumerate(((0, mid), (mid, n))):
        test_indices = np.arange(t0, t1)
        train_indices = np.setdiff1d(np.arange(n), test_indices)

        # Validation = first half of the test set, per Neuroprobe convention.
        test_len = len(test_indices)
        val_split = t0 + (test_len // 2)
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
    """Build a label map for multiclass labels across tasks."""
    # TODO: replace identity maps once canonical class-name metadata is available.
    maps: dict[str, dict[str, dict[str, str]]] = {"multiclass": {}}
    for task, labels in label_tasks.items():
        unique = sorted(set(int(x) for x in np.unique(labels)))
        maps["multiclass"][task] = {str(v): str(v) for v in unique}
    return maps


def _log_prepare_audit(event: dict[str, object]) -> None:
    """Emit one structured prepare audit event to command-line logs."""
    logging.info("BYD_PREP_AUDIT %s", json.dumps(event, sort_keys=True, default=str))


def process_file(
    input_file: str,
    output_dir: str,
    *,
    labels_dir: str | None,
    label_files: Iterable[str],
    pre_offset_s: float,
    post_offset_s: float,
    no_splits: bool = False,
    balance_splits: bool = False,
    balance_seed: int = 0,
) -> str:
    """Process a single NWB file into a brainsets HDF5 output."""
    labels_dir = (
        labels_dir
        or "/home/geeling/Projects/ieeg_project/movie-features/output_labels/byd"
    )
    recording_id, subject_number, session_number = _parse_recording_id_from_input_file(
        input_file
    )
    subject_id = f"sub-CS{subject_number}"
    session_id = str(session_number)
    output_path = Path(output_dir) / f"{recording_id}.h5"
    enc_start = None
    enc_stop = None
    aligned_domain_start = None
    aligned_domain_end = None
    label_window_counts: dict[str, dict[str, object]] = {}

    try:
        with NWBHDF5IO(input_file, "r") as io:
            nwbfile = io.read()

            lfp_data, lfp_time, electrode_df, sampling_rate_hz = _extract_lfp(nwbfile)
            trials_df = (
                nwbfile.trials.to_dataframe() if nwbfile.trials is not None else None
            )
            enc_start, enc_stop = _extract_encoding_interval(trials_df)
            lfp_data, lfp_time = _apply_task_window(
                lfp_data, lfp_time, trials_df, EDGE_OFFSET_S
            )

            channels = _build_channels(electrode_df, subject_number=subject_number)
            channels.included = _compute_included_mask(electrode_df)

            seeg_data = RegularTimeSeries(
                data=lfp_data.astype(np.float32),
                sampling_rate=float(sampling_rate_hz),
                domain_start=float(lfp_time[0]),
                domain="auto",
            )

            # Metadata
            brainset_description = get_brainset_description()
            subject = _infer_subject_metadata(nwbfile)
            session_metadata = _infer_session_metadata(nwbfile, session_id)
            device = _infer_device_metadata(nwbfile)

        # Keep session.id canonical and globally unique.
        session = Data(
            id=recording_id,
            recording_date=getattr(
                session_metadata,
                "recording_date",
                datetime.datetime(1970, 1, 1),
            ),
            task=getattr(
                session_metadata,
                "task",
                7,
            ),
        )
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session,
            device=device,
            seeg_data=seeg_data,
            channels=channels,
            domain=seeg_data.domain,
        )

        if seeg_data.timestamps.size == 0:
            raise ValueError(
                "Aligned SEEG timestamps are empty; cannot build label windows."
            )
        aligned_domain_start = float(np.min(seeg_data.timestamps))
        aligned_domain_end = float(np.max(seeg_data.timestamps))
        if aligned_domain_end <= aligned_domain_start:
            raise ValueError(
                "Invalid aligned SEEG domain for split generation: "
                f"start={aligned_domain_start}, end={aligned_domain_end}"
            )

        alignment_parameters = {
            "lfp_movie_pad_s": float(LFP_MOVIE_PAD_S),
            "edge_offset_s": float(EDGE_OFFSET_S),
            "encoding_start_s": float(enc_start),
            "encoding_stop_s": float(enc_stop),
        }
        data.alignment_version = ALIGNMENT_VERSION
        data.alignment_reference = ALIGNMENT_REFERENCE
        data.alignment_method = ALIGNMENT_METHOD
        data.alignment_parameters_json = json.dumps(
            alignment_parameters, sort_keys=True
        )
        data.alignment_applied_at_prepare = True
        data.seeg_sampling_rate_hz = float(sampling_rate_hz)

        label_tasks_for_maps: dict[str, np.ndarray] = {}

        if not no_splits:
            data.splits = Data(domain=seeg_data.domain)
            data.channel_splits = Data()
            if not label_files:
                label_files = sorted(p.name for p in Path(labels_dir).glob("*.csv"))
            else:
                label_files = list(label_files)
            if len(label_files) == 0:
                raise ValueError("No label files found in labels-dir.")
            seen_mode_task_pairs: set[tuple[str, str]] = set()

            for label_file in label_files:
                csv_path = Path(labels_dir) / label_file
                task_name, label_mode = _parse_task_and_label_mode(label_file)
                mode_task = (label_mode, task_name)
                if mode_task in seen_mode_task_pairs:
                    raise ValueError(
                        "Duplicate task/mode label files would overwrite split attributes: "
                        f"mode={label_mode}, task={task_name}"
                    )
                seen_mode_task_pairs.add(mode_task)
                times, labels = _load_label_csv(csv_path)
                raw_count = int(len(times))

                start, end, labels = _build_windows(
                    times,
                    labels,
                    pre_offset_s=pre_offset_s,
                    post_offset_s=post_offset_s,
                    domain_start=aligned_domain_start,
                    domain_end=aligned_domain_end,
                )
                kept_count = int(len(start))
                label_window_counts[label_file] = {
                    "mode": label_mode,
                    "task_name": task_name,
                    "raw_rows": raw_count,
                    "kept_windows": kept_count,
                    "dropped_windows": int(raw_count - kept_count),
                }

                labels_for_mode = _labels_for_mode(
                    labels, mode=label_mode, csv_path=csv_path
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
                for fold_idx, split in folds.items():
                    for split_name in ("train", "val", "test"):
                        selector_key = split_selector_key(
                            label_mode=label_mode,
                            task_name=task_name,
                            fold_idx=fold_idx,
                            split_name=split_name,
                        )
                        setattr(data.splits, selector_key, split[split_name])
                        setattr(
                            data.channel_splits,
                            selector_key,
                            channels.included.astype(bool),
                        )

                if label_mode == "multiclass":
                    label_tasks_for_maps[task_name] = labels_for_mode

        if label_tasks_for_maps:
            label_maps = _build_label_maps(label_tasks_for_maps)
            data.label_maps_json = json.dumps(label_maps, sort_keys=True)

        with h5py.File(output_path, "w") as file:
            data.to_hdf5(file)
            file["seeg_data"].attrs["unit"] = "V"
            file["seeg_data"].attrs["scale_to_uV"] = 1e6

        logging.info(f"Saved processed file to {output_path}")
        _log_prepare_audit(
            {
                "status": "success",
                "input_file": str(input_file),
                "output_file": str(output_path),
                "recording_id": recording_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "alignment_version": ALIGNMENT_VERSION,
                "alignment_method": ALIGNMENT_METHOD,
                "lfp_movie_pad_s": float(LFP_MOVIE_PAD_S),
                "edge_offset_s": float(EDGE_OFFSET_S),
                "encoding_start_s": enc_start,
                "encoding_stop_s": enc_stop,
                "aligned_domain_start_s": aligned_domain_start,
                "aligned_domain_end_s": aligned_domain_end,
                "label_window_counts": label_window_counts,
            }
        )
        return str(output_path)
    except Exception as exc:
        _log_prepare_audit(
            {
                "status": "failure",
                "input_file": str(input_file),
                "output_file": str(output_path),
                "recording_id": recording_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "alignment_version": ALIGNMENT_VERSION,
                "alignment_method": ALIGNMENT_METHOD,
                "lfp_movie_pad_s": float(LFP_MOVIE_PAD_S),
                "edge_offset_s": float(EDGE_OFFSET_S),
                "encoding_start_s": enc_start,
                "encoding_stop_s": enc_stop,
                "aligned_domain_start_s": aligned_domain_start,
                "aligned_domain_end_s": aligned_domain_end,
                "label_window_counts": label_window_counts,
                "failure_reason": str(exc),
                "failure_type": type(exc).__name__,
            }
        )
        raise
