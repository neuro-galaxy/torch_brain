"""Generate the full multi-task Calcium POYO+ dataset config.

This script reads a per-session task-availability CSV (`task_contents.csv`),
groups every Allen Visual Coding session by the exact set of tasks it supports,
and writes a Hydra dataset config with one entry per group. Each group's
`multitask_readout` block is built from `torch_brain.registry.MODALITY_REGISTRY`
metadata (timestamp/value keys, dim, type) plus the normalization constants
computed by `scripts/calculate_normalization_scales.py` and the per-task loss
weights from POYO+ Table A2.

Output goes to `examples/poyo_plus/configs/dataset/calcium_poyo_plus.yaml` by
default. A small reference stub (`calcium_poyo_plus_example.yaml`) lives next
to the output so the expected schema is discoverable in-tree without running
this generator.

Usage:
    cd examples/poyo_plus/scripts
    python generate_config.py
    # then run training with: dataset=calcium_poyo_plus.yaml
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import yaml

from torch_brain.registry import MODALITY_REGISTRY, DataType

# Holdout sessions from calcium_poyo_plus_single_session.yaml
HOLDOUT_SESSIONS: Set[str] = {
    "710504563",
    "623339221",
    "589441079",
    "603763073",
    "676503588",
    "652092676",
    "649409874",
    "671164733",
    "623347352",
    "649401936",
    "555042467",
    "646016204",
    "595273803",
    "539487468",
    "637669284",
    "539497234",
    "652737678",
    "654532828",
    "669233895",
    "560926639",
    "547388708",
    "595806300",
    "689388034",
    "649938038",
    "645689073",
    "510514474",
    "505695962",
    "512326618",
    "562122508",
    "653122667",
}

# Mapping from CSV task flag column -> registry modality name
TASK_TO_MODALITY: Dict[str, str] = {
    "drifting_gratings.orientation": "drifting_gratings_orientation",
    "drifting_gratings.temporal_frequency": "drifting_gratings_temporal_frequency",
    "static_gratings.orientation": "static_gratings_orientation",
    "static_gratings.spatial_frequency": "static_gratings_spatial_frequency",
    "static_gratings.phase": "static_gratings_phase",
    "natural_scenes.frame": "natural_scenes",
    "natural_movie_one.frame": "natural_movie_one_frame",
    "natural_movie_two.frame": "natural_movie_two_frame",
    "natural_movie_three.frame": "natural_movie_three_frame",
    "locally_sparse_noise.frame": "locally_sparse_noise_frame",
    "running.running_speed": "running_speed",
    "pupil.location": "pupil_location",
}

REQUIRED_CSV_COLUMNS: Tuple[str, ...] = ("session_id",) + tuple(TASK_TO_MODALITY.keys())

# Normalization values from calculate_normalization_scales.py output
NORMALIZATION_VALUES: Dict[str, Dict[str, float]] = {
    "running_speed": {"mean": 6.80354332, "std": 13.87822103},
    "pupil_location.x": {"mean": 11.02599208, "std": 15.94543917},
    "pupil_location.y": {"mean": 16.91118513, "std": 6.83344030},
}

# Task weights from Table A2
TASK_WEIGHTS: Dict[str, float] = {
    "drifting_gratings_orientation": 1.0,
    "drifting_gratings_temporal_frequency": 1.0,
    "natural_movie_one_frame": 0.25,
    "natural_movie_two_frame": 0.2,
    "natural_movie_three_frame": 0.2,
    "locally_sparse_noise_frame": 1.0,
    "static_gratings_orientation": 1.0,
    "static_gratings_spatial_frequency": 1.0,
    "static_gratings_phase": 1.0,
    "natural_scenes": 0.3,
    "running_speed": 1.5,
    "pupil_location": 8.0,
}

# Natural-movie readouts are ordinal (frame index), so a strict top-1 hit is
# overly punitive. Add a tolerant WithinDeltaAccuracy that counts any prediction
# within +/- NM_TOLERANCE_FRAMES of the true frame as correct (e.g. +/- 30
# frames @ 30 Hz ~= +/- 1 s of slack, which roughly matches one model context
# window).
NM_TOLERANCE_FRAMES: int = 30
NATURAL_MOVIE_READOUTS: Set[str] = {
    "natural_movie_one_frame",
    "natural_movie_two_frame",
    "natural_movie_three_frame",
}

# Mapping from modality name to its domain interval key (used as the key under
# `weights:` so the loss can be masked to the relevant time intervals).
MODALITY_TO_DOMAIN: Dict[str, str] = {
    "pupil_location": "pupil.domain",
    "running_speed": "running.domain",
    "drifting_gratings_orientation": "drifting_gratings",
    "drifting_gratings_temporal_frequency": "drifting_gratings",
    "static_gratings_orientation": "static_gratings",
    "static_gratings_spatial_frequency": "static_gratings",
    "static_gratings_phase": "static_gratings",
    "natural_scenes": "natural_scenes",
    "natural_movie_one_frame": "natural_movie_one",
    "natural_movie_two_frame": "natural_movie_two",
    "natural_movie_three_frame": "natural_movie_three",
    "locally_sparse_noise_frame": "locally_sparse_noise",
}


def _to_bool(s: str) -> bool:
    """Lenient boolean parse for CSV flag columns ('True'/'true'/'1'/'yes')."""
    return str(s).strip().lower() in {"true", "1", "yes"}


def read_task_contents(
    csv_path: str,
    holdout_sessions: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, bool]]:
    """Read CSV and return {session_id: {task_flag: bool}}, excluding holdouts."""
    if holdout_sessions is None:
        holdout_sessions = HOLDOUT_SESSIONS

    sessions: Dict[str, Dict[str, bool]] = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        # Validate headers up front so a missing column produces an actionable
        # error rather than a per-row KeyError.
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        missing = [c for c in REQUIRED_CSV_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            session_id = row["session_id"].replace(".h5", "")
            if session_id in holdout_sessions:
                continue
            sessions[session_id] = {
                task_flag: _to_bool(row[task_flag])
                for task_flag in TASK_TO_MODALITY.keys()
            }
    return sessions


def group_sessions_by_tasks(
    sessions: Dict[str, Dict[str, bool]],
) -> Dict[Tuple, List[str]]:
    """Group sessions by their exact task combination."""
    groups: Dict[Tuple, List[str]] = defaultdict(list)
    for session_id, tasks in sessions.items():
        task_key = tuple(sorted(tasks.items()))
        groups[task_key].append(session_id)
    return groups


def create_readout_config(tasks: Dict[str, bool]) -> List[Dict]:
    """Build a multitask_readout list for one group, using the modality registry."""
    readouts: List[Dict] = []

    for task_key, is_active in tasks.items():
        if not is_active:
            continue

        modality_name = TASK_TO_MODALITY.get(task_key)
        if not modality_name or modality_name not in MODALITY_REGISTRY:
            continue

        modality_spec = MODALITY_REGISTRY[modality_name]

        readout: Dict = {
            "readout_id": modality_name,
        }

        if modality_name in TASK_WEIGHTS:
            weight_value = TASK_WEIGHTS[modality_name]
            if modality_name in MODALITY_TO_DOMAIN:
                readout["weights"] = {MODALITY_TO_DOMAIN[modality_name]: weight_value}
            else:
                readout["weights"] = weight_value

        is_pupil_2d = (
            modality_name == "pupil_location"
            and modality_spec.type == DataType.CONTINUOUS
            and modality_spec.dim == 2
        )

        if is_pupil_2d:
            if (
                "pupil_location.x" in NORMALIZATION_VALUES
                and "pupil_location.y" in NORMALIZATION_VALUES
            ):
                readout["normalize_mean"] = [
                    NORMALIZATION_VALUES["pupil_location.x"]["mean"],
                    NORMALIZATION_VALUES["pupil_location.y"]["mean"],
                ]
                readout["normalize_std"] = [
                    NORMALIZATION_VALUES["pupil_location.x"]["std"],
                    NORMALIZATION_VALUES["pupil_location.y"]["std"],
                ]
            else:
                readout["normalize_mean"] = [0.0, 0.0]
                readout["normalize_std"] = [1.0, 1.0]
        elif modality_spec.type == DataType.CONTINUOUS:
            if modality_name in NORMALIZATION_VALUES:
                readout["normalize_mean"] = NORMALIZATION_VALUES[modality_name]["mean"]
                readout["normalize_std"] = NORMALIZATION_VALUES[modality_name]["std"]
            else:
                readout["normalize_mean"] = 0.0
                readout["normalize_std"] = 1.0

        readout["timestamp_key"] = modality_spec.timestamp_key
        readout["value_key"] = modality_spec.value_key

        if modality_spec.type == DataType.MULTINOMIAL:
            metrics: List[Dict] = [
                {
                    "metric": {
                        "_target_": "torchmetrics.Accuracy",
                        "task": "multiclass",
                        "num_classes": modality_spec.dim,
                    }
                }
            ]
            if modality_name in NATURAL_MOVIE_READOUTS:
                metrics.append(
                    {
                        "metric": {
                            "_target_": "torch_brain.metrics.WithinDeltaAccuracy",
                            "tolerance": NM_TOLERANCE_FRAMES,
                        }
                    }
                )
        else:
            # Continuous (and any unhandled types) default to MSE.
            metrics = [{"metric": {"_target_": "torchmetrics.MeanSquaredError"}}]
        readout["metrics"] = metrics

        readouts.append(readout)

    return readouts


def _format_task_name(task_key: str) -> str:
    """Short label used in the per-group YAML comment headers."""
    short = {
        "drifting_gratings.orientation": "DG_orient",
        "drifting_gratings.temporal_frequency": "DG_temp_freq",
        "static_gratings.orientation": "SG_orient",
        "static_gratings.spatial_frequency": "SG_spat_freq",
        "static_gratings.phase": "SG_phase",
        "natural_scenes.frame": "NS",
        "natural_movie_one.frame": "NM1",
        "natural_movie_two.frame": "NM2",
        "natural_movie_three.frame": "NM3",
        "locally_sparse_noise.frame": "LSN",
        "running.running_speed": "running",
        "pupil.location": "pupil",
    }
    return short.get(task_key, task_key)


def generate_yaml_config(
    groups: Dict[Tuple, List[str]],
    sessions: Dict[str, Dict[str, bool]],
) -> List[Tuple[str, Dict]]:
    """Build (comment, group_config) tuples sorted by group size descending."""
    out: List[Tuple[str, Dict]] = []

    for i, (task_key, session_list) in enumerate(
        sorted(groups.items(), key=lambda x: len(x[1]), reverse=True), start=1
    ):
        tasks_dict = dict(task_key)
        active = [_format_task_name(k) for k, v in tasks_dict.items() if v]
        comment = f"# Group {i} - {len(session_list)} sessions"
        if active:
            comment += f" - {', '.join(active)}"

        first_session = session_list[0]
        readouts = create_readout_config(sessions[first_session])

        group_config: Dict = {
            "selection": [
                {
                    "brainset": "allen_visual_coding_ophys_2016",
                    "sessions": sorted(session_list),
                }
            ],
        }
        if readouts:
            group_config["config"] = {"multitask_readout": readouts}
        else:
            group_config["config"] = {}

        out.append((comment, group_config))

    return out


def _dump_groups(blocks: List[Tuple[str, Dict]]) -> str:
    """Serialize blocks to a YAML document, preserving comment headers per group.

    `yaml.safe_dump` is used to write each group as a list-element document so
    floats, lists, and any future bool/null values are emitted correctly. The
    per-group comment line is prepended manually since PyYAML drops comments.
    """
    chunks: List[str] = []
    for comment, group_config in blocks:
        body = yaml.safe_dump(
            [group_config],
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            width=10_000,  # avoid PyYAML breaking long session-id lists
        )
        chunks.append(f"{comment}\n{body}")
    return "\n".join(chunks)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "task_contents.csv")
    output_path = os.path.normpath(
        os.path.join(script_dir, "..", "configs", "dataset", "calcium_poyo_plus.yaml")
    )

    print(f"Excluding {len(HOLDOUT_SESSIONS)} holdout sessions")

    sessions = read_task_contents(csv_path, HOLDOUT_SESSIONS)
    groups = group_sessions_by_tasks(sessions)

    print(f"Number of groups: {len(groups)}")
    print(f"Total sessions: {sum(len(s) for s in groups.values())}")

    print("\nGroup sizes:")
    for i, (task_key, session_list) in enumerate(
        sorted(groups.items(), key=lambda x: len(x[1]), reverse=True), start=1
    ):
        tasks_dict = dict(task_key)
        active = [_format_task_name(k) for k, v in tasks_dict.items() if v]
        print(
            f"  Group {i}: {len(session_list)} sessions - "
            f"Tasks: {', '.join(active) if active else 'None'}"
        )

    blocks = generate_yaml_config(groups, sessions)
    yaml_text = _dump_groups(blocks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_text)

    print(f"\nGenerated config written to {output_path}")


if __name__ == "__main__":
    main()
