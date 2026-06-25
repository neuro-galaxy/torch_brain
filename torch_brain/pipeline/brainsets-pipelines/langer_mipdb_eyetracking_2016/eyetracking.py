"""Parsing and synchronization of SMI iView eyetracking text exports.

Part of the :mod:`pipeline` for the CMI MIPDB EEG + eyetracking dataset.
The public entry point is :func:`process_session`, which returns a
``torch_brain.data.Data`` object containing up to four
``IrregularTimeSeries``: *samples*, *fixations*, *saccades*, *blinks*.
All timestamps are converted to the EEG timescale via a linear transform
fitted from matched event codes.

Synchronization strategy:
1) Build an index of all ET files and their paradigm segments.
2) Match the EEG session to the best ET file. Paradigms are grouped by
   start code, then full event code sequences are compared via LCS to
   resolve ambiguities (e.g. false starts that share the same start code).
3) Align per-paradigm event sequences (LCS) to collect anchor pairs.
4) Fit one linear ET->EEG timestamp transform for the session.
5) Parse the full ET file, apply the transform, clip to the EEG domain.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from constants import PARADIGM_MAP, SAMPLES_COLUMNS

from torch_brain.data import Data, Interval, IrregularTimeSeries


def parse_comma_separated_numbers(s: str) -> list[int | float]:
    """Parse a comma-separated string of numbers into a list of numeric values."""
    if not s:
        return []
    result: list[int | float] = []
    for token in s.split(","):
        token = token.strip()
        try:
            result.append(int(token))
        except ValueError:
            result.append(float(token))
    return result


def _group_eeg_events_by_paradigm(
    paradigm_intervals: Interval,
    paradigm_events: Interval | None = None,
) -> list[list[tuple[float, int]]]:
    """Return per-paradigm EEG events as ``(onset_s, code)`` lists.

    Prefers the normalized ``paradigm_events`` table. Falls back to legacy
    ``paradigm_intervals.event_codes`` / ``event_onsets`` if needed.
    """
    grouped: list[list[tuple[float, int]]] = [
        [] for _ in range(len(paradigm_intervals))
    ]
    if len(paradigm_intervals) == 0:
        return grouped

    if (
        paradigm_events is not None
        and len(paradigm_events) > 0
        and hasattr(paradigm_events, "paradigm_idx")
        and hasattr(paradigm_events, "code")
    ):
        for idx in range(len(paradigm_events)):
            paradigm_idx = int(paradigm_events.paradigm_idx[idx])
            if paradigm_idx < 0 or paradigm_idx >= len(grouped):
                continue
            grouped[paradigm_idx].append(
                (
                    float(paradigm_events.start[idx]),
                    int(paradigm_events.code[idx]),
                )
            )
        for events in grouped:
            events.sort(key=lambda item: item[0])
        return grouped

    if hasattr(paradigm_intervals, "event_codes") and hasattr(
        paradigm_intervals, "event_onsets"
    ):
        for paradigm_idx in range(len(paradigm_intervals)):
            event_codes = parse_comma_separated_numbers(
                paradigm_intervals.event_codes[paradigm_idx]
            )
            event_onsets = parse_comma_separated_numbers(
                paradigm_intervals.event_onsets[paradigm_idx]
            )
            grouped[paradigm_idx] = [
                (float(onset), int(code))
                for onset, code in zip(event_onsets, event_codes, strict=True)
            ]
        return grouped

    return grouped


def _parse_user_events_from_file(path: Path) -> list[tuple[int, str]]:
    """Read all UserEvent lines from an ET file (Events or Samples).

    UserEvent lines are used to match ET data to EEG paradigms.

    Returns list of (timestamp_µs, description) tuples.
    """
    user_events = []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("UserEvent"):
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    try:
                        ts = int(parts[3])
                    except ValueError:
                        continue
                    user_events.append((ts, parts[4].strip()))
    return user_events


def _identify_segments(user_events: list[tuple[int, str]]) -> list[dict]:
    """Identify paradigm segments bounded by ``start_eye_recording`` markers.

    Returns list of dicts with keys: code, paradigm_ts, start_ts, end_ts, all_events.
    ``end_ts`` is ``None`` for the last segment.
    ``all_events`` is a list of ``(timestamp_us, code)`` for every numeric
    ``# Message: <int>`` UserEvent within ``[start_ts, end_ts)``.
    """
    recording_starts = [ts for ts, desc in user_events if "start_eye_recording" in desc]

    numeric_events: list[tuple[int, int]] = []
    for ts, desc in user_events:
        m = re.match(r"# Message: (\d+)$", desc)
        if m:
            numeric_events.append((ts, int(m.group(1))))

    segments = []
    for ts, desc in user_events:
        m = re.match(r"# Message: (\d+)$", desc)
        if not m:
            continue
        code = int(m.group(1))
        if code not in PARADIGM_MAP:
            continue

        preceding_start = None
        for rs in recording_starts:
            if rs < ts:
                preceding_start = rs
        following_start = None
        for rs in recording_starts:
            if rs > ts:
                following_start = rs
                break

        start_ts = preceding_start if preceding_start else user_events[0][0]
        end_ts = following_start

        seg_events = [
            (ev_ts, ev_code)
            for ev_ts, ev_code in numeric_events
            if ev_ts >= start_ts and (end_ts is None or ev_ts < end_ts)
        ]

        segments.append(
            {
                "code": code,
                "paradigm_ts": ts,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "all_events": seg_events,
            }
        )
    return segments


def _et_file_base(path: Path) -> str:
    """Return the base name of an ET file (without ' Events.txt' or ' Samples.txt')."""
    name = path.name
    for suffix in (" Events.txt", " Samples.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_et_index(et_dir: Path) -> list[dict]:
    """Build a unified index of all eyetracking data from all files (Samples and Events).

    Scans all ET data files and groups them.
    Reads UserEvents from each group and identifies all paradigm segments.
    Returns an index with all paradigm segments found.

    Returns a list of dicts, one per ET "base" (logical file pair). Each dict has:
    - ``base_id``: str
    - ``samples_path``: Path | None
    - ``events_path``: Path | None
    - ``segments``: list of segment dicts (code, paradigm_ts, start_ts, end_ts)
    """
    if not et_dir or not et_dir.exists():
        return []

    bases: dict[str, dict[str, Path | None]] = {}
    for path in et_dir.glob("* Samples.txt"):
        base = _et_file_base(path)
        bases.setdefault(base, {"samples_path": None, "events_path": None})
        bases[base]["samples_path"] = path
    for path in et_dir.glob("* Events.txt"):
        base = _et_file_base(path)
        bases.setdefault(base, {"samples_path": None, "events_path": None})
        bases[base]["events_path"] = path

    index: list[dict] = []
    for base_id in sorted(bases.keys()):
        entry = bases[base_id]
        samples_path = entry["samples_path"]
        events_path = entry["events_path"]

        if events_path is not None and events_path.exists():
            user_events = _parse_user_events_from_file(events_path)
        elif samples_path is not None and samples_path.exists():
            user_events = _parse_user_events_from_file(samples_path)
        else:
            continue

        if not user_events:
            continue

        segments = _identify_segments(user_events)
        index.append(
            {
                "base_id": base_id,
                "samples_path": (
                    samples_path if samples_path and samples_path.exists() else None
                ),
                "events_path": (
                    events_path if events_path and events_path.exists() else None
                ),
                "segments": segments,
            }
        )

    return index


def _find_lcs(seq_a: list[int], seq_b: list[int]) -> list[tuple[int, int]]:
    """Return index pairs for one Longest Common Subsequence of *seq_a* and *seq_b*."""
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for r in range(n - 1, -1, -1):
        val = seq_a[r]
        for c in range(m - 1, -1, -1):
            if val == seq_b[c]:
                dp[r][c] = dp[r + 1][c + 1] + 1
            else:
                dp[r][c] = max(dp[r + 1][c], dp[r][c + 1])

    r, c = 0, 0
    pairs: list[tuple[int, int]] = []
    while r < n and c < m:
        if seq_a[r] == seq_b[c]:
            pairs.append((r, c))
            r += 1
            c += 1
        elif dp[r + 1][c] >= dp[r][c + 1]:
            r += 1
        else:
            c += 1
    return pairs


def _match_paradigms_for_code(
    eeg_indices: list[int],
    et_indices: list[int],
    eeg_event_seqs: list[list[int]],
    et_event_seqs: list[list[int]],
) -> list[tuple[int, int, int]]:
    """Find the best EEG-to-ET pairing for paradigms sharing the same start code.

    Returns list of ``(eeg_idx, et_idx, similarity)`` triples.
    """
    if len(eeg_indices) == 1 and len(et_indices) == 1:
        sim = len(
            _find_lcs(eeg_event_seqs[eeg_indices[0]], et_event_seqs[et_indices[0]])
        )
        return [(eeg_indices[0], et_indices[0], sim)]

    scores: list[tuple[int, int, int]] = []
    for eeg_i in eeg_indices:
        for et_j in et_indices:
            eeg_seq = eeg_event_seqs[eeg_i]
            et_seq = et_event_seqs[et_j]
            sim = len(_find_lcs(eeg_seq, et_seq)) / max(len(eeg_seq), len(et_seq))
            scores.append((eeg_i, et_j, sim))

    # Sort by similarity score, descending
    scores.sort(key=lambda x: x[2], reverse=True)

    used_eeg: set[int] = set()
    used_et: set[int] = set()
    result: list[tuple[int, int, int]] = []
    for eeg_i, et_j, sim in scores:
        if eeg_i not in used_eeg and et_j not in used_et and sim > 0:
            result.append((eeg_i, et_j, sim))
            used_eeg.add(eeg_i)
            used_et.add(et_j)
    return result


def match_eeg_to_et(
    paradigm_intervals: Interval,
    paradigm_events: Interval | None,
    et_index: list[dict],
) -> dict | None:
    """Match an EEG session to the best ET file using full paradigm event sequences.

    For each ET file, paradigms are grouped by start code. When multiple EEG
    paradigms share the same start code (e.g. due to false starts), compared the
    full event code sequence of each paradigm is via LCS to determine the
    correct pairing.

    Returns ``None`` if no match, or a dict with:
        ``base_id``, ``samples_path``, ``events_path`` — the matched ET file,
        ``segment_map`` — ``{eeg_paradigm_idx: segment_dict}`` mapping.
    """
    if len(paradigm_intervals) == 0 or not et_index:
        return None

    eeg_start_codes = [int(c) for c in paradigm_intervals.start_code]

    eeg_events_by_paradigm = _group_eeg_events_by_paradigm(
        paradigm_intervals, paradigm_events
    )
    eeg_event_seqs = [
        [code for _, code in paradigm_events]
        for paradigm_events in eeg_events_by_paradigm
    ]

    best_entry = None
    best_segment_map: dict[int, dict] = {}
    best_score: tuple[int, int] = (0, 0)

    for entry in et_index:
        segments = entry["segments"]
        et_codes = [int(seg["code"]) for seg in segments]
        et_event_seqs = [
            [code for _, code in seg.get("all_events", [])] for seg in segments
        ]

        common_codes = set(eeg_start_codes) & set(et_codes)
        if not common_codes:
            continue

        segment_map: dict[int, dict] = {}
        total_matches = 0
        total_similarity = 0

        for code in common_codes:
            eeg_idxs = [i for i, c in enumerate(eeg_start_codes) if c == code]
            et_idxs = [i for i, c in enumerate(et_codes) if c == code]

            matched_pairs = _match_paradigms_for_code(
                eeg_idxs, et_idxs, eeg_event_seqs, et_event_seqs
            )
            for eeg_i, et_i, sim in matched_pairs:
                segment_map[eeg_i] = segments[et_i]
                total_matches += 1
                total_similarity += sim

        score = (total_matches, total_similarity)
        if score > best_score:
            best_score = score
            best_segment_map = segment_map
            best_entry = entry

    if not best_segment_map or best_entry is None:
        return None

    return {
        "base_id": best_entry["base_id"],
        "samples_path": best_entry["samples_path"],
        "events_path": best_entry["events_path"],
        "segment_map": best_segment_map,
    }


def _match_eeg_to_et_event_timestamps(
    eeg_events: list[tuple[float, int]],
    et_events: list[tuple[int, int]],
) -> list[tuple[float, int]]:
    """Match EEG and ET event timestamps within one matched paradigm.

    Returns a list of ``(eeg_timestamp_s, et_timestamp_us)`` anchor pairs.
    """
    return [
        (eeg_events[i][0], et_events[i][0])
        for i in range(len(eeg_events))
        if eeg_events[i][1] == et_events[i][1]
    ]


def compute_et_to_eeg_time_transform(
    matched: dict,
    paradigm_intervals: Interval,
    paradigm_events: Interval | None,
) -> tuple[float, float] | None:
    """Fit a linear ET-to-EEG timestamp transform from all matched paradigms.
    Finds the relationship between the ET and EEG timestamps for a given paradigm.

    Returns ``(slope, intercept)`` or ``None`` if no anchors are found.
    """
    eeg_events_by_paradigm = _group_eeg_events_by_paradigm(
        paradigm_intervals, paradigm_events
    )
    all_anchors: list[tuple[float, int]] = []
    for paradigm_idx, seg in matched["segment_map"].items():
        eeg_events = eeg_events_by_paradigm[paradigm_idx]
        et_events = seg.get("all_events", [])
        anchors = _match_eeg_to_et_event_timestamps(eeg_events, et_events)
        if not anchors:
            eeg_start = float(paradigm_intervals.start[paradigm_idx])
            et_start = seg.get("paradigm_ts")
            if et_start is not None:
                anchors = [(eeg_start, int(et_start))]
                logging.warning(
                    "No sub-event matches for paradigm code %s; "
                    "using paradigm-start anchor only.",
                    seg.get("code"),
                )
        all_anchors.extend(anchors)

    if not all_anchors:
        return None

    et_s = np.array([a[1] for a in all_anchors], dtype=np.float64) / 1_000_000.0
    eeg_s = np.array([a[0] for a in all_anchors], dtype=np.float64)

    if len(all_anchors) == 1:
        slope = 1.0
        intercept = float(eeg_s[0] - et_s[0])
    else:
        slope, intercept = np.polyfit(et_s, eeg_s, 1)
        slope = float(slope)
        intercept = float(intercept)
        residuals = slope * et_s + intercept - eeg_s
        rmse = float(np.sqrt(np.mean(residuals**2)))
        logging.info(
            "Matching EEG and ET timescales",
            len(all_anchors),
            slope,
            intercept,
            rmse,
        )

    return (slope, intercept)


def _et_to_eeg_timestamps(
    t_us: int | np.ndarray, slope: float, intercept: float
) -> np.ndarray:
    """Convert ET microsecond timestamps to EEG seconds."""
    return slope * np.asarray(t_us, dtype=np.float64) / 1_000_000.0 + intercept


def parse_samples(
    samples_path: Path,
    slope: float,
    intercept: float,
    eeg_start_s: float,
    eeg_end_s: float,
) -> IrregularTimeSeries | None:
    """Parse SMP rows from a Samples.txt file, convert to EEG seconds, clip to EEG domain."""
    header_idx = None
    with open(samples_path, errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("Time\t"):
                header_idx = i
                break
    if header_idx is None:
        logging.warning(f"No column header found in {samples_path}")
        return None

    df = pd.read_csv(
        samples_path,
        sep="\t",
        skiprows=header_idx,
        low_memory=False,
        on_bad_lines="skip",
    )

    smp_df = df[df["Type"] == "SMP"].copy()
    if smp_df.empty:
        return None

    smp_df["Time"] = pd.to_numeric(smp_df["Time"], errors="coerce")
    smp_df = smp_df.dropna(subset=["Time"])
    smp_df["Time"] = smp_df["Time"].astype(np.int64)

    timestamps = _et_to_eeg_timestamps(smp_df["Time"].values, slope, intercept)

    keep = (timestamps >= eeg_start_s) & (timestamps <= eeg_end_s)
    smp_df = smp_df[keep]
    timestamps = timestamps[keep]

    if timestamps.size == 0:
        return None

    kwargs: dict = {}
    for src_col, dst_col in SAMPLES_COLUMNS.items():
        if src_col in smp_df.columns:
            kwargs[dst_col] = pd.to_numeric(
                smp_df[src_col], errors="coerce"
            ).values.astype(np.float32)

    return IrregularTimeSeries(
        timestamps=timestamps,
        domain="auto",
        **kwargs,
    )


def parse_events(
    events_path: Path,
    slope: float,
    intercept: float,
    eeg_start_s: float,
    eeg_end_s: float,
) -> dict:
    """Parse fixation / saccade / blink rows from an Events.txt file.

    Converts all timestamps to EEG seconds and clips to ``[eeg_start_s, eeg_end_s]``.
    Returns a dict whose values are ``IrregularTimeSeries`` keyed by
    ``"fixations"``, ``"saccades"``, ``"blinks"`` (only non-empty keys).
    """
    fixations: list[dict] = []
    saccades: list[dict] = []
    blinks: list[dict] = []

    with open(events_path, errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            etype = parts[0]

            try:
                if etype.startswith("Fixation") and len(parts) >= 13:
                    fixations.append(
                        {
                            "ts": int(parts[3]),
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                            "location_x": float(parts[6]),
                            "location_y": float(parts[7]),
                            "dispersion_x": float(parts[8]),
                            "dispersion_y": float(parts[9]),
                            "avg_pupil_size_x": float(parts[11]),
                            "avg_pupil_size_y": float(parts[12]),
                        }
                    )
                elif etype.startswith("Saccade") and len(parts) >= 14:
                    saccades.append(
                        {
                            "ts": int(parts[3]),
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                            "start_x": float(parts[6]),
                            "start_y": float(parts[7]),
                            "end_x": float(parts[8]),
                            "end_y": float(parts[9]),
                            "amplitude": float(parts[10]),
                            "peak_speed": float(parts[11]),
                            "avg_speed": float(parts[13]),
                        }
                    )
                elif etype.startswith("Blink") and len(parts) >= 6:
                    blinks.append(
                        {
                            "ts": int(parts[3]),
                            "eye": etype.split()[-1],
                            "duration": int(parts[5]),
                        }
                    )
            except (ValueError, IndexError):
                continue

    result: dict = {}

    if fixations:
        raw_ts = np.array([f["ts"] for f in fixations])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [fixations[i] for i in np.where(keep)[0]]
            result["fixations"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([f["eye"] for f in kept], dtype="U1"),
                duration=np.array([f["duration"] for f in kept], dtype=np.float64)
                / 1e6,
                location_x=np.array([f["location_x"] for f in kept], dtype=np.float32),
                location_y=np.array([f["location_y"] for f in kept], dtype=np.float32),
                dispersion_x=np.array(
                    [f["dispersion_x"] for f in kept], dtype=np.float32
                ),
                dispersion_y=np.array(
                    [f["dispersion_y"] for f in kept], dtype=np.float32
                ),
                avg_pupil_size_x=np.array(
                    [f["avg_pupil_size_x"] for f in kept], dtype=np.float32
                ),
                avg_pupil_size_y=np.array(
                    [f["avg_pupil_size_y"] for f in kept], dtype=np.float32
                ),
            )

    if saccades:
        raw_ts = np.array([s["ts"] for s in saccades])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [saccades[i] for i in np.where(keep)[0]]
            result["saccades"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([s["eye"] for s in kept], dtype="U1"),
                duration=np.array([s["duration"] for s in kept], dtype=np.float64)
                / 1e6,
                start_x=np.array([s["start_x"] for s in kept], dtype=np.float32),
                start_y=np.array([s["start_y"] for s in kept], dtype=np.float32),
                end_x=np.array([s["end_x"] for s in kept], dtype=np.float32),
                end_y=np.array([s["end_y"] for s in kept], dtype=np.float32),
                amplitude=np.array([s["amplitude"] for s in kept], dtype=np.float32),
                peak_speed=np.array([s["peak_speed"] for s in kept], dtype=np.float32),
                avg_speed=np.array([s["avg_speed"] for s in kept], dtype=np.float32),
            )

    if blinks:
        raw_ts = np.array([b["ts"] for b in blinks])
        eeg_ts = _et_to_eeg_timestamps(raw_ts, slope, intercept)
        keep = (eeg_ts >= eeg_start_s) & (eeg_ts <= eeg_end_s)
        if np.any(keep):
            kept = [blinks[i] for i in np.where(keep)[0]]
            result["blinks"] = IrregularTimeSeries(
                timestamps=eeg_ts[keep],
                domain="auto",
                eye=np.array([b["eye"] for b in kept], dtype="U1"),
                duration=np.array([b["duration"] for b in kept], dtype=np.float64)
                / 1e6,
            )

    return result


def process_session(
    et_dir: Path,
    paradigm_intervals: Interval,
    paradigm_events: Interval | None,
    eeg_start_s: float,
    eeg_end_s: float,
) -> Data | None:
    """Process continuous ET data for one EEG session, aligned to the EEG timescale.

    Returns a ``Data`` with up to four ``IrregularTimeSeries``
    (``samples``, ``fixations``, ``saccades``, ``blinks``), or ``None``.
    """
    et_index = build_et_index(et_dir)
    if not et_index:
        return None

    matched = match_eeg_to_et(paradigm_intervals, paradigm_events, et_index)
    if matched is None:
        return None

    transform = compute_et_to_eeg_time_transform(
        matched, paradigm_intervals, paradigm_events
    )
    if transform is None:
        return None
    slope, intercept = transform

    et_kwargs: dict = {}

    samples_path = matched["samples_path"]
    if samples_path is not None:
        samples = parse_samples(samples_path, slope, intercept, eeg_start_s, eeg_end_s)
        if samples is not None:
            et_kwargs["samples"] = samples

    events_path = matched["events_path"]
    if events_path is not None:
        et_kwargs.update(
            parse_events(events_path, slope, intercept, eeg_start_s, eeg_end_s)
        )

    if not et_kwargs:
        return None
    return Data(domain="auto", **et_kwargs)
