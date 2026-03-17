"""Benchmark suite for temporaldata.

Benchmarks are modeled on real torch_brain workloads: Data.slice() on
realistic lazy-loaded recording objects, IrregularTimeSeries/Interval
inner-loop slicing, and Interval set operations at production-typical sizes.

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --json
    uv run python benchmarks/benchmark.py --save results.jsonl

Set TEMPORALDATA_SOURCE to override where temporaldata is imported from
(used by compare.py to benchmark code from arbitrary commits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import timeit
import traceback

_source = os.environ.get(
    "TEMPORALDATA_SOURCE", os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, _source)

import h5py
import numpy as np

from temporaldata import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    LazyInterval,
    RegularTimeSeries,
)


def _bench(label: str, stmt, number: int) -> dict:
    times = timeit.repeat(stmt, number=number, repeat=5)
    mean_us = np.mean(times) / number * 1e6
    return {"label": label, "number": number, "mean_us": round(mean_us, 3)}


def _make_disjoint_intervals(n, min_gap=1.0, min_dur=0.5, max_dur=2.0, seed=42):
    rng = np.random.RandomState(seed)
    starts = np.empty(n, dtype=np.float64)
    ends = np.empty(n, dtype=np.float64)
    t = 0.0
    for i in range(n):
        t += rng.uniform(min_gap, min_gap + 3.0)
        dur = rng.uniform(min_dur, max_dur)
        starts[i] = t
        ends[i] = t + dur
        t = ends[i]
    return Interval(start=starts, end=ends)


def _build_realistic_data():
    """Build a large realistic Data object with nested splits.

    Matches real brainsets structure: ecog, pose, behavior intervals,
    channels, metadata Data children (brainset/subject/session/device),
    and a nested splits Data containing ~27 Intervals for
    3 task types x 3 folds x train/valid/test.
    """
    rng = np.random.RandomState(42)

    domain_starts = np.array([0.0, 120.0, 250.0, 400.0, 550.0, 700.0, 850.0])
    domain_ends = np.array([100.0, 230.0, 380.0, 520.0, 680.0, 830.0, 980.0])
    domain = Interval(start=domain_starts, end=domain_ends)

    ecog = RegularTimeSeries(
        signal=rng.standard_normal((500_000, 4)),
        sampling_rate=500.0,
        domain=Interval(0.0, 1000.0),
    )

    pose_kwargs = {}
    for part in [
        "r_wrist",
        "l_wrist",
        "l_ear",
        "l_elbow",
        "l_shoulder",
        "nose",
        "r_ear",
        "r_elbow",
        "r_shoulder",
    ]:
        pose_kwargs[part] = rng.standard_normal((30_000, 2))
    pose = RegularTimeSeries(
        **pose_kwargs,
        sampling_rate=30.0,
        domain=Interval(0.0, 1000.0),
    )

    n_spikes = 50_000
    spike_times = np.sort(rng.uniform(0, 1000, n_spikes))
    spikes = IrregularTimeSeries(
        timestamps=spike_times,
        unit_index=rng.randint(0, 100, n_spikes),
        waveforms=rng.standard_normal((n_spikes, 48)),
        domain=Interval(0.0, 1000.0),
    )

    n_trials = 500
    trial_starts = np.arange(0, n_trials * 18, 18, dtype=np.float64)
    trial_dur = rng.uniform(5, 15, n_trials)
    trial_ends = trial_starts + trial_dur
    active_behavior_trials = Interval(
        start=trial_starts,
        end=trial_ends,
        behavior_id=rng.randint(0, 4, n_trials),
        go_cue_time=trial_starts + rng.uniform(0.5, 2.0, n_trials),
        timekeys=["start", "end", "go_cue_time"],
    )
    active_vs_inactive_trials = Interval(
        start=trial_starts.copy(),
        end=trial_ends.copy(),
        behavior_id=rng.randint(0, 2, n_trials),
    )

    n_ch = 128
    channels = ArrayDict(
        id=np.arange(n_ch),
        hemisphere=rng.randint(0, 2, n_ch),
        surface=rng.randint(0, 2, n_ch),
    )

    splits_kwargs = {}
    for task in ["task_1", "task_2", "task_3"]:
        for fold in range(3):
            for split_name in ["train", "valid", "test"]:
                key = f"{task}_fold_{fold}_{split_name}"
                n_seg = int(rng.randint(400, 1200))
                gap = 90000.0 / n_seg
                s = np.arange(n_seg, dtype=np.float64) * gap
                e = s + rng.uniform(3, gap * 0.8, n_seg)
                splits_kwargs[key] = Interval(start=s, end=e)

    splits = Data(
        **splits_kwargs,
        domain=active_vs_inactive_trials,
    )

    brainset = Data(
        id="large_realistic_data",
        origin_version="0.0.1",
        source="synthetic",
    )
    subject = Data(id="sub_01", species="human")
    session = Data(id="sess_01", recording_date="2026-01-01")
    device = Data(id="ecog_grid", recording_tech="ECoG")

    return Data(
        ecog=ecog,
        pose=pose,
        spikes=spikes,
        active_behavior_trials=active_behavior_trials,
        active_vs_inactive_trials=active_vs_inactive_trials,
        channels=channels,
        splits=splits,
        brainset=brainset,
        subject=subject,
        session=session,
        device=device,
        pose_valid_domain=Interval(start=domain_starts.copy(), end=domain_ends.copy()),
        domain=domain,
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_data_slice_lazy():
    """
    Data.slice() on a lazy-loaded realistic recording.
    """
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    path = tmpfile.name
    tmpfile.close()

    try:
        data = _build_realistic_data()
        data.save(path)

        with h5py.File(path, "r") as f:
            lazy_data = Data.from_hdf5(f, lazy=True)

            def go():
                lazy_data.slice(300.0, 301.0)

            return _bench("Data.slice() (lazy, realistic)", go, number=200)
    finally:
        os.unlink(path)


def bench_data_slice_inmemory():
    """Data.slice() on an in-memory realistic recording."""
    data = _build_realistic_data()

    def go():
        data.slice(300.0, 301.0)

    return _bench("Data.slice() (in-memory)", go, number=500)


def bench_its_slice():
    """IrregularTimeSeries.slice() on a realistic recording."""
    rng = np.random.RandomState(42)
    n = 50_000
    ts = np.sort(rng.uniform(0, 1000, n))
    its = IrregularTimeSeries(
        timestamps=ts,
        unit_index=rng.randint(0, 100, n),
        waveforms=rng.standard_normal((n, 48)),
        domain=Interval(0.0, 1000.0),
    )

    def go():
        its.slice(500.0, 501.0)

    return _bench("IrregularTimeSeries.slice()", go, number=1_000)


def bench_rts_slice():
    """RegularTimeSeries.slice() on a realistic recording."""
    rng = np.random.RandomState(42)
    n = 50_000
    rts = RegularTimeSeries(
        sampling_rate=50,
        waveforms=rng.standard_normal((n, 48)),
        domain_start=0.0,
        domain="auto",
    )

    def go():
        rts.slice(500.0, 501.0)

    return _bench("RegularTimeSeries.slice()", go, number=1_000)


def bench_interval_slice():
    """Interval.slice() — 100 trial intervals over 1000s, slice a 1s window."""
    starts = np.arange(0, 1000, 10, dtype=np.float64)
    ends = starts + 5.0
    iv = Interval(start=starts, end=ends)

    def go():
        iv.slice(500.0, 501.0)

    return _bench("Interval.slice()", go, number=2_000)


def bench_interval_and_single():
    """Interval.__and__ 1000 segments & single window."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    single = Interval(500.0, 600.0)

    def go():
        d1 & single

    return _bench("Interval.__and__ (1k&single)", go, number=1_000)


def bench_interval_and_multi():
    """Interval.__and__ 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 & d2

    return _bench("Interval.__and__ (1k&100)", go, number=200)


def bench_interval_or():
    """Interval.__or__ 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 | d2

    return _bench("Interval.__or__ (1k|100)", go, number=200)


def bench_interval_difference():
    """Interval.difference 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1.difference(d2)

    return _bench("Interval.difference (1k-100)", go, number=200)


def bench_arraydict_keys():
    """ArrayDict.keys() tests the caching optimization."""
    ad = ArrayDict(**{f"key_{i}": np.arange(100, dtype=np.float64) for i in range(10)})

    def go():
        ad.keys()

    return _bench("ArrayDict.keys() x100k", go, number=100_000)


def bench_lazy_interval_access():
    """LazyInterval with 10 attributes stresses the _n_lazy O(1) counter."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    path = tmpfile.name
    tmpfile.close()

    rng = np.random.RandomState(42)
    n_intervals = 200
    starts = np.sort(rng.uniform(0, 10_000, n_intervals))
    ends = starts + rng.uniform(0.5, 2.0, n_intervals)

    iv = Interval(
        start=starts,
        end=ends,
        trial_type=rng.randint(0, 5, n_intervals),
        condition=rng.randint(0, 3, n_intervals),
        reward=rng.standard_normal(n_intervals),
        go_cue_time=starts + rng.uniform(0.1, 0.3, n_intervals),
        reaction_time=rng.uniform(0.15, 0.5, n_intervals),
        success=rng.randint(0, 2, n_intervals),
        target_pos_x=rng.standard_normal(n_intervals),
        target_pos_y=rng.standard_normal(n_intervals),
        timekeys=["start", "end", "go_cue_time"],
    )

    try:
        with h5py.File(path, "w") as f:
            iv.to_hdf5(f)

        with h5py.File(path, "r") as f:

            def go():
                lazy = LazyInterval.from_hdf5(f)
                _ = lazy.start
                _ = lazy.end
                _ = lazy.trial_type
                _ = lazy.condition
                _ = lazy.reward
                _ = lazy.go_cue_time
                _ = lazy.reaction_time
                _ = lazy.success
                _ = lazy.target_pos_x
                _ = lazy.target_pos_y

            return _bench("LazyInterval access (10 attrs)", go, number=2_000)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    bench_data_slice_lazy,
    bench_data_slice_inmemory,
    bench_its_slice,
    bench_rts_slice,
    bench_interval_slice,
    bench_interval_and_single,
    bench_interval_and_multi,
    bench_interval_or,
    bench_interval_difference,
    bench_arraydict_keys,
    bench_lazy_interval_access,
]


def main():
    parser = argparse.ArgumentParser(description="Run temporaldata benchmarks.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--save", type=str, default=None, help="Append results to a JSONL file"
    )
    args = parser.parse_args()

    results = []
    if not args.json:
        print(f"{'Benchmark':<42} {'Iters':>8} {'Mean (µs)':>12}")
        print("-" * 65)

    for bench_fn in BENCHMARKS:
        try:
            r = bench_fn()
        except Exception:
            r = {"label": bench_fn.__name__, "error": traceback.format_exc()}
        results.append(r)
        if not args.json:
            if "error" in r:
                print(f"{r['label']:<42} {'ERROR':>8} {'---':>12}")
            else:
                print(f"{r['label']:<42} {r['number']:>8} {r['mean_us']:>12.3f}")

    if args.json:
        print(json.dumps({"results": results}))

    if args.save:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        with open(args.save, "a") as f:
            f.write(json.dumps(record) + "\n")
        if not args.json:
            print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
