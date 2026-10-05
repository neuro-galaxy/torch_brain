"""torch_brain.data benchmarks.

Data.slice() on realistic lazy/in-memory recordings, IrregularTimeSeries /
RegularTimeSeries / Interval inner-loop slicing, Interval set operations,
ArrayDict access, and lazy attribute access for every Lazy* class (swept over
the number of attributes), all at production-typical sizes. The fixtures
that build the synthetic recordings live here alongside the benchmarks.

The sys.path shim that resolves ``torch_brain`` lives in benchmark.py and runs
before this module is imported, so the imports below pick up the code under
test (see TORCH_BRAIN_SOURCE).
"""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
from harness import bench

from torch_brain.data import (
    ArrayDict,
    Data,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    RegularTimeSeries,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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

            return bench("Data.slice() (lazy, realistic)", go, number=200)
    finally:
        os.unlink(path)


def bench_data_from_hdf5_lazy():
    """Data.from_hdf5(lazy=True) on the realistic recording: the per-file cost of
    opening a recording lazily, before any slicing or reads."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    path = tmpfile.name
    tmpfile.close()

    try:
        _build_realistic_data().save(path)

        with h5py.File(path, "r") as f:

            def go():
                Data.from_hdf5(f, lazy=True)

            return bench("Data.from_hdf5() (lazy, realistic)", go, number=200)
    finally:
        os.unlink(path)


def bench_data_slice_inmemory():
    """Data.slice() on an in-memory realistic recording."""
    data = _build_realistic_data()

    def go():
        data.slice(300.0, 301.0)

    return bench("Data.slice() (in-memory)", go, number=500)


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

    return bench("IrregularTimeSeries.slice()", go, number=1_000)


def bench_rts_slice():
    """RegularTimeSeries.slice() on a realistic recording."""
    rng = np.random.RandomState(42)
    n = 50_000
    rts = RegularTimeSeries(
        sampling_rate=50,
        waveforms=rng.standard_normal((n, 48)),
        domain_start=0.0,
    )

    def go():
        rts.slice(500.0, 501.0)

    return bench("RegularTimeSeries.slice()", go, number=1_000)


def bench_interval_slice():
    """Interval.slice() — 100 trial intervals over 1000s, slice a 1s window."""
    starts = np.arange(0, 1000, 10, dtype=np.float64)
    ends = starts + 5.0
    iv = Interval(start=starts, end=ends)

    def go():
        iv.slice(500.0, 501.0)

    return bench("Interval.slice()", go, number=2_000)


def bench_interval_and_single():
    """Interval.__and__ 1000 segments & single window."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    single = Interval(500.0, 600.0)

    def go():
        d1 & single

    return bench("Interval.__and__ (1k&single)", go, number=1_000)


def bench_interval_and_multi():
    """Interval.__and__ 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 & d2

    return bench("Interval.__and__ (1k&100)", go, number=200)


def bench_interval_or():
    """Interval.__or__ 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1 | d2

    return bench("Interval.__or__ (1k|100)", go, number=200)


def bench_interval_difference():
    """Interval.difference 1000 & 100 segments."""
    d1 = _make_disjoint_intervals(1000, seed=42)
    d2 = _make_disjoint_intervals(100, seed=99)

    def go():
        d1.difference(d2)

    return bench("Interval.difference (1k-100)", go, number=200)


def bench_arraydict_keys():
    """ArrayDict.keys() on 10 keys. keys() is not cached: it rebuilds a filtered
    list from __dict__ on every call, so this tracks that per-call cost."""
    ad = ArrayDict(**{f"key_{i}": np.arange(100, dtype=np.float64) for i in range(10)})

    def go():
        ad.keys()

    return bench("ArrayDict.keys() x100k", go, number=100_000)


# Lazy attribute access. Bookkeeping in Lazy*.__getattribute__ that scales with
# the number of attributes k (e.g. keys() lookups) makes reading all k
# attributes O(k^2). The k sweep guards against that: per-attribute time should
# stay flat as k grows.

_LAZY_N_ROWS = 1_000
_LAZY_DURATION = 100.0
_LAZY_KS = (10, 50, 100)


def _extra_attrs(n_attrs, rng):
    return {f"attr_{i}": rng.standard_normal(_LAZY_N_ROWS) for i in range(n_attrs)}


def _make_arraydict(k, rng):
    return ArrayDict(**_extra_attrs(k, rng))


def _make_interval(k, rng):
    # start/end count toward k
    starts = np.arange(_LAZY_N_ROWS) * (_LAZY_DURATION / _LAZY_N_ROWS)
    return Interval(start=starts, end=starts + 0.05, **_extra_attrs(k - 2, rng))


def _make_irregular_ts(k, rng):
    # timestamps counts toward k
    return IrregularTimeSeries(
        timestamps=np.sort(rng.uniform(0.0, _LAZY_DURATION, _LAZY_N_ROWS)),
        domain=Interval(0.0, _LAZY_DURATION),
        **_extra_attrs(k - 1, rng),
    )


def _make_regular_ts(k, rng):
    return RegularTimeSeries(
        sampling_rate=_LAZY_N_ROWS / _LAZY_DURATION,
        domain_start=0.0,
        **_extra_attrs(k, rng),
    )


def _run_lazy_access(label, obj, lazy_cls, mode, number):
    """Shared driver: save obj to HDF5, then time one lazy-access pattern.

    mode:
      "read"        load lazily, read every attribute (object materializes)
      "sliced"      load lazily, slice a 1s window, read every attribute
      "load-only"   load lazily, nothing else -- the from_hdf5 cost alone
      "slice-only"  load lazily, slice a 1s window, read nothing; subtract
                    load-only to get the cost of slice() itself
      "read-1"      load lazily, read a single attribute -- the "load many,
                    use few" pattern, where from_hdf5 dominates
    """
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    path = tmpfile.name
    tmpfile.close()

    keys = list(obj.keys())
    window = (_LAZY_DURATION / 2, _LAZY_DURATION / 2 + 1.0)

    try:
        with h5py.File(path, "w") as f:
            obj.to_hdf5(f)

        with h5py.File(path, "r") as f:
            to_read = {"read": keys, "sliced": keys, "read-1": keys[-1:]}.get(mode, [])

            def go():
                lazy = lazy_cls.from_hdf5(f)
                if mode in ("sliced", "slice-only"):
                    lazy = lazy.slice(*window)
                for key in to_read:
                    getattr(lazy, key)
                return lazy

            # sanity check: we must be timing the code path we think we are
            materialized = type(go()) is not lazy_cls
            assert materialized == (mode in ("read", "sliced")), f"{label}: wrong path"

            return bench(label, go, number=number)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _lazy_access_bench(short_name, make, lazy_cls, k, mode="read"):
    suffix = "" if mode == "read" else f", {mode}"
    label = f"Lazy{short_name} access (k={k}{suffix})"
    # modes that read at most one attribute are far cheaper per call
    cheap = mode in ("load-only", "slice-only", "read-1")
    number = 2_000 // k * (20 if cheap else 1)

    def fn():
        obj = make(k, np.random.RandomState(42))
        return _run_lazy_access(label, obj, lazy_cls, mode, number)

    fn.__name__ = f"bench_lazy_{short_name.lower()}_access_k{k}" + (
        "" if mode == "read" else "_" + mode.replace("-", "_")
    )
    fn.__doc__ = f"{label}: {_LAZY_N_ROWS} rows, {k} attributes."
    return fn


_LAZY_CLASSES = [
    ("ArrayDict", _make_arraydict, LazyArrayDict),
    ("Interval", _make_interval, LazyInterval),
    ("IrregularTS", _make_irregular_ts, LazyIrregularTimeSeries),
    ("RegularTS", _make_regular_ts, LazyRegularTimeSeries),
]
# LazyArrayDict has no slice()
_SLICEABLE_LAZY_CLASSES = [c for c in _LAZY_CLASSES if c[0] != "ArrayDict"]

LAZY_ACCESS_BENCHMARKS = (
    [
        _lazy_access_bench(name, make, lazy_cls, k)
        for name, make, lazy_cls in _LAZY_CLASSES
        for k in _LAZY_KS
    ]
    + [
        # slice()/unresolved_slice are resolved on first attribute read, so the
        # sliced read path has its own per-attribute cost
        _lazy_access_bench(name, make, lazy_cls, k, mode="sliced")
        for name, make, lazy_cls in _SLICEABLE_LAZY_CLASSES
        for k in _LAZY_KS
    ]
    + [
        _lazy_access_bench(name, make, lazy_cls, 100, mode=mode)
        for name, make, lazy_cls in _LAZY_CLASSES
        for mode in ("load-only", "read-1")
    ]
    + [
        _lazy_access_bench(name, make, lazy_cls, 100, mode="slice-only")
        for name, make, lazy_cls in _SLICEABLE_LAZY_CLASSES
    ]
)


def _data_slice_many_children_bench(n_children):
    label = f"Data.slice() ({n_children} children)"

    def fn():
        rng = np.random.RandomState(42)
        data = Data(
            **{f"ts_{i}": _make_irregular_ts(5, rng) for i in range(n_children)},
            domain=Interval(0.0, _LAZY_DURATION),
        )

        def go():
            data.slice(50.0, 51.0)

        return bench(label, go, number=4_000 // n_children)

    fn.__name__ = f"bench_data_slice_{n_children}_children"
    fn.__doc__ = f"{label}: Data.slice() cost grows with the number of children."
    return fn


DATA_SLICE_CHILDREN_BENCHMARKS = [_data_slice_many_children_bench(n) for n in (10, 50)]


BENCHMARKS = [
    bench_data_from_hdf5_lazy,
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
    *DATA_SLICE_CHILDREN_BENCHMARKS,
    *LAZY_ACCESS_BENCHMARKS,
]
