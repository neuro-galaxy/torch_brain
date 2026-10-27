import datetime
from argparse import Namespace
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
from _utils import add_pipelines_to_path
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import BehavioralTimeSeries

from torch_brain.data import Data
from torch_brain.datasets import KarpowiczPandarinathFalcon2024

add_pipelines_to_path()
from karpowicz_pandarinath_falcon_h1_2024 import (  # noqa: E402  # ty:ignore[unresolved-import]
    pipeline as h1,
)
from karpowicz_pandarinath_falcon_h1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_h1,
)
from karpowicz_pandarinath_falcon_h1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    trials_from_numbers,
)
from karpowicz_pandarinath_falcon_m1_2024 import (  # noqa: E402  # ty:ignore[unresolved-import]
    pipeline as m1,
)
from karpowicz_pandarinath_falcon_m1_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_m1,
)
from karpowicz_pandarinath_falcon_m2_2024 import (  # noqa: E402  # ty:ignore[unresolved-import]
    pipeline as m2,
)
from karpowicz_pandarinath_falcon_m2_2024.pipeline import (  # noqa: E402  # ty:ignore[unresolved-import]
    parse_filename as parse_m2,
)

PERIOD = 0.02
N_SAMPLES = 50  # 1 s of behavior at 50 Hz
T0 = 1.0  # FALCON time bases do not start at zero
SPIKE_TIMES = [[0.1, 0.5, 0.9], [], [1.25, 1.3]]  # unit 1 is silent
EVAL_MASK = np.r_[np.zeros(20, bool), np.ones(20, bool), np.zeros(10, bool)]
TRIAL_NUM = np.repeat([1, 2, 3], [20, 20, 10])


def _inside(interval, t):
    return np.array([((interval.start <= x) & (x < interval.end)).any() for x in t])


# ---------------------------------------------------------------------------
# synthetic NWB files
# ---------------------------------------------------------------------------


def _timestamps():
    return T0 + np.arange(N_SAMPLES) * PERIOD


def _new_nwbfile():
    return NWBFile(
        session_description="synthetic FALCON session",
        identifier="falcon-test",
        session_start_time=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )


def _add_units(nwbfile, with_electrodes):
    """Threshold-crossing channels. M1/M2 reference an electrodes table, H1 does not."""
    if with_electrodes:
        device = nwbfile.create_device(name="array")
        group = nwbfile.create_electrode_group(
            name="M1", description="", location="M1", device=device
        )
        for _ in SPIKE_TIMES:
            nwbfile.add_electrode(group=group, location="M1")
    for i, times in enumerate(SPIKE_TIMES):
        kwargs = {"electrodes": [i]} if with_electrodes else {}
        nwbfile.add_unit(spike_times=times, **kwargs)


def _add_eval_mask(nwbfile):
    nwbfile.add_acquisition(
        TimeSeries(
            name="eval_mask", data=EVAL_MASK, unit="bool", timestamps=_timestamps()
        )
    )


def _behavioral_container(name, channel_names, data):
    """A BehavioralTimeSeries with one timestamped series per channel."""
    series = [
        TimeSeries(name=n, data=data[:, i], unit="a.u.", timestamps=_timestamps())
        for i, n in enumerate(channel_names)
    ]
    return BehavioralTimeSeries(name=name, time_series=series)


def _write(nwbfile, path):
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    return path


def _make_h1(path):
    nwbfile = _new_nwbfile()
    _add_units(nwbfile, with_electrodes=False)
    rng = np.random.default_rng(0)
    for name in ("OpenLoopKinematics", "OpenLoopKinematicsVelocity"):
        # FALCON stores the sampling period (not the rate) in `rate`
        nwbfile.add_acquisition(
            TimeSeries(
                name=name,
                data=rng.standard_normal((N_SAMPLES, 7)),
                unit="a.u.",
                rate=PERIOD,
                starting_time=T0,
            )
        )
    _add_eval_mask(nwbfile)
    nwbfile.add_acquisition(
        TimeSeries(name="TrialNum", data=TRIAL_NUM, unit="", rate=PERIOD)
    )
    return _write(nwbfile, path)


def _make_m1(path):
    nwbfile = _new_nwbfile()
    _add_units(nwbfile, with_electrodes=True)
    rng = np.random.default_rng(0)
    muscles = ["EMG_BIC", "EMG_ECR", "EMG_FCR"]  # HDF5 returns children sorted
    nwbfile.add_acquisition(
        _behavioral_container(
            "preprocessed_emg", muscles, rng.standard_normal((N_SAMPLES, 3))
        )
    )
    _add_eval_mask(nwbfile)
    nwbfile.add_trial_column(name="result", description="free text (dropped)")
    nwbfile.add_trial_column(name="gocue_time", description="numeric (kept)")
    nwbfile.add_trial(start_time=1.0, stop_time=1.4, result="R", gocue_time=1.1)
    nwbfile.add_trial(start_time=1.4, stop_time=1.8, result="F", gocue_time=1.5)
    return _write(nwbfile, path)


def _make_m2(path):
    nwbfile = _new_nwbfile()
    _add_units(nwbfile, with_electrodes=True)
    rng = np.random.default_rng(0)
    fingers = ["index", "mrs"]
    for name in ("finger_pos", "finger_vel"):
        nwbfile.add_acquisition(
            _behavioral_container(name, fingers, rng.standard_normal((N_SAMPLES, 2)))
        )
    _add_eval_mask(nwbfile)
    nwbfile.add_trial_column(
        name="tgt_loc", description="target per finger group", index=True
    )
    nwbfile.add_trial(start_time=1.0, stop_time=1.4, tgt_loc=[0.1, 0.2])
    nwbfile.add_trial(start_time=1.4, stop_time=1.8, tgt_loc=[0.3, 0.4])
    return _write(nwbfile, path)


# ``(module, builder, FALCON file name, session id)`` for each dataset
H1_NAME = "sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"
M1_NAME = "sub-MonkeyL-held-in-minival_ses-20121004_behavior+ecephys.nwb"
M2_NAME = "sub-MonkeyN-held-out-calib_ses-2020-10-19-Run2_behavior+ecephys.nwb"
CASES = {
    "h1": (h1, _make_h1, H1_NAME, "human_pitt_19250108T110520_held_in_calib"),
    "m1": (m1, _make_m1, M1_NAME, "monkey_l_20121004_held_in_minival"),
    "m2": (m2, _make_m2, M2_NAME, "monkey_n_20201019_run2_held_out_calib"),
}
MANIFEST_IDS = {
    "h1": "held_in_calib_19250108T110520",
    "m1": "held_in_minival_20121004",
    "m2": "held_out_calib_20201019_run2",
}


def _open(path):
    io = NWBHDF5IO(path, "r")
    return io, io.read()


def _run_pipeline(module, nwb_path, processed_dir, reprocess=False):
    pipeline = module.Pipeline(
        raw_dir=nwb_path.parent,
        processed_dir=processed_dir,
        args=Namespace(redownload=False, reprocess=reprocess),
    )
    pipeline.process(nwb_path)
    return pipeline


def _load(h5_path):
    return Data.from_hdf5(h5py.File(h5_path, "r"))


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


# ``mask_to_interval`` is copied into each of the three pipelines
@pytest.mark.parametrize("module", [h1, m1, m2])
def test_mask_to_interval_matches_mask(module):
    t = np.arange(10) * 0.02
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    interval = module.mask_to_interval(mask, t)
    assert len(interval.start) == 3
    assert (_inside(interval, t) == mask).all()
    assert interval.is_sorted() and interval.is_disjoint()


@pytest.mark.parametrize("module", [h1, m1, m2])
def test_mask_to_interval_empty_and_length_check(module):
    t = np.arange(5) * 0.02
    assert len(module.mask_to_interval(np.zeros(5, dtype=bool), t).start) == 0
    with pytest.raises(ValueError):
        module.mask_to_interval(np.ones(4, dtype=bool), t)


def test_trials_from_numbers():
    t = np.arange(7) * 0.02
    trials = trials_from_numbers(np.array([3, 3, 4, 4, 4, 7, 7]), t)
    assert list(trials.number) == [3, 4, 7]
    assert (_inside(trials, t)).all()
    assert trials.is_disjoint()


def test_parse_filenames():
    assert parse_h1(
        "sub-HumanPitt/sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"
    ) == ("held_in_calib", "19250108T110520")
    assert parse_m1("sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys.nwb") == (
        "held_out_calib",
        "20121004",
    )
    assert parse_m2(
        "sub-MonkeyN-held-in-minival_ses-2020-10-19-Run2_behavior+ecephys.nwb"
    ) == ("held_in_minival", "20201019", "2")
    with pytest.raises(ValueError):
        parse_m1("sub-MonkeyL_ses-20121004.nwb")


# ---------------------------------------------------------------------------
# NWB extraction helpers
# ---------------------------------------------------------------------------


def test_h1_extract_spikes(tmp_path):
    io, nwbfile = _open(_make_h1(tmp_path / H1_NAME))
    with io:
        spikes, units = h1.extract_spikes(nwbfile)
    assert list(units.id.astype(str)) == ["chan000", "chan001", "chan002"]
    assert list(units.count) == [3, 0, 2]
    assert spikes.is_sorted()
    np.testing.assert_allclose(spikes.timestamps, [0.1, 0.5, 0.9, 1.25, 1.3])
    np.testing.assert_array_equal(spikes.unit_index, [0, 0, 0, 2, 2])


def test_m1_extract_emg(tmp_path):
    io, nwbfile = _open(_make_m1(tmp_path / M1_NAME))
    with io:
        emg, timestamps, muscles = m1.extract_emg(nwbfile)
    assert emg.sampling_rate == pytest.approx(1 / PERIOD)
    assert emg.data.shape == (N_SAMPLES, 3) and emg.data.dtype == np.float32
    np.testing.assert_allclose(timestamps, _timestamps())
    np.testing.assert_allclose(emg.timestamps, _timestamps())
    assert list(muscles.name.astype(str)) == ["EMG_BIC", "EMG_ECR", "EMG_FCR"]


def test_m1_extract_trials(tmp_path):
    io, nwbfile = _open(_make_m1(tmp_path / M1_NAME))
    with io:
        trials = m1.extract_trials(nwbfile)
        result_is_object = pd.api.types.is_object_dtype(
            nwbfile.trials.to_dataframe()["result"]
        )
    np.testing.assert_allclose(trials.start, [1.0, 1.4])
    np.testing.assert_allclose(trials.end, [1.4, 1.8])
    np.testing.assert_allclose(trials.gocue_time, [1.1, 1.5])
    # free-text columns are dropped when pandas types them as ``object`` (pandas < 3);
    # pandas >= 3 infers a string dtype, which is kept. Either way the result must
    # still be writable to HDF5.
    assert hasattr(trials, "result") == (not result_is_object)
    with h5py.File(tmp_path / "trials.h5", "w") as f:
        trials.to_hdf5(f)


def test_m2_extract_finger(tmp_path):
    io, nwbfile = _open(_make_m2(tmp_path / M2_NAME))
    with io:
        finger, timestamps, groups = m2.extract_finger(nwbfile)
        mrs_pos = nwbfile.acquisition["finger_pos"].get_timeseries("mrs").data[:]
    assert finger.pos.shape == finger.vel.shape == (N_SAMPLES, 2)
    assert finger.pos.dtype == finger.vel.dtype == np.float32
    np.testing.assert_allclose(finger.pos[:, 1], mrs_pos)
    np.testing.assert_allclose(timestamps, _timestamps())
    assert list(groups.name.astype(str)) == ["index", "mrs"]


def test_m2_extract_trials_stacks_targets(tmp_path):
    io, nwbfile = _open(_make_m2(tmp_path / M2_NAME))
    with io:
        trials = m2.extract_trials(nwbfile)
    np.testing.assert_allclose(trials.start, [1.0, 1.4])
    np.testing.assert_allclose(trials.target, [[0.1, 0.2], [0.3, 0.4]])
    assert trials.target.dtype == np.float32


@pytest.mark.parametrize("module", [m1, m2])
def test_irregular_behavior_timestamps_rejected(tmp_path, module):
    """Series with their own time base, or jittered ones, must not be stacked."""
    make, key = (
        (_make_m1, "preprocessed_emg") if module is m1 else (_make_m2, "finger_vel")
    )
    extract = module.extract_emg if module is m1 else module.extract_finger

    io, nwbfile = _open(make(tmp_path / "own.nwb"))
    with io:
        ts = nwbfile.acquisition[key].get_timeseries(
            list(nwbfile.acquisition[key].time_series)[-1]
        )
        ts.fields["timestamps"] = _timestamps() + 0.001
        with pytest.raises(ValueError, match="own timestamps"):
            extract(nwbfile)

    io, nwbfile = _open(make(tmp_path / "jitter.nwb"))
    with io:
        jittered = _timestamps()
        jittered[10] += 0.005
        for container in (
            [nwbfile.acquisition[key]]
            if module is m1
            else [nwbfile.acquisition["finger_pos"], nwbfile.acquisition["finger_vel"]]
        ):
            for ts in container.time_series.values():
                ts.fields["timestamps"] = jittered
        with pytest.raises(ValueError, match="not regularly sampled"):
            extract(nwbfile)


# ---------------------------------------------------------------------------
# whole per-file processing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", list(CASES))
def test_process_writes_hdf5(tmp_path, task):
    module, make, name, session_id = CASES[task]
    nwb_path = make(tmp_path / name)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    _run_pipeline(module, nwb_path, processed_dir)

    data = _load(processed_dir / f"{session_id}.h5")
    assert data.session.id == session_id
    assert data.brainset.id == module.Pipeline.brainset_id
    assert data.falcon_split == name.split("_ses-")[0].split("-", 2)[2].replace(
        "-", "_"
    )
    assert len(data.units.id) == 3 and len(data.spikes.timestamps) == 5

    behavior = getattr(data, {"h1": "kinematics", "m1": "emg", "m2": "finger"}[task])
    assert behavior.sampling_rate == pytest.approx(1 / PERIOD)
    np.testing.assert_allclose(behavior.timestamps, _timestamps())
    np.testing.assert_allclose(data.domain.start, [T0])
    np.testing.assert_allclose(data.domain.end, [T0 + N_SAMPLES * PERIOD])

    # evaluation mask -> one interval covering exactly the masked samples
    assert len(data.falcon_eval_intervals.start) == 1
    assert (_inside(data.falcon_eval_intervals, _timestamps()) == EVAL_MASK).all()

    if task == "h1":
        assert data.kinematics.pos.shape == data.kinematics.vel.shape == (N_SAMPLES, 7)
        assert list(data.kinematic_dims.name.astype(str)) == list(h1.KINEMATIC_DIMS)
        assert list(data.trials.number) == [1, 2, 3]
        assert (_inside(data.trials, _timestamps())).all()
    else:
        assert len(data.trials.start) == 2
    if task == "m1":
        assert data.emg.data.shape == (N_SAMPLES, 3)
        assert list(data.muscles.name.astype(str)) == ["EMG_BIC", "EMG_ECR", "EMG_FCR"]
    if task == "m2":
        assert data.finger.vel.shape == (N_SAMPLES, 2)
        assert list(data.finger_groups.name.astype(str)) == ["index", "mrs"]
        assert data.trials.target.shape == (2, 2)


def test_assign_domains_per_split():
    expected = {
        "held_in_calib": ("train",),
        "held_in_minival": ("valid",),
        "held_out_calib": (),
    }
    for split, populated in expected.items():
        data = Data(domain=h1.mask_to_interval(np.ones(3, bool), np.arange(3) * 0.02))
        h1.assign_domains(data, split)
        for name in ("train", "valid", "test"):
            domain = getattr(data, f"{name}_domain")
            assert (len(domain.start) == 1) == (name in populated), (split, name)


def test_process_skips_existing_output(tmp_path, capsys):
    module, make, name, session_id = CASES["m1"]
    nwb_path = make(tmp_path / name)
    (tmp_path / f"{session_id}.h5").write_bytes(b"")
    _run_pipeline(module, nwb_path, tmp_path)
    assert "Skipped Processing" in capsys.readouterr().out
    assert (tmp_path / f"{session_id}.h5").stat().st_size == 0

    _run_pipeline(module, nwb_path, tmp_path, reprocess=True)
    assert (tmp_path / f"{session_id}.h5").stat().st_size > 0


@pytest.mark.parametrize("task", list(CASES))
def test_manifest_download_process_roundtrip(tmp_path, monkeypatch, task):
    """``get_manifest`` -> ``download`` -> ``process`` with DANDI calls stubbed out."""
    module, make, name, session_id = CASES[task]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sub = "-".join(name.split("-")[:2])  # e.g. sub-MonkeyL
    (raw_dir / sub).mkdir()
    nwb_path = make(raw_dir / sub / name)
    url = f"https://example.invalid/{name}"
    downloads = []

    monkeypatch.setattr(
        module,
        "get_nwb_asset_list",
        lambda dandiset_id: [SimpleNamespace(path=f"{sub}/{name}", download_url=url)],
    )

    def fake_download(path, url_, raw_dir_, overwrite):
        downloads.append((path, url_, raw_dir_, overwrite))
        return raw_dir_ / path

    monkeypatch.setattr(module, "download_file", fake_download)

    manifest = module.Pipeline.get_manifest(raw_dir, None)
    assert manifest.index.tolist() == [MANIFEST_IDS[task]]
    assert manifest.loc[MANIFEST_IDS[task], "url"] == url

    pipeline = module.Pipeline(
        raw_dir=raw_dir,
        processed_dir=tmp_path,
        args=Namespace(redownload=True, reprocess=False),
    )
    pipeline._run_item(next(manifest.itertuples()))
    assert downloads == [(f"{sub}/{name}", url, raw_dir, True)]
    assert nwb_path.exists()
    assert (tmp_path / f"{session_id}.h5").exists()


# ---------------------------------------------------------------------------
# KarpowiczPandarinathFalcon2024 dataset class
# ---------------------------------------------------------------------------


def _processed_dataset(tmp_path, task, names):
    module, make, _, _ = CASES[task]
    processed_dir = tmp_path / f"karpowicz_pandarinath_falcon_{task}_2024"
    processed_dir.mkdir()
    for name in names:
        _run_pipeline(module, make(tmp_path / name), processed_dir)
    return KarpowiczPandarinathFalcon2024(task=task, root=tmp_path)


def test_dataset_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task must be"):
        KarpowiczPandarinathFalcon2024(task="h2", root=tmp_path)


def test_dataset_splits_and_unit_ids(tmp_path):
    dataset = _processed_dataset(
        tmp_path,
        "m1",
        [
            "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb",
            "sub-MonkeyL-held-in-minival_ses-20120926_behavior+ecephys.nwb",
            "sub-MonkeyL-held-out-calib_ses-20121004_behavior+ecephys.nwb",
        ],
    )
    calib, minival, held_out = dataset.recording_ids
    assert (calib, minival, held_out) == (
        "monkey_l_20120924_held_in_calib",
        "monkey_l_20120926_held_in_minival",
        "monkey_l_20121004_held_out_calib",
    )

    def populated(intervals):
        return sorted(rid for rid, itv in intervals.items() if len(itv.start))

    assert populated(dataset.get_sampling_intervals("train")) == [calib]
    assert populated(dataset.get_sampling_intervals("valid")) == [minival]
    assert populated(dataset.get_sampling_intervals("test")) == []
    assert populated(dataset.get_sampling_intervals()) == [calib, minival, held_out]
    assert list(dataset.get_sampling_intervals("held_out_calib")) == [held_out]

    # unit ids are prefixed with the session id so they stay unique across sessions
    unit_ids = dataset.get_unit_ids()
    assert len(unit_ids) == 9 and len(set(unit_ids)) == 9
    assert all(u.split("/", 1)[0] in dataset.recording_ids for u in unit_ids)


def test_dataset_default_dirname_and_override(tmp_path):
    dataset = _processed_dataset(tmp_path, "h1", [H1_NAME])
    assert dataset.recording_ids == ["human_pitt_19250108T110520_held_in_calib"]
    with pytest.raises(ValueError, match="No recordings found"):
        KarpowiczPandarinathFalcon2024(task="h1", root=tmp_path, dirname="elsewhere")
