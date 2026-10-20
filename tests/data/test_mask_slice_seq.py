import copy
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

from torch_brain.data import Data, IrregularTimeSeries, LazyIrregularTimeSeries

N_TIMESTAMPS = 1000

SERIES = {
    "mask_first": [
        "mask",
        "slice",
        "slice_wide",
        "mask",
        "mask",
        "slice",
        "mask",
        "slice",
    ],
    "slice_first": [
        "slice",
        "mask",
        "mask",
        "slice_wide",
        "slice",
        "mask",
        "slice",
        "mask",
    ],
}
N_STEPS = len(SERIES["mask_first"])
assert all(len(s) == N_STEPS for s in SERIES.values())


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name

    def finalizer():
        tmpfile.close()
        # clean up the temporary file after the test
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


@dataclass
class Op:
    kind: str
    arg: object

    def apply(self, irr_ts):
        if self.kind == "mask":
            return irr_ts.select_by_mask(self.arg)
        else:
            return irr_ts.slice(*self.arg)


def _mask_op(step_idx: int, idxs: np.ndarray, timestamps: np.ndarray):
    """Returns mask operation, and resulting idxs and timestamps."""
    mask = (np.arange(len(timestamps)) + step_idx) % 4 != 0
    op = Op("mask", mask)
    new_idxs = idxs[mask]
    new_timestamps = timestamps[mask]
    return op, new_idxs, new_timestamps


def _slice_op(kind: str, idxs: np.ndarray, timestamps: np.ndarray):
    """Returns slice operation, and resulting idxs and timestamps."""
    n = len(timestamps)
    if kind == "slice_wide":
        pre = -5
        post = 5
    else:
        pre = -0.005
        post = -0.005

    k = max(n // 10, 1)
    start = timestamps[k] + pre
    end = timestamps[n - k] + post
    mask = (timestamps >= start) & (timestamps < end)
    op = Op(kind, (start, end))
    new_idxs = idxs[mask]
    new_timestamps = timestamps[mask] - start
    return op, new_idxs, new_timestamps


def build_slice_mask_ops(timestamps: np.ndarray, series: list[str]):
    """Returns ops to apply and expected idxs and timestamps at each step."""
    idxs = np.arange(len(timestamps))
    ops = []
    all_idxs = [idxs]
    all_timestamps = [timestamps]
    for i, kind in enumerate(series):
        if kind == "mask":
            op, idxs, timestamps = _mask_op(i, idxs, timestamps)
        else:
            op, idxs, timestamps = _slice_op(kind, idxs, timestamps)
        ops.append(op)
        all_idxs.append(idxs)
        all_timestamps.append(timestamps)
    return ops, all_idxs, all_timestamps


def save_data(path, timestamps, unit_id, incl_unloaded_data: bool):
    data = Data(
        spikes=IrregularTimeSeries(timestamps=timestamps, unit_id=unit_id),
        domain="auto",
    )
    if incl_unloaded_data:
        data.spikes.unloaded_data = np.arange(len(timestamps)) * 0.5
    data.save(path)


@dataclass
class SequenceChecker:
    series: list[str]
    all_idxs: list[np.ndarray]
    all_timestamps: list[np.ndarray]
    unit_id: np.ndarray
    materialize_step: int
    incl_unloaded_data: bool

    def _context(self, step: int, end: bool = False) -> str:
        end_str = ", and timestamps materialized" if end else ""
        return (
            f"after steps {self.series[:step]} "
            f"(unit_id materialized at step {self.materialize_step}{end_str})"
        )

    def check_mat_unit_id(self, irr_ts, step: int):
        np.testing.assert_array_equal(
            irr_ts.unit_id,
            self.unit_id[self.all_idxs[step]],
            f"Unit ID array incorrect {self._context(step)}",
        )

    def check_length(self, irr_ts, step: int):
        expected = len(self.all_idxs[step])
        assert len(irr_ts) == expected, (
            f"Data length {len(irr_ts)} != expected {expected} {self._context(step)}"
        )

    def check_load_from_deepcopy(self, irr_ts, name: str, expected, step: int):
        loaded = getattr(copy.deepcopy(irr_ts), name)
        np.testing.assert_allclose(
            loaded,
            expected,
            err_msg=f"{name} loaded incorrectly {self._context(step)}",
        )

    def expected_lazy(self, step: int, end: bool = False) -> bool:
        if self.incl_unloaded_data:
            return True  # `unloaded_data` is never accessed
        if step < self.materialize_step:
            return True  # unit_id not materialized yet
        sliced = any(op in {"slice", "slice_wide"} for op in self.series[:step])
        return not (sliced or end)

    def check_laziness(self, irr_ts, step: int, end: bool = False):
        expected = self.expected_lazy(step, end)
        lazy_str = "be lazy" if expected else "not be lazy"
        assert isinstance(irr_ts, LazyIrregularTimeSeries) == expected, (
            f"irr_ts should {lazy_str} {self._context(step, end)}"
        )

    def check_step(self, irr_ts, step: int):
        self.check_length(irr_ts, step)
        self.check_load_from_deepcopy(
            irr_ts, "timestamps", self.all_timestamps[step], step
        )
        if step < self.materialize_step:
            self.check_load_from_deepcopy(
                irr_ts, "unit_id", self.unit_id[self.all_idxs[step]], step
            )
        else:
            self.check_mat_unit_id(irr_ts, step)
            self.check_laziness(irr_ts, step)


@pytest.mark.parametrize("series_name", SERIES)
@pytest.mark.parametrize("materialize_step", range(N_STEPS + 1))
@pytest.mark.parametrize(
    "incl_unloaded_data", [True, False], ids=["unloaded_data", "no_unloaded_data"]
)
def test_lazy_mask_slice_sequence(
    test_filepath, series_name, materialize_step, incl_unloaded_data
):
    series = SERIES[series_name]
    timestamps = np.arange(N_TIMESTAMPS) / 100 * 0.234
    unit_id = np.arange(len(timestamps))
    ops, all_idxs, all_timestamps = build_slice_mask_ops(timestamps, series)
    save_data(test_filepath, timestamps, unit_id, incl_unloaded_data)

    checker = SequenceChecker(
        series, all_idxs, all_timestamps, unit_id, materialize_step, incl_unloaded_data
    )

    with Data.load(test_filepath) as session:
        irr_ts = session.spikes

        if materialize_step == 0:
            checker.check_mat_unit_id(irr_ts, 0)
            checker.check_laziness(irr_ts, 0)

        for step, op in enumerate(ops, start=1):
            irr_ts = op.apply(irr_ts)
            checker.check_step(irr_ts, step)

        np.testing.assert_allclose(
            irr_ts.timestamps,
            checker.all_timestamps[-1],
            err_msg=f"Final timestamps are incorrect (steps: {series}).",
        )
        checker.check_laziness(irr_ts, len(ops), end=True)
