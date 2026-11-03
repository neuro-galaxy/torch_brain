import copy
import os
import tempfile
from dataclasses import dataclass

import h5py
import numpy as np
import pytest

from torch_brain.data import (
    Interval,
    IrregularTimeSeries,
    LazyIrregularTimeSeries,
)

N_TIMESTAMPS = 1000

SEQUENCES = {
    "mask_first": [
        "mask",
        "slice",
        "slice_wide",
        "mask",
        "interval",
        "mask",
        "slice",
        "interval",
        "mask",
        "slice",
    ],
    "slice_first": [
        "slice",
        "mask",
        "slice",
        "interval",
        "slice_wide",
        "slice",
        "mask",
        "interval",
        "slice",
        "mask",
    ],
    "interval_first": [
        "interval",
        "slice",
        "mask",
        "mask",
        "slice_wide",
        "slice",
        "mask",
        "interval",
        "slice",
        "mask",
    ],
}
N_STEPS = len(SEQUENCES["mask_first"])
assert all(len(s) == N_STEPS for s in SEQUENCES.values())


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
        elif self.kind == "interval":
            return irr_ts.select_by_interval(self.arg)
        else:
            return irr_ts.slice(*self.arg)


def get_mask_op(step_idx: int, timestamps: np.ndarray):
    """Returns mask operation."""
    mask = (np.arange(len(timestamps)) + step_idx) % 4 != 0

    op = Op("mask", mask)
    return op


def get_interval_op(timestamps: np.ndarray):
    """Returns interval operation."""
    n = len(timestamps)
    k = max(n // 10, 1)
    start = timestamps[k]
    end = timestamps[n - k]
    diff = (end - start) / 2.2
    starts = [start, end - diff]
    ends = [start + diff, end]
    interval = Interval(starts, ends)

    op = Op("interval", interval)
    return op


def get_slice_op(kind: str, timestamps: np.ndarray):
    """Returns slice operation."""
    n = len(timestamps)
    k = max(n // 10, 1)
    start = timestamps[k]
    end = timestamps[n - k]
    if kind == "slice_wide":
        start = start - 5
        end = end + 5

    op = Op(kind, (start, end))
    return op


def build_slice_mask_ops(irr_ts: IrregularTimeSeries, sequence: list[str]):
    """Returns ops to apply and expected timestamps and unit_ids at each step."""
    ops = []
    all_timestamps = [irr_ts.timestamps.copy()]
    all_unit_ids = [irr_ts.unit_id.copy()]
    all_other_timekeys = [irr_ts.other_timekey.copy()]
    for i, kind in enumerate(sequence):
        if kind == "mask":
            op = get_mask_op(i, irr_ts.timestamps)
        elif kind == "interval":
            op = get_interval_op(irr_ts.timestamps)
        else:
            op = get_slice_op(kind, irr_ts.timestamps)
        ops.append(op)
        irr_ts = op.apply(irr_ts)

        all_timestamps.append(irr_ts.timestamps)
        all_unit_ids.append(irr_ts.unit_id)
        all_other_timekeys.append(irr_ts.other_timekey)

    return ops, all_timestamps, all_unit_ids, all_other_timekeys


def get_irr_ts(incl_unloaded_data: bool):
    """Returns an IrregularTimeSeries with timestamps, unit_id, and other_timekey."""
    timestamps = np.arange(N_TIMESTAMPS) / 100 * 0.234
    unit_id = np.arange(len(timestamps))
    other_timekey = np.linspace(-100, 100, len(timestamps)) * 5.678

    irr_ts = IrregularTimeSeries(
        timestamps=timestamps,
        unit_id=unit_id,
        other_timekey=other_timekey,
        timekeys=["timestamps", "other_timekey"],
    )
    if incl_unloaded_data:
        irr_ts.unloaded_data = np.arange(len(timestamps)) * 0.5
    return irr_ts


@dataclass
class SequenceChecker:
    sequence: list[str]
    all_timestamps: list[np.ndarray]
    all_unit_ids: list[np.ndarray]
    all_other_timekeys: list[np.ndarray]
    materialize_step: int
    incl_unloaded_data: bool
    load_timestamps_first: bool

    def _context(self, step: int, end: bool = False) -> str:
        end_str = ", and timestamps materialized" if end else ""
        return (
            f"after steps {self.sequence[:step]} "
            f"(unit_id materialized at step {self.materialize_step}{end_str})"
        )

    def check_mat_unit_id_and_other_timekey(self, irr_ts, step: int):
        np.testing.assert_array_equal(
            irr_ts.unit_id,
            self.all_unit_ids[step],
            f"Unit ID array incorrect {self._context(step)}",
        )
        np.testing.assert_allclose(
            irr_ts.other_timekey,
            self.all_other_timekeys[step],
            err_msg=f"Other timekey array incorrect {self._context(step)}",
        )

    def check_length(self, irr_ts, step: int):
        expected = len(self.all_timestamps[step])
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
        interval = any(op == "interval" for op in self.sequence[:step])
        if self.load_timestamps_first or interval:
            return False  # timestamps already materialized
        sliced = any(op in {"slice", "slice_wide"} for op in self.sequence[:step])
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
                irr_ts, "unit_id", self.all_unit_ids[step], step
            )
            self.check_load_from_deepcopy(
                irr_ts, "other_timekey", self.all_other_timekeys[step], step
            )

        else:
            self.check_mat_unit_id_and_other_timekey(irr_ts, step)
            self.check_laziness(irr_ts, step)


@pytest.mark.parametrize("sequence_name", SEQUENCES)
@pytest.mark.parametrize("materialize_step", range(N_STEPS + 1))
@pytest.mark.parametrize(
    "load_timestamps_first", [True, False], ids=["timestamps_first", "timestamps_last"]
)
@pytest.mark.parametrize(
    "incl_unloaded_data", [True, False], ids=["unloaded_data", "no_unloaded_data"]
)
def test_lazy_mask_slice_sequence(
    test_filepath,
    sequence_name,
    materialize_step,
    incl_unloaded_data,
    load_timestamps_first,
):

    irr_ts = get_irr_ts(incl_unloaded_data)
    with h5py.File(test_filepath, "w") as f:
        irr_ts.to_hdf5(f)

    sequence = SEQUENCES[sequence_name]

    ops, all_timestamps, all_unit_ids, all_other_timekeys = build_slice_mask_ops(
        irr_ts, sequence
    )

    del irr_ts

    checker = SequenceChecker(
        sequence,
        all_timestamps,
        all_unit_ids,
        all_other_timekeys,
        materialize_step,
        incl_unloaded_data,
        load_timestamps_first,
    )

    with h5py.File(test_filepath, "r") as f:
        irr_ts = LazyIrregularTimeSeries.from_hdf5(f)

        if load_timestamps_first:
            irr_ts.timestamps  # noqa: B018

        if materialize_step == 0:
            checker.check_mat_unit_id_and_other_timekey(irr_ts, 0)
            checker.check_laziness(irr_ts, 0)

        for step, op in enumerate(ops, start=1):
            irr_ts = op.apply(irr_ts)
            checker.check_step(irr_ts, step)

        np.testing.assert_allclose(
            irr_ts.timestamps,
            checker.all_timestamps[-1],
            err_msg=f"Final timestamps are incorrect (steps: {sequence}).",
        )
        checker.check_laziness(irr_ts, len(ops), end=True)
