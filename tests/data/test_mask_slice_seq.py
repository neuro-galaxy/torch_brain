import copy
import os
import tempfile

import numpy as np
import pytest

from torch_brain.data import Data, IrregularTimeSeries, LazyIrregularTimeSeries

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


def build_slice_mask_ops(timestamps, series):

    idxs = np.arange(len(timestamps))

    ops, states = [], [(idxs, timestamps)]
    for i, op_kind in enumerate(series):
        n = len(timestamps)
        if op_kind == "mask":
            mask = (np.arange(n) + i) % 4 != 0
            idxs = idxs[mask]
            timestamps = timestamps[mask]
            ops.append(("mask", mask))
        else:
            pre, post = -0.005, -0.005
            if op_kind == "slice_wide":
                pre, post = -5, 5
            k = max(n // 10, 1)
            idx_slice = (
                timestamps[k] + pre,
                timestamps[n - k] + post,
            )  # trim off the ends
            mask = (timestamps >= idx_slice[0]) & (timestamps < idx_slice[1])
            idxs = idxs[mask]
            timestamps = timestamps[mask] - idx_slice[0]
            ops.append((op_kind, idx_slice))

        states.append((idxs, timestamps))

    return ops, states


@pytest.mark.parametrize("series_name", SERIES)
@pytest.mark.parametrize("materialize_step", range(len(SERIES["mask_first"]) + 1))
@pytest.mark.parametrize("incl_unloaded_data", [True, False])
def test_lazy_mask_slice_sequence(
    test_filepath, series_name, materialize_step, incl_unloaded_data
):

    timestamps = np.arange(1000) / 100 * 0.234
    unit_id = np.arange(len(timestamps))
    series = SERIES[series_name]

    ops, states = build_slice_mask_ops(timestamps, series)

    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_id=unit_id,
        ),
        domain="auto",
    )

    if incl_unloaded_data:
        unloaded_data = np.arange(len(timestamps)) * 0.5
        data.spikes.unloaded_data = unloaded_data

    data.save(test_filepath)

    del data

    with Data.load(test_filepath) as session:
        irr_ts = session.spikes

        def check_unit_id(step):
            err_msg = "Unit ID array incorrect"
            if step > materialize_step:
                err_msg = f"{err_msg} after steps: {series[:step]} (materialized after {materialize_step} steps)."
            else:
                err_msg = f"{err_msg} when materialized after steps: {series[:step]}."
            np.testing.assert_array_equal(
                irr_ts.unit_id, unit_id[states[step][0]], err_msg
            )

        def check_irr_ts_length(step):
            if len(irr_ts) != len(states[step][0]):
                if materialize_step <= step:
                    mat_str = "unit ID not yet materialized"
                else:
                    mat_str = f"unit ID materialized after {materialize_step} steps"
                raise RuntimeError(
                    f"Data length ({len(irr_ts)}) not as expected {len(states[step][0])} "
                    f"(steps to date: {series[:step]}, {mat_str})."
                )

        def copy_load_and_check(name, step):
            copied_and_loaded = getattr(copy.deepcopy(irr_ts), name)
            if name == "timestamps":
                gt = states[step][1]
            elif name == "unit_id":
                gt = unit_id[states[step][0]]
            err_msg = f"Timestamp loading is incorrect after steps: {series[:step]}."
            np.testing.assert_allclose(copied_and_loaded, gt, err_msg=err_msg)

        def check_lazy_or_not(step, end=False):
            if incl_unloaded_data:
                if not isinstance(irr_ts, LazyIrregularTimeSeries):
                    raise RuntimeError(
                        "irr_ts should still be lazy as 'unloaded_data' was not accessed."
                    )
            else:
                if step < materialize_step:
                    if not isinstance(irr_ts, LazyIrregularTimeSeries):
                        raise RuntimeError(
                            "irr_ts should still be lazy as unit ID has not been materialized "
                            f"yet (steps to date: {series[:step]})."
                        )
                else:
                    sliced = "slice" in series[:step] or "slice_wide" in series[:step]
                    if (sliced or end) and isinstance(irr_ts, LazyIrregularTimeSeries):
                        if sliced:
                            msg = "irr_ts was sliced"
                        else:
                            msg = "timestamps were accessed"
                        raise RuntimeError(
                            f"irr_ts should no longer be lazy as unit ID was materialized and {msg} "
                            f"(steps to date: {series[:step]})."
                        )
                    elif not sliced and not isinstance(irr_ts, LazyIrregularTimeSeries):
                        raise RuntimeError(
                            "irr_ts should still be lazy as unit ID was materialized, but irr_ts "
                            f"was not sliced (steps to date: {series[:step]})."
                        )

        # Run through the masking and slicing sequence
        if materialize_step == 0:
            check_unit_id(0)
            check_lazy_or_not(0)

        for step, (op_kind, arg) in enumerate(ops, start=1):
            if op_kind == "mask":
                irr_ts = irr_ts.select_by_mask(arg)
            else:
                irr_ts = irr_ts.slice(*arg)

            check_irr_ts_length(step)
            copy_load_and_check("timestamps", step)
            if step < materialize_step:
                copy_load_and_check("unit_id", step)
            else:
                check_unit_id(step)  # materializes unit_id
                check_lazy_or_not(step)

        err_msg = f"Final timestamps are incorrect (steps: {series})."
        np.testing.assert_allclose(irr_ts.timestamps, states[-1][1], err_msg=err_msg)

        check_lazy_or_not(step, end=True)
