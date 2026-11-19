"""Shared benchmark harness: the timeit-based timing helper.

Kept in its own module so both bench_data and bench_utils can import it without
depending on benchmark.py (the entry point), which would create an import
cycle. The TORCH_BRAIN_SOURCE / sys.path shim lives in benchmark.py and runs
before either benchmark module is imported.
"""

from __future__ import annotations

import timeit
from collections.abc import Callable

DEFAULT_NUM_BATCHES = 10

_num_batches = DEFAULT_NUM_BATCHES


def set_num_batches(num_batches: int) -> None:
    """Set how many timed batches bench() runs (benchmark.py's --num-batches)."""
    global _num_batches
    _num_batches = num_batches


def bench(label: str, stmt: Callable[[], object], number: int) -> dict:
    """Time ``stmt`` in batches of ``number`` calls; report every batch's per-call
    time in µs.

    All batch times are kept so compare.py can report the median, mean,
    min and standard deviation, and test whether the baseline and target
    differ significantly. Called from inside each benchmark's setup,
    so its fixtures stay alive while the batches run.
    """
    timeit.timeit(stmt, number=number)  # untimed warm-up batch
    times = timeit.repeat(stmt, number=number, repeat=_num_batches)
    times_us = [round(t / number * 1e6, 3) for t in times]
    return {"label": label, "number": number, "times_us": times_us}
