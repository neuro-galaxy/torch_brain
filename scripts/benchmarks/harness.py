"""Shared benchmark harness: the timeit-based timing helper.

Kept in its own module so both bench_data and bench_utils can import it without
depending on benchmark.py (the entry point), which would create an import
cycle. The TORCH_BRAIN_SOURCE / sys.path shim lives in benchmark.py and runs
before either benchmark module is imported.
"""

import timeit

import numpy as np


def bench(label: str, stmt, number: int, critical: bool = False) -> dict:
    """Time ``stmt`` and return its mean per-call time in µs.

    ``critical`` marks benchmarks that model the training hot path; compare.py
    always lists them in the report summary instead of only in the full table.
    """
    times = timeit.repeat(stmt, number=number, repeat=5)
    mean_us = np.mean(times) / number * 1e6
    result = {"label": label, "number": number, "mean_us": round(mean_us, 3)}
    if critical:
        result["critical"] = True
    return result
