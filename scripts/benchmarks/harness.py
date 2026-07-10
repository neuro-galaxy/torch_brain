"""Shared benchmark harness: the timeit-based timing helper.

Kept in its own module so both bench_data and bench_utils can import it without
depending on benchmark.py (the entry point), which would create an import
cycle. The TORCH_BRAIN_SOURCE / sys.path shim lives in benchmark.py and runs
before either benchmark module is imported.
"""

import timeit

import numpy as np


def bench(label: str, stmt, number: int) -> dict:
    times = timeit.repeat(stmt, number=number, repeat=5)
    mean_us = np.mean(times) / number * 1e6
    return {"label": label, "number": number, "mean_us": round(mean_us, 3)}
