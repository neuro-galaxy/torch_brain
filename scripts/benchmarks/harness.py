"""Shared benchmark harness: the timeit-based timing helper.

Kept in its own module so both bench_data and bench_utils can import it without
depending on benchmark.py (the entry point), which would create an import
cycle. The TORCH_BRAIN_SOURCE / sys.path shim lives in benchmark.py and runs
before either benchmark module is imported.
"""

import timeit


def bench(label: str, stmt, number: int) -> dict:
    """Time ``stmt`` over 5 batches of ``number`` calls; report the per-call time
    of the fastest batch in µs.

    The minimum, not the mean, as the timeit docs recommend: noise (other
    processes, shared CI runners) only ever adds time, so the fastest batch is
    the best estimate of the code's own cost and the least noisy across runs.
    """
    times = timeit.repeat(stmt, number=number, repeat=5)
    time_us = min(times) / number * 1e6
    return {"label": label, "number": number, "time_us": round(time_us, 3)}
