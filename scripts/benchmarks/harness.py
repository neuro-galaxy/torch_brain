"""Shared benchmark harness: the timeit-based timing helper.

Kept in its own module so both bench_data and bench_utils can import it without
depending on benchmark.py (the entry point), which would create an import
cycle. The TORCH_BRAIN_SOURCE / sys.path shim lives in benchmark.py and runs
before either benchmark module is imported.

Two modes:

- standalone (default): ``bench`` runs all batches back to back.
- worker (``start_worker``, used by compare.py through ``benchmark.py
  --worker``): ``bench`` sets up as usual, then runs one batch each time the
  driver asks, over a line-based JSON protocol. compare.py alternates batches
  between two workers, one per commit, so that drift in machine speed during
  the run affects both commits alike.
"""

from __future__ import annotations

import json
import sys
import timeit
from collections.abc import Callable

DEFAULT_NUM_BATCHES = 10

_num_batches = DEFAULT_NUM_BATCHES

_worker = None  # (command stream, reply stream) when driven by compare.py
current_index = 0  # position of the running benchmark in its suite (worker mode)


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
    if _worker is None:
        times = timeit.repeat(stmt, number=number, repeat=_num_batches)
    else:
        times = _serve_batches(label, stmt, number)
    times_us = [round(t / number * 1e6, 3) for t in times]
    return {"label": label, "number": number, "times_us": times_us}


def start_worker(commands, replies) -> None:
    """Switch ``bench`` to worker mode: read commands from ``commands``, write
    JSON replies to ``replies``."""
    global _worker
    _worker = (commands, replies)


def send(msg: dict) -> None:
    """Send one protocol message to the driver, tagged with the benchmark index
    so the driver can keep two workers in step (worker mode only)."""
    _, replies = _worker
    replies.write(json.dumps({"index": current_index, **msg}) + "\n")
    replies.flush()


def _serve_batches(label: str, stmt: Callable[[], object], number: int) -> list[float]:
    """Announce the benchmark, then run one batch per "batch" command until the
    driver sends "done". Returns the batch totals in seconds."""
    commands, _ = _worker
    timer = timeit.Timer(stmt)
    send({"event": "start", "label": label, "number": number})
    totals = []
    for line in commands:
        cmd = line.strip()
        if cmd == "batch":
            total = timer.timeit(number=number)
            totals.append(total)
            send({"time_us": round(total / number * 1e6, 3)})
        elif cmd == "done":
            return totals
        else:
            raise RuntimeError(f"unknown benchmark worker command: {cmd!r}")
    sys.exit(0)  # the driver closed the pipe
