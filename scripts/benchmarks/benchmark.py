"""Benchmark suite for torch_brain.

Benchmarks are modeled on real torch_brain workloads: Data.slice() on
realistic lazy-loaded recording objects, IrregularTimeSeries/Interval
inner-loop slicing, Interval set operations, and bin_spikes, all at
production-typical sizes. Benchmarks are grouped by subpackage:
bench_data.py (torch_brain.data) and bench_utils.py (torch_brain.utils);
the shared timing helper lives in harness.py.

Usage:
    uv run python scripts/benchmarks/benchmark.py
    uv run python scripts/benchmarks/benchmark.py --json
    uv run python scripts/benchmarks/benchmark.py --save results.jsonl
    uv run python scripts/benchmarks/benchmark.py --suite data

Set TORCH_BRAIN_SOURCE to override where torch_brain is imported from (used by
compare.py to benchmark code from arbitrary commits).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

_source = os.environ.get(
    "TORCH_BRAIN_SOURCE", os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, _source)

# Imported after the sys.path shim above so their top-level torch_brain imports
# resolve to the code under test.
import bench_data  # noqa: E402
import bench_utils  # noqa: E402

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SUITES = {
    "data": bench_data.BENCHMARKS,
    "utils": bench_utils.BENCHMARKS,
    "all": bench_data.BENCHMARKS + bench_utils.BENCHMARKS,
}


def main():
    parser = argparse.ArgumentParser(description="Run torch_brain benchmarks.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--save", type=str, default=None, help="Append results to a JSONL file"
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="all",
        help="Which benchmark suite to run (default: all)",
    )
    args = parser.parse_args()

    results = []
    if not args.json:
        print(f"{'Benchmark':<42} {'Iters':>8} {'Mean (µs)':>12}")
        print("-" * 65)

    for bench_fn in SUITES[args.suite]:
        try:
            r = bench_fn()
        except Exception:
            r = {"label": bench_fn.__name__, "error": traceback.format_exc()}
        results.append(r)
        if not args.json:
            if "error" in r:
                print(f"{r['label']:<42} {'ERROR':>8} {'---':>12}")
            else:
                print(f"{r['label']:<42} {r['number']:>8} {r['mean_us']:>12.3f}")

    if args.json:
        print(json.dumps({"results": results}))

    if args.save:
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }
        with open(args.save, "a") as f:
            f.write(json.dumps(record) + "\n")
        if not args.json:
            print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
