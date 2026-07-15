"""Compare torch_brain benchmarks across git commits.

Extracts torch_brain source from arbitrary commits via `git archive` and
runs the current benchmark.py against each, then displays a side-by-side
comparison table.

Usage:
    uv run python scripts/benchmarks/compare.py                      # benchmark working tree
    uv run python scripts/benchmarks/compare.py <commit>              # <commit> vs working tree
    uv run python scripts/benchmarks/compare.py <commitA> <commitB>   # commitA vs commitB

Options:
    --save PATH    Append comparison results as JSONL to PATH.
    --suite NAME   Which benchmark suite to run: data, utils, or all (default: all).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

BENCH_SCRIPT = os.path.join(os.path.dirname(__file__), "benchmark.py")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def resolve_commit(ref: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(
            f"Error: cannot resolve ref '{ref}': {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout.strip()


def short_hash(full_hash: str) -> str:
    return full_hash[:10]


def _archive_pathspec(commit: str, pathspec: str, tmpdir: str) -> str | None:
    """Extract a single pathspec from a commit into tmpdir.

    Returns None on success, or an error string on failure (so callers can
    decide whether the failure is fatal).
    """
    git_proc = subprocess.run(
        ["git", "archive", commit, "--", pathspec],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if git_proc.returncode != 0:
        return f"git archive: {git_proc.stderr.decode(errors='replace').strip()}"

    tar_proc = subprocess.run(
        ["tar", "xf", "-", "-C", tmpdir],
        input=git_proc.stdout,
        capture_output=True,
        check=False,
    )
    if tar_proc.returncode != 0:
        return f"tar: {tar_proc.stderr.decode(errors='replace').strip()}"

    return None


def extract_source(commit: str) -> str:
    """Extract torch_brain source needed by the benchmarks into a temp dir.

    torch_brain/data/ is required; torch_brain/utils/ is extracted best-effort
    so the bin_spikes benchmark can import (older commits without it simply
    skip that one benchmark). A stub torch_brain/__init__.py is written so
    that ``from torch_brain.data import ...`` resolves to the extracted code
    without triggering the full package's imports (which may pull in heavy
    dependencies like torch).
    """
    tmpdir = tempfile.mkdtemp(prefix="tdbench_")

    err = _archive_pathspec(commit, "torch_brain/data/", tmpdir)
    if err is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"Error: extracting torch_brain/data/ for {short_hash(commit)}: {err}")
        sys.exit(1)

    # Best-effort: absent on commits predating the module; the bin_spikes
    # benchmark then errors in isolation instead of breaking the whole run.
    utils_err = _archive_pathspec(commit, "torch_brain/utils/", tmpdir)
    if utils_err is not None:
        print(
            f"Note: torch_brain/utils/ unavailable for {short_hash(commit)} "
            f"({utils_err}); bin_spikes benchmark will be skipped.",
            file=sys.stderr,
        )

    # Write a minimal stub so `import torch_brain` succeeds without
    # pulling in the real package's __init__.py and its heavy deps.
    pkg_init = os.path.join(tmpdir, "torch_brain", "__init__.py")
    with open(pkg_init, "w") as f:
        f.write("")

    return tmpdir


def run_benchmark(
    source_dir: str | None, label: str, suite: str = "all"
) -> list[dict] | None:
    """Run benchmark.py, optionally overriding the import source.

    Returns the results list, or ``None`` if the benchmark subprocess failed
    (e.g. import errors in the extracted source from an older commit).
    """
    env = os.environ.copy()
    if source_dir is not None:
        env["TORCH_BRAIN_SOURCE"] = source_dir

    print(f"Running benchmarks for {label}...", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, BENCH_SCRIPT, "--json", "--suite", suite],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"Benchmark run FAILED for {label}:")
        print(result.stderr)
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output for {label}:")
        print(result.stdout[:500])
        return None

    return data["results"]


def print_single(results: list[dict], label: str):
    print(f"\n  Results for {label}\n")
    print(f"  {'Benchmark':<42} {'Iters':>8} {'Mean (µs)':>12}")
    print(f"  {'-' * 65}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<42} {'ERROR':>8} {'---':>12}")
        else:
            print(f"  {r['label']:<42} {r['number']:>8} {r['mean_us']:>12.3f}")


def print_comparison(
    results_a: list[dict], results_b: list[dict], label_a: str, label_b: str
):
    index_b = {r["label"]: r for r in results_b}

    col_a = f"{label_a} (µs)"
    col_b = f"{label_b} (µs)"
    print(f"\n  {'Benchmark':<42} {col_a:>18} {col_b:>18} {'Speedup':>10}")
    print(f"  {'-' * 92}")

    for ra in results_a:
        label = ra["label"]
        rb = index_b.get(label)

        val_a = f"{ra['mean_us']:.3f}" if "error" not in ra else "ERROR"
        if rb is None:
            val_b = "n/a"
            speedup = ""
        elif "error" in rb:
            val_b = "ERROR"
            speedup = ""
        else:
            val_b = f"{rb['mean_us']:.3f}"
            if "error" not in ra and rb["mean_us"] > 0:
                ratio = ra["mean_us"] / rb["mean_us"]
                speedup = f"{ratio:.2f}x"
            else:
                speedup = ""

        print(f"  {label:<42} {val_a:>18} {val_b:>18} {speedup:>10}")

    # benchmarks only in B
    labels_a = {r["label"] for r in results_a}
    for rb in results_b:
        if rb["label"] not in labels_a:
            val_b = f"{rb['mean_us']:.3f}" if "error" not in rb else "ERROR"
            print(f"  {rb['label']:<42} {'n/a':>18} {val_b:>18} {''!s:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare torch_brain benchmarks across git commits.",
        epilog="Examples:\n"
        "  uv run python scripts/benchmarks/compare.py\n"
        "  uv run python scripts/benchmarks/compare.py abc123\n"
        "  uv run python scripts/benchmarks/compare.py abc123 def456\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "commits", nargs="*", help="0, 1, or 2 commit refs to benchmark"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Append results to a JSONL file"
    )
    parser.add_argument(
        "--suite",
        choices=["data", "utils", "all"],
        default="all",
        help="Which benchmark suite to run (default: all)",
    )
    args = parser.parse_args()

    if len(args.commits) > 2:
        parser.error("At most 2 commit refs can be provided.")

    tmpdirs: list[str] = []
    had_failures = False
    try:
        if len(args.commits) == 0:
            results = run_benchmark(None, "working tree", args.suite)
            if results is None:
                had_failures = True
            else:
                print_single(results, "working tree")
            save_record = {
                "baseline": "working-tree",
                "target": None,
                "results_baseline": results,
                "results_target": None,
            }

        elif len(args.commits) == 1:
            commit = resolve_commit(args.commits[0])
            label_a = short_hash(commit)

            tmpdir = extract_source(commit)
            tmpdirs.append(tmpdir)

            results_a = run_benchmark(tmpdir, label_a, args.suite)
            results_b = run_benchmark(None, "working tree", args.suite)

            if results_a is None or results_b is None:
                had_failures = True
            if results_a is not None and results_b is not None:
                print_comparison(results_a, results_b, label_a, "working tree")
            elif results_b is not None:
                print(f"\n  WARNING: baseline ({label_a}) benchmark failed.")
                print_single(results_b, "working tree")
            elif results_a is not None:
                print("\n  WARNING: target (working tree) benchmark failed.")
                print_single(results_a, label_a)

            save_record = {
                "baseline": label_a,
                "target": "working-tree",
                "results_baseline": results_a,
                "results_target": results_b,
            }

        else:
            commit_a = resolve_commit(args.commits[0])
            commit_b = resolve_commit(args.commits[1])
            label_a = short_hash(commit_a)
            label_b = short_hash(commit_b)

            tmpdir_a = extract_source(commit_a)
            tmpdirs.append(tmpdir_a)
            tmpdir_b = extract_source(commit_b)
            tmpdirs.append(tmpdir_b)

            results_a = run_benchmark(tmpdir_a, label_a, args.suite)
            results_b = run_benchmark(tmpdir_b, label_b, args.suite)

            if results_a is None or results_b is None:
                had_failures = True
            if results_a is not None and results_b is not None:
                print_comparison(results_a, results_b, label_a, label_b)
            elif results_b is not None:
                print(f"\n  WARNING: baseline ({label_a}) benchmark failed.")
                print_single(results_b, label_b)
            elif results_a is not None:
                print(f"\n  WARNING: target ({label_b}) benchmark failed.")
                print_single(results_a, label_a)

            save_record = {
                "baseline": label_a,
                "target": label_b,
                "results_baseline": results_a,
                "results_target": results_b,
            }

        if args.save:
            save_record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(args.save, "a") as f:
                f.write(json.dumps(save_record) + "\n")
            print(f"\nResults saved to {args.save}")

    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    if had_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
