"""Compare temporaldata benchmarks across git commits.

Extracts temporaldata source from arbitrary commits via `git archive` and
runs the current benchmark.py against each, then displays a side-by-side
comparison table.

Usage:
    uv run python benchmarks/compare.py                      # benchmark working tree
    uv run python benchmarks/compare.py <commit>              # <commit> vs working tree
    uv run python benchmarks/compare.py <commitA> <commitB>   # commitA vs commitB

Options:
    --save PATH   Append comparison results as JSONL to PATH.
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
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")


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


def extract_source(commit: str) -> str:
    """Extract temporaldata/ from a commit into a temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="tdbench_")
    git_proc = subprocess.run(
        ["git", "archive", commit, "--", "temporaldata/"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if git_proc.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(
            (
                f"Error: git archive failed for {short_hash(commit)}: "
                f"{git_proc.stderr.decode(errors='replace').strip()}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    tar_proc = subprocess.run(
        ["tar", "xf", "-", "-C", tmpdir],
        input=git_proc.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tar_proc.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(
            (
                f"Error: tar extraction failed for {short_hash(commit)}: "
                f"{tar_proc.stderr.decode(errors='replace').strip()}"
            ),
            file=sys.stderr,
        )
        sys.exit(1)
    return tmpdir


def run_benchmark(source_dir: str | None, label: str) -> list[dict]:
    """Run benchmark.py, optionally overriding the import source."""
    env = os.environ.copy()
    if source_dir is not None:
        env["TEMPORALDATA_SOURCE"] = source_dir

    print(f"Running benchmarks for {label}...")
    result = subprocess.run(
        [sys.executable, BENCH_SCRIPT, "--json"],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"Benchmark run failed for {label}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output for {label}:", file=sys.stderr)
        print(result.stdout[:500], file=sys.stderr)
        sys.exit(1)

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
        description="Compare temporaldata benchmarks across git commits.",
        epilog="Examples:\n"
        "  uv run python benchmarks/compare.py\n"
        "  uv run python benchmarks/compare.py abc123\n"
        "  uv run python benchmarks/compare.py abc123 def456\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "commits", nargs="*", help="0, 1, or 2 commit refs to benchmark"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Append results to a JSONL file"
    )
    args = parser.parse_args()

    if len(args.commits) > 2:
        parser.error("At most 2 commit refs can be provided.")

    tmpdirs: list[str] = []
    try:
        if len(args.commits) == 0:
            results = run_benchmark(None, "working tree")
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

            results_a = run_benchmark(tmpdir, label_a)
            results_b = run_benchmark(None, "working tree")
            print_comparison(results_a, results_b, label_a, "working tree")
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

            results_a = run_benchmark(tmpdir_a, label_a)
            results_b = run_benchmark(tmpdir_b, label_b)
            print_comparison(results_a, results_b, label_a, label_b)
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


if __name__ == "__main__":
    main()
