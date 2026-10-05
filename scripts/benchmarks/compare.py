"""Compare torch_brain benchmarks across git commits.

Extracts torch_brain source from arbitrary commits via `git archive` and
runs the current benchmark.py against each, then displays a side-by-side
comparison table.

Usage:
    uv run python scripts/benchmarks/compare.py                      # benchmark working tree
    uv run python scripts/benchmarks/compare.py <commit>              # <commit> vs working tree
    uv run python scripts/benchmarks/compare.py <commitA> <commitB>   # commitA vs commitB

Options:
    --save PATH       Append comparison results as JSONL to PATH.
    --suite NAME      Which benchmark suite to run: data, utils, or all (default: all).
    --markdown PATH   Also write a Markdown report to PATH (used for the PR comment):
                      summary, the outliers past either threshold (marked 🔴/🟢),
                      and every other benchmark collapsed.
    --regression-threshold X
                      Speedup below which a benchmark is a regression, shown in
                      red (default: 0.95, i.e. more than 5% slower).
    --improvement-threshold X
                      Speedup above which a benchmark is an improvement, shown in
                      green (default: 1.05, i.e. more than 5% faster).

Speedup is baseline time / target time, so > 1 means the target is faster.
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
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Speedup bounds; anything in between is reported as unchanged."""

    regression: float = 0.95
    improvement: float = 1.05


DEFAULT_THRESHOLDS = Thresholds()


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


def _fmt_us(r: dict | None) -> str:
    if r is None:
        return "n/a"
    if "error" in r:
        return "ERROR"
    return f"{r['mean_us']:.3e}"


def comparison_rows(results_a: list[dict], results_b: list[dict]) -> list[dict]:
    """Pair baseline/target results by label (baseline order, then target-only)."""
    index_a = {r["label"]: r for r in results_a}
    index_b = {r["label"]: r for r in results_b}
    labels = list(index_a) + [lbl for lbl in index_b if lbl not in index_a]

    rows = []
    for label in labels:
        ra, rb = index_a.get(label), index_b.get(label)
        speedup = None
        if ra and rb and "error" not in ra and "error" not in rb and rb["mean_us"] > 0:
            speedup = ra["mean_us"] / rb["mean_us"]
        rows.append(
            {
                "label": label,
                "a": _fmt_us(ra),
                "b": _fmt_us(rb),
                "speedup": speedup,
                "target_error": rb is not None and "error" in rb,
            }
        )
    return rows


def classify(row: dict, thresholds: Thresholds) -> str | None:
    """Return "regression", "improvement", or None (unchanged / not comparable)."""
    # a target-side error is a regression too: the PR broke it
    if row["target_error"]:
        return "regression"
    if row["speedup"] is None:
        return None
    if row["speedup"] < thresholds.regression:
        return "regression"
    if row["speedup"] > thresholds.improvement:
        return "improvement"
    return None


def print_single(results: list[dict], label: str):
    print(f"\n  Results for {label}\n")
    print(f"  {'Benchmark':<42} {'Iters':>8} {'Mean (µs)':>12}")
    print(f"  {'-' * 65}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<42} {'ERROR':>8} {'---':>12}")
        else:
            print(f"  {r['label']:<42} {r['number']:>8} {r['mean_us']:>12.3e}")


def print_comparison(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
):
    # only emit ANSI colors on a terminal, never into tee'd CI logs/files
    tty = sys.stdout.isatty()
    colors = {"regression": "\033[31m", "improvement": "\033[32m"} if tty else {}
    reset = "\033[0m" if tty else ""

    col_a = f"{label_a} (µs)"
    col_b = f"{label_b} (µs)"
    print(f"\n  {'Benchmark':<42} {col_a:>18} {col_b:>18} {'Speedup':>10}")
    print(f"  {'-' * 92}")

    for row in comparison_rows(results_a, results_b):
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] is not None else ""
        line = f"  {row['label']:<42} {row['a']:>18} {row['b']:>18} {speedup:>10}"
        color = colors.get(classify(row, thresholds))
        print(f"{color}{line}{reset}" if color else line)


_MD_MARKERS = {"regression": "🔴", "improvement": "🟢"}


def _md_label(label: str) -> str:
    # code span keeps "__or__" from rendering as bold; GitHub tables still need
    # "|" escaped inside code spans ("Interval.__or__ (1k|100)")
    return "`" + label.replace("|", "\\|") + "`"


def _md_table(
    rows: list[dict], label_a: str, label_b: str, thresholds: Thresholds
) -> str:
    lines = [
        f"| Benchmark | `{label_a}` (µs) | `{label_b}` (µs) | Speedup |",
        "|---|--:|--:|--:|",
    ]
    for row in rows:
        kind = classify(row, thresholds)
        label = _md_label(row["label"])
        if kind:
            label = f"{_MD_MARKERS[kind]} {label}"
        if row["target_error"]:
            speedup = "ERROR"
        elif row["speedup"] is not None:
            speedup = f"{row['speedup']:.2f}x"
        else:
            speedup = ""
        lines.append(f"| {label} | {row['a']} | {row['b']} | {speedup} |")
    return "\n".join(lines)


def markdown_comparison(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Outliers (past either threshold, or erroring) are listed up front; every
    other benchmark is collapsed, so each one appears exactly once."""
    rows = comparison_rows(results_a, results_b)
    kinds = [classify(r, thresholds) for r in rows]
    n_regressed = kinds.count("regression")
    n_improved = kinds.count("improvement")
    lo, hi = thresholds.regression, thresholds.improvement

    outliers = [r for r, k in zip(rows, kinds, strict=True) if k]
    unchanged = [r for r, k in zip(rows, kinds, strict=True) if not k]

    if n_regressed:
        summary = f"🔴 **{n_regressed} regressed** (< {lo:.2f}x or error)"
    else:
        summary = "✅ **No regressions**"
    if n_improved:
        summary += f" · 🟢 **{n_improved} improved** (> {hi:.2f}x)"
    summary += f" · {len(rows)} benchmarks"

    def table(subset):
        return _md_table(subset, label_a, label_b, thresholds)

    parts = [summary, ""]
    if outliers:
        parts += ["### Outliers", "", table(outliers), ""]
    if unchanged:
        parts += [
            "<details>",
            f"<summary>{len(unchanged)} unchanged benchmarks "
            f"({lo:.2f}x – {hi:.2f}x)</summary>",
            "",
            table(unchanged),
            "",
            "</details>",
            "",
        ]
    parts += [
        f"Shared CI runners are noisy, so treat isolated changes close to "
        f"{lo:.2f}x or {hi:.2f}x with caution.",
    ]
    return "\n".join(parts) + "\n"


def markdown_single(results: list[dict], label: str, warning: str = "") -> str:
    lines = [f"⚠️ **{warning}**", ""] if warning else []
    lines += [
        "<details>",
        f"<summary>Results for `{label}` ({len(results)} benchmarks)</summary>",
        "",
        f"| Benchmark | `{label}` (µs) |",
        "|---|--:|",
    ]
    lines += [f"| {_md_label(r['label'])} | {_fmt_us(r)} |" for r in results]
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def report_comparison(
    results_a: list[dict] | None,
    results_b: list[dict] | None,
    label_a: str,
    label_b: str,
    thresholds: Thresholds,
) -> str | None:
    """Print the comparison (or whichever side succeeded) and return Markdown."""
    if results_a is not None and results_b is not None:
        print_comparison(results_a, results_b, label_a, label_b, thresholds)
        return markdown_comparison(results_a, results_b, label_a, label_b, thresholds)
    if results_b is not None:
        warning = f"Baseline ({label_a}) benchmark failed; no comparison."
        print(f"\n  WARNING: {warning}")
        print_single(results_b, label_b)
        return markdown_single(results_b, label_b, warning)
    if results_a is not None:
        warning = f"Target ({label_b}) benchmark failed; no comparison."
        print(f"\n  WARNING: {warning}")
        print_single(results_a, label_a)
        return markdown_single(results_a, label_a, warning)
    return "❌ **Both benchmark runs failed.** See the workflow logs.\n"


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
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        help="Also write a Markdown report to this path",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=Thresholds.regression,
        help="Speedup below which a benchmark is flagged as a regression "
        f"(default: {Thresholds.regression})",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=Thresholds.improvement,
        help="Speedup above which a benchmark is flagged as an improvement "
        f"(default: {Thresholds.improvement})",
    )
    args = parser.parse_args()
    if args.regression_threshold > args.improvement_threshold:
        parser.error("--regression-threshold must be <= --improvement-threshold")
    thresholds = Thresholds(args.regression_threshold, args.improvement_threshold)
    markdown = None

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
                markdown = markdown_single(results, "working tree")
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
            markdown = report_comparison(
                results_a, results_b, label_a, "working tree", thresholds
            )

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
            markdown = report_comparison(
                results_a, results_b, label_a, label_b, thresholds
            )

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

        if args.markdown and markdown is not None:
            with open(args.markdown, "w") as f:
                f.write(markdown)
            print(f"Markdown report written to {args.markdown}")

    finally:
        for d in tmpdirs:
            shutil.rmtree(d, ignore_errors=True)

    if had_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
