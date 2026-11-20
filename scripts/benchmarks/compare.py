"""Compare torch_brain benchmarks across git commits.

Extracts torch_brain source from arbitrary commits via `git archive` and runs
the current benchmark.py against each, then displays a side-by-side comparison.

Usage:
    uv run python scripts/benchmarks/compare.py                      # benchmark working tree
    uv run python scripts/benchmarks/compare.py <commit>              # <commit> vs working tree
    uv run python scripts/benchmarks/compare.py <commitA> <commitB>   # commitA vs commitB

Options:
    --save PATH       Append comparison results (every batch time) as JSONL to PATH.
    --suite NAME      Which benchmark suite to run: data, utils, or all (default: all).
    --num-batches N   Timed batches per benchmark and side (default: 10).
    --markdown PATH   Also write a Markdown report to PATH (used for the PR comment):
                      summary, the flagged benchmarks, and full statistics
                      for every benchmark, collapsed.
    --regression-threshold X
                      Speedup below which a benchmark counts as a regression
                      (default: 0.95, i.e. more than 5% slower).
    --improvement-threshold X
                      Speedup above which a benchmark counts as an improvement
                      (default: 1.05, i.e. more than 5% faster).
    --alpha X         Significance level for the Holm-corrected p-values
                      (default: 0.05).

Speedup is baseline median / target median, so > 1 means the target is faster.
Significance computed with a two-sided Mann-Whitney U test on the per-batch times,
Holm-corrected across all compared benchmarks. A benchmark is flagged if it is
significant or past either threshold. The target only counts as a regression or an
improvement if it is both. A target-side error always counts as a regression.
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

import numpy as np
from harness import DEFAULT_NUM_BATCHES
from scipy.stats import mannwhitneyu


@dataclass(frozen=True)
class Thresholds:
    """Speedup bounds (anything in between is considered within noise) and the
    significance level for the Holm-corrected p-values."""

    regression: float = 0.95
    improvement: float = 1.05
    alpha: float = 0.05


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
    source_dir: str | None,
    label: str,
    suite: str = "all",
    num_batches: int = DEFAULT_NUM_BATCHES,
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
        [
            sys.executable,
            BENCH_SCRIPT,
            "--json",
            "--suite",
            suite,
            "--num-batches",
            str(num_batches),
        ],
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


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def summarize(times_us: list[float]) -> dict:
    """Median, mean, min and sample standard deviation of per-call times."""
    times = np.asarray(times_us, dtype=np.float64)
    return {
        "n": len(times),
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * (m - np.arange(m))
    adjusted_sorted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def _stats_or_status(r: dict | None) -> dict | str:
    if r is None:
        return "missing"
    if "error" in r:
        return "error"
    return summarize(r["times_us"])


def comparison_rows(results_a: list[dict], results_b: list[dict]) -> list[dict]:
    """Pair baseline/target results by label (baseline order, then target-only),

    Summary statistics, the speedup of the medians and Holm-corrected
    Mann-Whitney U p-values are included.
    """
    index_a = {r["label"]: r for r in results_a}
    index_b = {r["label"]: r for r in results_b}
    labels = list(index_a) + [lbl for lbl in index_b if lbl not in index_a]

    rows = []
    for label in labels:
        ra, rb = index_a.get(label), index_b.get(label)
        a, b = _stats_or_status(ra), _stats_or_status(rb)
        speedup = p_value = prob_faster = None
        if isinstance(a, dict) and isinstance(b, dict) and b["median"] > 0:
            speedup = a["median"] / b["median"]
            if a["n"] > 1 and b["n"] > 1:
                test = mannwhitneyu(
                    ra["times_us"], rb["times_us"], alternative="two-sided"
                )
                p_value = float(test.pvalue)
                # U / (n_a * n_b): probability that a baseline batch is slower
                # than a target batch (> 0.5 means the target tends to be faster)
                prob_faster = float(test.statistic) / (a["n"] * b["n"])
        rows.append(
            {
                "label": label,
                "a": a,
                "b": b,
                "speedup": speedup,
                "p": p_value,
                "p_adj": None,
                "prob_faster": prob_faster,
                "target_error": b == "error",
            }
        )

    tested = [row for row in rows if row["p"] is not None]
    for row, p_adj in zip(tested, holm([row["p"] for row in tested]), strict=True):
        row["p_adj"] = p_adj
    return rows


def classify(row: dict, thresholds: Thresholds) -> str | None:
    """Return "regression" or "improvement" (significant and past a threshold),
    "unconfirmed" (past a threshold but not significant), "minor_slower" or
    "minor_faster" (significant but within the thresholds), or None
    (unchanged / not comparable)."""
    # a target-side error is a regression too: the PR broke it
    if row["target_error"]:
        return "regression"
    if row["speedup"] is None:
        return None
    significant = row["p_adj"] is not None and row["p_adj"] < thresholds.alpha
    slower = row["speedup"] < thresholds.regression
    faster = row["speedup"] > thresholds.improvement
    if significant and slower:
        return "regression"
    if significant and faster:
        return "improvement"
    if slower or faster:
        return "unconfirmed"
    if significant:
        if row["speedup"] != 1.0:
            target_faster = row["speedup"] > 1.0
        else:
            # equal medians: take the direction the significance test detected
            target_faster = row["prob_faster"] > 0.5
        return "minor_faster" if target_faster else "minor_slower"
    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _time_decimals(us: float) -> int:
    """Decimal places that show ``us`` to 3 significant figures (none at or
    above 100 µs, so large times are never written with an exponent)."""
    us = float(f"{us:.3g}")
    if us <= 0:
        return 0
    return max(0, 2 - int(np.floor(np.log10(us))))


def _fmt_time(us: float, decimals: int | None = None) -> str:
    """Per-call time in µs, to 3 significant figures unless ``decimals`` is given."""
    if decimals is None:
        decimals = _time_decimals(us)
    return f"{us:,.{decimals}f}"


def _fmt_stat(side: dict | str, key: str) -> str:
    if side == "missing":
        return "n/a"
    if side == "error":
        return "ERROR"
    if key == "mean_std":
        # round the std to the mean's precision, e.g. "26.0 ± 0.9", not "26 ± 0.883"
        decimals = _time_decimals(side["mean"])
        mean = _fmt_time(side["mean"], decimals)
        std = _fmt_time(side["std"], decimals)
        return f"{mean} ± {std}"
    return _fmt_time(side[key])


def _fmt_speedup(row: dict) -> str:
    if row["target_error"]:
        return "ERROR"
    if row["speedup"] is None:
        return ""
    return f"{row['speedup']:.2f}x"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def print_single(results: list[dict], label: str):
    print(f"\n  Results for {label} (per-call times in µs)\n")
    print(f"  {'Benchmark':<42} {'Median':>10} {'Mean ± std':>22} {'Min':>10}")
    print(f"  {'-' * 87}")
    for r in results:
        side = _stats_or_status(r)
        print(
            f"  {r['label']:<42} {_fmt_stat(side, 'median'):>10} "
            f"{_fmt_stat(side, 'mean_std'):>22} {_fmt_stat(side, 'min'):>10}"
        )


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

    col_a = f"{label_a} median (µs)"
    col_b = f"{label_b} median (µs)"
    print(
        f"\n  {'Benchmark':<42} {col_a:>24} {col_b:>24} {'Speedup':>9} "
        f"{'p (corrected)':>9}  Flag"
    )
    print(f"  {'-' * 124}")

    for row in comparison_rows(results_a, results_b):
        kind = classify(row, thresholds)
        line = (
            f"  {row['label']:<42} {_fmt_stat(row['a'], 'median'):>24} "
            f"{_fmt_stat(row['b'], 'median'):>24} {_fmt_speedup(row):>9} "
            f"{_fmt_p(row['p_adj']):>9}  {kind or ''}"
        )
        color = colors.get(kind)
        print(f"{color}{line}{reset}" if color else line)


_MD_MARKERS = {
    "regression": "🔴",
    "improvement": "🟢",
    "minor_slower": "🔸",
    "minor_faster": "🔹",
    "unconfirmed": "⚠️",
}


def _md_label(label: str) -> str:
    # code span keeps "__or__" from rendering as bold; GitHub tables still need
    # "|" escaped inside code spans ("Interval.__or__ (1k|100)")
    return "`" + label.replace("|", "\\|") + "`"


def _md_flagged_table(
    rows: list[dict], kinds: list[str], label_a: str, label_b: str
) -> str:
    lines = [
        f"| | Benchmark | Baseline {label_a} median (µs) | Target {label_b} median (µs) | Speedup | p (Holm) |",
        "|---|---|--:|--:|--:|--:|",
    ]
    for row, kind in zip(rows, kinds, strict=True):
        lines.append(
            f"| {_MD_MARKERS[kind]} | {_md_label(row['label'])} "
            f"| {_fmt_stat(row['a'], 'median')} | {_fmt_stat(row['b'], 'median')} "
            f"| {_fmt_speedup(row)} | {_fmt_p(row['p_adj'])} |"
        )
    return "\n".join(lines)


def _md_full_table(
    rows: list[dict], kinds: list[str | None], label_a: str, label_b: str
) -> str:
    """Labels and units are stated once above the table to keep headers short."""
    lines = [
        f"**Baseline (B):** {label_a} · **Target (T):** {label_b} · Times in µs",
        "",
        "| | Benchmark | B median | B mean ± std | B min "
        "| T median | T mean ± std | T min | Speedup | p (Holm) |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row, kind in zip(rows, kinds, strict=True):
        a, b = row["a"], row["b"]
        marker = _MD_MARKERS[kind] if kind else ""
        lines.append(
            f"| {marker} | {_md_label(row['label'])} "
            f"| {_fmt_stat(a, 'median')} | {_fmt_stat(a, 'mean_std')} "
            f"| {_fmt_stat(a, 'min')} | {_fmt_stat(b, 'median')} "
            f"| {_fmt_stat(b, 'mean_std')} | {_fmt_stat(b, 'min')} "
            f"| {_fmt_speedup(row)} | {_fmt_p(row['p_adj'])} |"
        )
    return "\n".join(lines)


def _md_sort_key(item: tuple[dict, str]) -> tuple[int, float]:
    """Flag order, errors, then worst to best speedup first."""
    row, kind = item
    speedup = row["speedup"] if row["speedup"] is not None else 0.0
    marker_index = list(_MD_MARKERS).index(kind)
    return marker_index, speedup


def markdown_comparison(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Flagged benchmarks (significant or past either threshold, or erroring)
    are listed up front; full statistics for every benchmark are collapsed."""
    rows = comparison_rows(results_a, results_b)
    kinds = [classify(r, thresholds) for r in rows]
    n_regressed = kinds.count("regression")
    n_improved = kinds.count("improvement")
    n_other = sum(
        kinds.count(kind) for kind in ("unconfirmed", "minor_slower", "minor_faster")
    )
    n_tested = sum(row["p_adj"] is not None for row in rows)
    batch_counts = sorted(
        {
            side["n"]
            for row in rows
            for side in (row["a"], row["b"])
            if isinstance(side, dict)
        }
    )
    lo, hi = thresholds.regression, thresholds.improvement

    if n_regressed:
        summary = f"🔴 **{n_regressed} regressed**"
    else:
        summary = "✅ **No significant regressions**"
    if n_improved:
        summary += f" · 🟢 **{n_improved} improved**"
    if n_other:
        noun = "change" if n_other == 1 else "changes"
        summary += f" · {n_other} other {noun} flagged"
    summary += f" · {len(rows)} benchmarks"

    flagged = sorted(
        ((row, kind) for row, kind in zip(rows, kinds, strict=True) if kind),
        key=_md_sort_key,
    )

    parts = [summary, ""]
    if flagged:
        legend = [
            f"🔴 **regressed:** signif. and below {lo:.2f}x (or target-side error)",
            f"🟢 **improved:** signif. and above {hi:.2f}x",
            f"🔸 **slower:** signif., but above {lo:.2f}x",
            f"🔹 **faster:** signif., but below {hi:.2f}x",
            f"⚠️ **unconfirmed:** outside {lo:.2f}-{hi:.2f}x range, but not signif.",
        ]
        flagged_table = [
            "### Flagged",
            "",
            _md_flagged_table(
                [row for row, _ in flagged],
                [kind for _, kind in flagged],
                label_a,
                label_b,
            ),
            "",
            " \\\n".join(legend),  # hard line breaks
            "",
        ]
        parts += flagged_table

    details = [
        "<details>",
        f"<summary>Full statistics ({len(rows)} benchmarks)</summary>",
        "",
        _md_full_table(rows, kinds, label_a, label_b),
        "",
        "</details>",
        "",
    ]

    batches = "/".join(str(n) for n in batch_counts)
    notes = [
        f"Two-sided Mann-Whitney U test on the {batches} timed batches per "
        f"side (Holm-corrected across {n_tested} benchmarks). Significant if "
        f"p < {thresholds.alpha:g}.",
        "",
        f"Shared CI runners are noisy, so treat isolated changes close to "
        f"{lo:.2f}x or {hi:.2f}x with caution.",
    ]

    parts += details + notes

    return "\n".join(parts) + "\n"


def markdown_single(results: list[dict], label: str, warning: str = "") -> str:
    lines = [f"⚠️ **{warning}**", ""] if warning else []
    lines += [
        "<details>",
        f"<summary>Results for `{label}` ({len(results)} benchmarks)</summary>",
        "",
        "| Benchmark | Median (µs) | Mean ± std (µs) | Min (µs) |",
        "|---|--:|--:|--:|",
    ]
    for r in results:
        side = _stats_or_status(r)
        lines.append(
            f"| {_md_label(r['label'])} | {_fmt_stat(side, 'median')} "
            f"| {_fmt_stat(side, 'mean_std')} | {_fmt_stat(side, 'min')} |"
        )
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
        "--num-batches",
        type=int,
        default=DEFAULT_NUM_BATCHES,
        help="Timed number of batches per benchmark and side, after one warm-up batch "
        f"(default: {DEFAULT_NUM_BATCHES})",
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
        help="Speedup below which a significant change counts as a regression "
        f"(default: {Thresholds.regression})",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=Thresholds.improvement,
        help="Speedup above which a significant change counts as an improvement "
        f"(default: {Thresholds.improvement})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=Thresholds.alpha,
        help="Significance level for the Holm-corrected p-values "
        f"(default: {Thresholds.alpha})",
    )
    args = parser.parse_args()
    if args.regression_threshold > args.improvement_threshold:
        parser.error("--regression-threshold must be <= --improvement-threshold")
    if args.num_batches < 2:
        parser.error("--num-batches must be at least 2 to compute statistics")
    thresholds = Thresholds(
        args.regression_threshold, args.improvement_threshold, args.alpha
    )
    markdown = None

    if len(args.commits) > 2:
        parser.error("At most 2 commit refs can be provided.")

    tmpdirs: list[str] = []
    had_failures = False
    try:
        if len(args.commits) == 0:
            results = run_benchmark(None, "working tree", args.suite, args.num_batches)
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

            results_a = run_benchmark(tmpdir, label_a, args.suite, args.num_batches)
            results_b = run_benchmark(
                None, "working tree", args.suite, args.num_batches
            )

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

            results_a = run_benchmark(tmpdir_a, label_a, args.suite, args.num_batches)
            results_b = run_benchmark(tmpdir_b, label_b, args.suite, args.num_batches)

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
