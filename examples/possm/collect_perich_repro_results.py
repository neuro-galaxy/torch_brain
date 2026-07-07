import argparse
import pathlib
import re
import statistics


GROUPS = {
    "C-CO 2016": (range(0, 2), 0.9549, 0.0012),
    "T-CO": (range(2, 8), 0.8863, 0.0222),
    "T-RT": (range(8, 14), 0.7687, 0.0669),
}


def read_metric(path: pathlib.Path) -> float:
    text = path.read_text(errors="ignore")
    lines = [line for line in text.splitlines() if "average_test_metric" in line]
    if not lines:
        raise RuntimeError(f"No average_test_metric found in {path}")
    return float(re.findall(r"-?\d+(?:\.\d+)?", lines[-1])[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect POSSM Perich single-session reproduction metrics."
    )
    parser.add_argument("job_id", help="SLURM array job id, e.g. 10066755")
    parser.add_argument(
        "--log-root",
        type=pathlib.Path,
        default=pathlib.Path("logs/possm_perich_repro/slurm"),
        help="Directory containing SLURM array logs named <job_id>_<task_id>.out.",
    )
    args = parser.parse_args()

    complete = True
    for name, (task_ids, ref_mean, ref_std) in GROUPS.items():
        values = []
        for task_id in task_ids:
            path = args.log_root / f"{args.job_id}_{task_id}.out"
            if not path.exists():
                print(f"{name} task {task_id}: missing {path}")
                complete = False
                continue
            try:
                values.append(read_metric(path))
            except RuntimeError as exc:
                print(exc)
                complete = False

        print(name, [round(value, 4) for value in values])
        if len(values) == len(list(task_ids)):
            mean = statistics.mean(values)
            std = statistics.pstdev(values)
            print(
                f"  {mean:.4f} ± {std:.4f}; "
                f"reference {ref_mean:.4f} ± {ref_std:.4f}; "
                f"delta {mean - ref_mean:+.4f}"
            )

    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
