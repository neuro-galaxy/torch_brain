import sys
from typing import Type
from argparse import ArgumentParser
import os
import time
from collections import defaultdict
from typing import Dict, Any, List
from pathlib import Path
import ray
from ray.util.actor_pool import ActorPool

from rich.live import Live
from rich.console import Console
import pandas as pd

from brainsets.pipeline import BrainsetPipeline


def import_pipeline_cls_from_file(pipeline_filepath: Path) -> Type[BrainsetPipeline]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_filepath)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    return pipeline_module.Pipeline


@ray.remote
class StatusTracker:
    def __init__(self):
        self.statuses: Dict[Any, str] = {}

    def update_status(self, id: Any, status: str):
        self.statuses[id] = status

    def get_all_statuses(self):
        return self.statuses


def get_style(status: str) -> str:
    if "DONE" in status:
        return "green"
    elif "FAILED" in status:
        return "red"
    elif "DOWNLOADING" in status:
        return "blue"
    else:
        return "yellow"


def generate_status_table(status_dict: Dict[Any, str]) -> str:
    # Sort files for a consistent display
    sorted_files = sorted(status_dict.keys())

    ans = ""
    for file_id in sorted_files:
        status = status_dict[file_id]
        if status == "DONE":
            continue
        status_style = get_style(status)
        ans += f"{file_id}: [{status_style}]{status}[/]\n"

    summary_dict = defaultdict(int)
    for v in status_dict.values():
        summary_dict[v] += 1

    summary_list = [f"{v} [{get_style(k)}]{k}[/]" for k, v in summary_dict.items()]
    ans += "Summary: " + " | ".join(summary_list)
    return ans


@ray.remote
def run_pool_in_background(actor_pool: ActorPool, work_items: List[Any]):
    results_generator = actor_pool.map_unordered(
        lambda actor, task: actor._run_item_on_parallel_worker.remote(task),
        work_items,
    )

    # For the actors to start working, we need to request results by
    # iterating over the results_generator.
    for _ in results_generator:
        pass


def spin_on_tracker(tracker, manifest):
    """
    Spins until all manifest items are DONE or FAILED, updating and displaying progress/status
    in the terminal.

    Returns:
        bool: True if all manifest items were processed successfully (all status are DONE),
              False if any manifest items failed (any status is FAILED).
    """
    console = Console()
    with Live(
        generate_status_table({}),
        console=console,
        refresh_per_second=10,
    ) as live:
        all_fin = False
        while not all_fin:  # Spin loop
            # Get status and update TUI
            status_dict = ray.get(tracker.get_all_statuses.remote())
            live.update(generate_status_table(status_dict))

            # Check for completion
            if len(status_dict) == len(manifest):
                all_fin = all(s in ["DONE", "FAILED"] for s in status_dict.values())

            # Sleep to prevent loop from spinning too fast
            time.sleep(0.1)

    if all(s == "DONE" for s in status_dict.values()):
        console.print("\n[bold green]All manifest items processed[/]")
        return True
    else:
        num_failed = sum(s == "FAILED" for s in status_dict.values())
        console.print(f"\n[bold red]{num_failed} manifest items failed[/]")
        return False


def run():
    parser = ArgumentParser()
    parser.add_argument("pipeline_file", type=Path)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("-s", "--single", default=None, type=str)
    parser.add_argument("-c", "--cores", default=4, type=int)
    parser.add_argument("--list", action="store_true", help="List manifest and exit")
    parser.add_argument(
        "--download-only", action="store_true", help="Download raw data and exit"
    )
    args, remaining_args = parser.parse_known_args()

    pipeline_cls = import_pipeline_cls_from_file(args.pipeline_file)

    # Parse pipeline specific arguments
    pipeline_args = None
    if pipeline_cls.parser is not None:
        pipeline_args = pipeline_cls.parser.parse_args(remaining_args)

    # Set raw and processed dir
    raw_dir = args.raw_dir / pipeline_cls.brainset_id
    processed_dir = args.processed_dir / pipeline_cls.brainset_id

    manifest = pipeline_cls.get_manifest(
        raw_dir=raw_dir,
        args=pipeline_args,
    )
    print(f"Discovered {len(manifest)} manifest items")

    if args.list:
        with pd.option_context("display.max_rows", None):
            print(manifest)
        sys.exit(0)

    if args.single is None:
        # Parallel run

        # 1. Start ray
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # to avoid a warning
        ray.init("local", num_cpus=args.cores, log_to_driver=False)

        # 2. Start tracker and actors
        tracker = StatusTracker.remote()
        actor_cls = ray.remote(pipeline_cls)
        actor_pool = ActorPool(
            [
                actor_cls.remote(
                    tracker_handle=tracker,
                    raw_dir=raw_dir,
                    processed_dir=processed_dir,
                    args=pipeline_args,
                    download_only=args.download_only,
                )
                for _ in range(args.cores)
            ]
        )
        run_pool_in_background.remote(actor_pool, list(manifest.itertuples()))

        success = spin_on_tracker(tracker, manifest)
        if not success:
            sys.exit(1)
    else:
        # Single run
        manifest_item = manifest.loc[args.single]
        manifest_item.Index = args.single
        pipeline = pipeline_cls(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            args=pipeline_args,
            download_only=args.download_only,
        )
        pipeline._run_item(manifest_item)


if __name__ == "__main__":
    run()
