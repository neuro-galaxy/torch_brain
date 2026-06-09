from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import ray

from torch_brain.pipeline import BrainsetPipeline
from torch_brain.pipeline import runner as runner_mod
from torch_brain.pipeline.runner import (
    StatusTracker,
    generate_status_table,
    import_pipeline_cls_from_file,
    run,
    run_pool_in_background,
    spin_on_tracker,
)

RECORDING_PIPELINE = Path(__file__).parent / "_recording_pipeline.py"
RECORDING_PIPELINE_WITH_ARGS = (
    Path(__file__).parent / "_recording_pipeline_with_args.py"
)


def test_import_pipeline_cls_from_file():
    pipeline_cls = import_pipeline_cls_from_file(Path(__file__).parent / "_pipeline.py")
    assert issubclass(pipeline_cls, BrainsetPipeline)
    assert pipeline_cls is not BrainsetPipeline


class TestGenerateStatusTable:
    def test_empty(self):
        result = generate_status_table({})
        assert "Summary:" in result
        assert "0" not in result

    def test_all_done(self):
        result = generate_status_table({"a": "DONE", "b": "DONE"})
        assert "a:" not in result
        assert "b:" not in result
        assert "2 [green]DONE[/]" in result

    def test_sorts_keys(self):
        result = generate_status_table({"z": "FAILED", "a": "DOWNLOADING"})
        lines = result.strip().split("\n")
        assert lines[0].startswith("a:")
        assert lines[1].startswith("z:")

    def test_omits_done_from_items(self):
        result = generate_status_table(
            {"done_item": "DONE", "failed_item": "FAILED", "dl_item": "DOWNLOADING"}
        )
        assert "done_item:" not in result
        assert "failed_item:" in result
        assert "dl_item:" in result
        assert "1 [green]DONE[/]" in result
        assert "1 [red]FAILED[/]" in result
        assert "1 [blue]DOWNLOADING[/]" in result

    def test_unknown_status_yellow(self):
        result = generate_status_table({"x": "SOME_STATUS"})
        assert "[yellow]SOME_STATUS[/]" in result

    def test_summary_has_all_statuses(self):
        result = generate_status_table({"a": "DONE", "b": "DOWNLOADING", "c": "FAILED"})
        summary_line = result.split("Summary: ")[1]
        assert len(summary_line.split(" | ")) == 3


class TestStatusTracker:
    """StatusTracker logic, tested on the undecorated class (no Ray needed).

    ``ray.remote`` keeps the original class available via ``__ray_metadata__``;
    we instantiate the plain Python class so its method bodies run in-process
    and are visible to coverage.
    """

    def _make_tracker(self):
        plain_cls = StatusTracker.__ray_metadata__.modified_class
        return plain_cls()

    def test_update_and_get_statuses(self):
        tracker = self._make_tracker()
        tracker.update_status("a", "DOWNLOADING")
        tracker.update_status("b", "DONE")
        tracker.update_status("a", "DONE")  # overwrite
        assert tracker.get_all_statuses() == {"a": "DONE", "b": "DONE"}

    def test_starts_empty(self):
        assert self._make_tracker().get_all_statuses() == {}


class TestRunPoolInBackground:
    """Real-Ray behavior test: proves work is distributed across actors.

    Excluded from coverage runs (Ray worker code executes out-of-process and
    pytest-cov + raylet segfault); see pyproject ``addopts``/markers.
    """

    @pytest.mark.ray
    def test_all_items_processed(self, ray_session):
        from ray.util.actor_pool import ActorPool

        @ray.remote
        class Counter:
            def __init__(self, tracker):
                self.tracker = tracker

            def _run_item_on_parallel_worker(self, item):
                self.tracker.update_status.remote(item, "DONE")

        tracker = StatusTracker.remote()
        actors = [Counter.remote(tracker) for _ in range(2)]
        pool = ActorPool(actors)
        items = [f"item_{i}" for i in range(10)]
        ray.get(run_pool_in_background.remote(pool, items))
        statuses = ray.get(tracker.get_all_statuses.remote())
        assert all(v == "DONE" for v in statuses.values())
        assert len(statuses) == 10


class TestSpinOnTracker:
    """spin_on_tracker drives the live TUI; Live is mocked so the loop runs
    headless and we only assert the success/failure return contract."""

    def _tracker_returning(self, statuses):
        tracker = MagicMock()
        tracker.get_all_statuses.remote.return_value = "ref"
        return tracker, statuses

    def test_returns_true_when_all_done(self):
        statuses = {"a": "DONE", "b": "DONE"}
        manifest = pd.DataFrame(index=["a", "b"])
        tracker = MagicMock()
        with (
            patch.object(runner_mod, "Live"),
            patch.object(runner_mod.ray, "get", return_value=statuses),
        ):
            assert spin_on_tracker(tracker, manifest) is True

    def test_returns_false_when_any_failed(self):
        statuses = {"a": "DONE", "b": "FAILED"}
        manifest = pd.DataFrame(index=["a", "b"])
        tracker = MagicMock()
        with (
            patch.object(runner_mod, "Live"),
            patch.object(runner_mod.ray, "get", return_value=statuses),
        ):
            assert spin_on_tracker(tracker, manifest) is False

    def test_spins_until_all_reported(self):
        # First poll: only one item reported (loop must continue).
        # Second poll: both reported and DONE (loop exits, returns True).
        polls = [{"a": "DONE"}, {"a": "DONE", "b": "DONE"}]
        manifest = pd.DataFrame(index=["a", "b"])
        tracker = MagicMock()
        with (
            patch.object(runner_mod, "Live"),
            patch.object(runner_mod.ray, "get", side_effect=polls),
            patch.object(runner_mod.time, "sleep"),
        ):
            assert spin_on_tracker(tracker, manifest) is True


class TestRunSingle:
    """run() --single path: no Ray, exercises the end-to-end driver flow."""

    def _argv(self, raw_dir, processed_dir, single):
        return [
            "runner",
            str(RECORDING_PIPELINE),
            f"--raw-dir={raw_dir}",
            f"--processed-dir={processed_dir}",
            f"--single={single}",
        ]

    def test_runs_one_item(self, tmp_path):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        argv = self._argv(raw_dir, processed_dir, "item_b")

        with patch.object(runner_mod.sys, "argv", argv):
            run()

        bset = processed_dir / "test_brainset"
        assert (bset / "downloaded.log").read_text() == "item_b\n"
        assert (bset / "processed.log").read_text() == "item_b\n"
        # raw/processed dirs are created under brainset_id
        assert (raw_dir / "test_brainset").is_dir()

    def test_download_only_skips_process(self, tmp_path):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        argv = self._argv(raw_dir, processed_dir, "item_a") + ["--download-only"]

        with patch.object(runner_mod.sys, "argv", argv):
            run()

        bset = processed_dir / "test_brainset"
        assert (bset / "downloaded.log").read_text() == "item_a\n"
        assert not (bset / "processed.log").exists()

    def test_pipeline_specific_args_are_parsed(self, tmp_path):
        # The pipeline defines its own parser; extra args must be forwarded
        # to it and reach the pipeline via self.args.
        processed_dir = tmp_path / "processed"
        argv = [
            "runner",
            str(RECORDING_PIPELINE_WITH_ARGS),
            f"--raw-dir={tmp_path / 'raw'}",
            f"--processed-dir={processed_dir}",
            "--single=only_item",
            "--flavor=chocolate",
        ]
        with patch.object(runner_mod.sys, "argv", argv):
            run()

        bset = processed_dir / "test_brainset_args"
        assert (bset / "flavor.log").read_text() == "chocolate\n"


class TestRunList:
    def test_list_prints_and_exits(self, tmp_path, capsys):
        argv = [
            "runner",
            str(RECORDING_PIPELINE),
            f"--raw-dir={tmp_path / 'raw'}",
            f"--processed-dir={tmp_path / 'processed'}",
            "--list",
        ]
        with patch.object(runner_mod.sys, "argv", argv):
            with pytest.raises(SystemExit) as exc:
                run()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Discovered 3 manifest items" in out
        assert "item_a" in out


class TestRunParallel:
    """run() parallel path with all Ray pieces mocked: verifies orchestration
    (init, actor pool sizing, dispatch, exit code) without a real cluster."""

    def _argv(self, raw_dir, processed_dir, cores):
        return [
            "runner",
            str(RECORDING_PIPELINE),
            f"--raw-dir={raw_dir}",
            f"--processed-dir={processed_dir}",
            f"-c{cores}",
        ]

    def test_success_path(self, tmp_path):
        argv = self._argv(tmp_path / "raw", tmp_path / "processed", cores=3)
        actor_cls = MagicMock()
        with (
            patch.object(runner_mod.ray, "init") as mock_init,
            patch.object(runner_mod.ray, "remote", return_value=actor_cls),
            patch.object(runner_mod, "StatusTracker"),
            patch.object(runner_mod, "ActorPool") as mock_pool,
            patch.object(runner_mod, "run_pool_in_background"),
            patch.object(runner_mod, "spin_on_tracker", return_value=True),
            patch.object(runner_mod.sys, "argv", argv),
        ):
            run()  # should not raise

        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs["num_cpus"] == 3
        # one actor created per core
        assert actor_cls.remote.call_count == 3
        mock_pool.assert_called_once()

    def test_failure_exits_nonzero(self, tmp_path):
        argv = self._argv(tmp_path / "raw", tmp_path / "processed", cores=2)
        with (
            patch.object(runner_mod.ray, "init"),
            patch.object(runner_mod.ray, "remote", return_value=MagicMock()),
            patch.object(runner_mod, "StatusTracker"),
            patch.object(runner_mod, "ActorPool"),
            patch.object(runner_mod, "run_pool_in_background"),
            patch.object(runner_mod, "spin_on_tracker", return_value=False),
            patch.object(runner_mod.sys, "argv", argv),
        ):
            with pytest.raises(SystemExit) as exc:
                run()
        assert exc.value.code == 1
