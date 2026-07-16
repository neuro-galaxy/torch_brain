"""Tests for the per-item logging/stdio redirection in BrainsetPipeline.

Regression coverage for a bug where a pipeline calling `logging.basicConfig()`
at import time (several brainsets pipelines do) would bind its log handler to
the stream that was current *then*, rather than the one `_redirect_stdio`
later swaps in for each manifest item. Log records would then bypass the
per-item `pipeline_logs/<id>.err` file entirely.
"""

import logging

import pandas as pd
import pytest

from torch_brain.pipeline.pipeline import BrainsetPipeline


class _LoggingPipeline(BrainsetPipeline):
    """Minimal pipeline whose download() emits a log record and whose
    process() prints to stdout."""

    brainset_id = "logging_test"

    @classmethod
    def get_manifest(cls, raw_dir, args):
        return pd.DataFrame({"file": ["a"]})

    def download(self, manifest_item):
        logging.info("hello from download")
        return None

    def process(self, download_output):
        print("hello from process")


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """Prevent basicConfig(force=True) calls in these tests from leaking
    into other test files."""
    root_logger = logging.getLogger()
    handlers_before = root_logger.handlers[:]
    level_before = root_logger.level
    yield
    root_logger.handlers[:] = handlers_before
    root_logger.setLevel(level_before)


@pytest.fixture
def pipeline(tmp_path):
    return _LoggingPipeline(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        args=None,
    )


def test_logging_lands_in_per_item_log_file(pipeline, tmp_path):
    # Simulate a pipeline module that already called `logging.basicConfig()`
    # at import time, binding a handler to whatever stream was current then
    # (here: the test process's real stderr, captured before any redirect).
    # A plain StreamHandler is added instead of calling
    # `logging.basicConfig(force=True)`, which would tear down pytest's own
    # log-capture handler on the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.StreamHandler())

    pipeline._run_item_on_parallel_worker(pd.Series({"Index": "item0"}))

    log_file = tmp_path / "processed" / "pipeline_logs" / "item0.err"
    assert "hello from download" in log_file.read_text()


def test_print_output_lands_in_per_item_out_file(pipeline, tmp_path):
    # print() goes through plain sys.stdout redirection (no logging handler
    # involved), so this isn't exercising the bug above — it just confirms
    # _redirect_stdio's other half, the .out file, behaves as expected.
    pipeline._run_item_on_parallel_worker(pd.Series({"Index": "item0"}))

    log_file = tmp_path / "processed" / "pipeline_logs" / "item0.out"
    assert "hello from process" in log_file.read_text()


def test_restores_logging_handlers_after_item_runs(pipeline, tmp_path):
    root_logger = logging.getLogger()
    handlers_before = root_logger.handlers[:]
    level_before = root_logger.level

    pipeline._run_item_on_parallel_worker(pd.Series({"Index": "item0"}))

    assert root_logger.handlers == handlers_before
    assert root_logger.level == level_before
