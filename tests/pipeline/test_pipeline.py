"""Unit tests for BrainsetPipeline base class, in particular the delete_raw feature."""

from argparse import Namespace

import pandas as pd
import pytest

from torch_brain.pipeline.pipeline import BrainsetPipeline


class _DummyPipeline(BrainsetPipeline):
    """Minimal concrete pipeline for exercising base-class behavior."""

    brainset_id = "dummy"

    @classmethod
    def get_manifest(cls, raw_dir, args):
        return pd.DataFrame({"file": ["a"]})

    def download(self, manifest_item):
        return self._download_return

    def process(self, download_output):
        self.processed = download_output


@pytest.fixture
def make_pipeline(tmp_path):
    def _make(download_return, delete_raw=True, download_only=False):
        pipeline = _DummyPipeline(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            args=Namespace(),
            delete_raw=delete_raw,
            download_only=download_only,
        )
        pipeline._download_return = download_return
        return pipeline

    return _make


class TestExtractRawPaths:
    def test_single_path(self, tmp_path):
        assert BrainsetPipeline._extract_raw_paths(tmp_path) == [tmp_path]

    def test_ignores_plain_strings(self, tmp_path):
        # Only actual Path instances should be picked up, not arbitrary strings,
        # to avoid deleting unrelated files that happen to share a name.
        assert BrainsetPipeline._extract_raw_paths(str(tmp_path)) == []
        assert BrainsetPipeline._extract_raw_paths("some_session_type") == []

    def test_nested_tuple(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        result = BrainsetPipeline._extract_raw_paths((a, "session_type", 42, [b]))
        assert set(result) == {a, b}

    def test_object_with_path_attribute(self, tmp_path):
        class Asset:
            path = tmp_path / "asset.h5"

        assert BrainsetPipeline._extract_raw_paths(Asset()) == [Asset.path]

    def test_unsupported_value_returns_empty(self):
        assert BrainsetPipeline._extract_raw_paths(object()) == []


class TestCleanupRaw:
    def test_deletes_single_file(self, make_pipeline, tmp_path):
        fpath = tmp_path / "raw_file.mat"
        fpath.write_text("data")
        pipeline = make_pipeline(fpath)

        pipeline.cleanup_raw(fpath)

        assert not fpath.exists()

    def test_deletes_directory(self, make_pipeline, tmp_path):
        raw_dir = tmp_path / "extracted"
        raw_dir.mkdir()
        (raw_dir / "file.bin").write_text("data")
        pipeline = make_pipeline(raw_dir)

        pipeline.cleanup_raw(raw_dir)

        assert not raw_dir.exists()

    def test_deletes_multiple_files_from_tuple(self, make_pipeline, tmp_path):
        psg = tmp_path / "psg.edf"
        hyp = tmp_path / "hyp.edf"
        psg.write_text("data")
        hyp.write_text("data")
        pipeline = make_pipeline((psg, hyp))

        pipeline.cleanup_raw((psg, hyp))

        assert not psg.exists()
        assert not hyp.exists()

    def test_warns_when_no_path_found(self, make_pipeline, caplog):
        pipeline = make_pipeline("not_a_path")

        with caplog.at_level("WARNING"):
            pipeline.cleanup_raw("not_a_path")

        assert "delete_raw is enabled" in caplog.text

    def test_missing_file_does_not_raise(self, make_pipeline, tmp_path):
        fpath = tmp_path / "already_gone.mat"
        pipeline = make_pipeline(fpath)

        pipeline.cleanup_raw(fpath)  # should not raise


class TestRunItemDeleteRaw:
    def test_delete_raw_disabled_keeps_file(self, make_pipeline, tmp_path):
        fpath = tmp_path / "raw_file.mat"
        fpath.write_text("data")
        pipeline = make_pipeline(fpath, delete_raw=False)

        pipeline._run_item(pd.Series({"Index": "item0"}))

        assert fpath.exists()

    def test_delete_raw_enabled_deletes_after_process(self, make_pipeline, tmp_path):
        fpath = tmp_path / "raw_file.mat"
        fpath.write_text("data")
        pipeline = make_pipeline(fpath, delete_raw=True)

        pipeline._run_item(pd.Series({"Index": "item0"}))

        assert not fpath.exists()
        assert pipeline.processed == fpath

    def test_download_only_skips_process_and_cleanup(self, make_pipeline, tmp_path):
        fpath = tmp_path / "raw_file.mat"
        fpath.write_text("data")
        pipeline = make_pipeline(fpath, delete_raw=True, download_only=True)

        pipeline._run_item(pd.Series({"Index": "item0"}))

        assert fpath.exists()
        assert not hasattr(pipeline, "processed")

    def test_cleanup_not_called_if_process_raises(self, make_pipeline, tmp_path):
        fpath = tmp_path / "raw_file.mat"
        fpath.write_text("data")
        pipeline = make_pipeline(fpath, delete_raw=True)

        def _raise(download_output):
            raise RuntimeError("boom")

        pipeline.process = _raise

        with pytest.raises(RuntimeError):
            pipeline._run_item(pd.Series({"Index": "item0"}))

        assert fpath.exists()
