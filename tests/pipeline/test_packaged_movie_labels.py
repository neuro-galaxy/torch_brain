from pathlib import Path

import pytest
from _utils import add_pipelines_to_path

add_pipelines_to_path()
from berezutskaya_pippi_2022 import pipeline as pippi_pipeline  # noqa: E402
from keles_byd_2024 import pipeline as byd_pipeline  # noqa: E402


@pytest.mark.parametrize("pipeline_module", [byd_pipeline, pippi_pipeline])
def test_labels_dir_defaults_to_packaged_csvs(pipeline_module):
    args = pipeline_module.parser.parse_args([])

    labels_dir = pipeline_module._resolve_labels_dir(args.labels_dir)

    assert labels_dir == pipeline_module.PIPELINE_DIR / "labels"
    assert len(list(labels_dir.glob("*.csv"))) > 0


@pytest.mark.parametrize("pipeline_module", [byd_pipeline, pippi_pipeline])
def test_labels_dir_explicit_override_is_preserved(
    pipeline_module, tmp_path: Path
):
    labels_dir = tmp_path / "custom-labels"
    labels_dir.mkdir()
    (labels_dir / "custom.csv").write_text("start_time_s,label\n0.0,1\n")
    args = pipeline_module.parser.parse_args(["--labels-dir", str(labels_dir)])

    assert pipeline_module._resolve_labels_dir(args.labels_dir) == labels_dir


@pytest.mark.parametrize("pipeline_module", [byd_pipeline, pippi_pipeline])
def test_missing_packaged_labels_fail_clearly(
    pipeline_module, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(pipeline_module, "PIPELINE_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Packaged label directory not found"):
        pipeline_module._resolve_labels_dir(None)


@pytest.mark.parametrize("pipeline_module", [byd_pipeline, pippi_pipeline])
def test_empty_explicit_labels_dir_fails_clearly(pipeline_module, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Label directory contains no CSV"):
        pipeline_module._resolve_labels_dir(tmp_path)


def test_byd_process_passes_packaged_default_to_processor(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipeline"
    labels_dir = pipeline_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "speech_binary_labels.csv").write_text("start_time_s,label\n0,1\n")
    monkeypatch.setattr(byd_pipeline, "PIPELINE_DIR", pipeline_dir)
    captured = {}
    monkeypatch.setattr(
        byd_pipeline,
        "process_file",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    pipeline = byd_pipeline.Pipeline(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        args=byd_pipeline.parser.parse_args([]),
    )

    pipeline.process(tmp_path / "sub-CS41_ses-P41CSR1_behavior+ecephys.nwb")

    assert captured["labels_dir"] == str(labels_dir)


def test_pippi_process_passes_packaged_default_to_processor(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "pipeline"
    labels_dir = pipeline_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "speech_binary_labels.csv").write_text("start_time_s,label\n0,1\n")
    monkeypatch.setattr(pippi_pipeline, "PIPELINE_DIR", pipeline_dir)
    captured = {}
    monkeypatch.setattr(
        pippi_pipeline,
        "process_file",
        lambda **kwargs: captured.update(kwargs),
    )
    pipeline = pippi_pipeline.Pipeline(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        args=pippi_pipeline.parser.parse_args([]),
    )
    recording = tmp_path / "sub-01_ses-iemu_task-film_acq-clinical_run-1_ieeg.vhdr"

    pipeline.process(recording)

    assert captured["labels_dir"] == labels_dir
