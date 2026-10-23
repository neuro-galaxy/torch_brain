"""Tests for brainsets plugins: discovery, name resolution, and the list/prepare
commands. A fake plugin package is written to a temporary directory and exposed
through a patched entry point list.
"""

import sys
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from torch_brain.pipeline._cli import plugins as plugins_mod
from torch_brain.pipeline._cli.cli import cli
from torch_brain.pipeline._cli.plugins import (
    BrainsetNotFoundError,
    get_builtin_source,
    get_plugin_sources,
    resolve_brainset,
    split_name,
)

GROUP = "brainsets.pipelines"

PIPELINE_TEMPLATE = """\
# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["numpy"]
# ///

from torch_brain.pipeline import BrainsetPipeline


class Pipeline(BrainsetPipeline):
    brainset_id = "{brainset_id}"

    @classmethod
    def get_manifest(cls, raw_dir, args):
        raise NotImplementedError

    def download(self, manifest_item):
        raise NotImplementedError

    def process(self, download_output):
        raise NotImplementedError
"""


def _write_plugin(root: Path, package: str, brainsets: dict[str, str]) -> None:
    """Write ``<root>/<package>/pipelines/<folder>/pipeline.py`` for each brainset."""
    pkg = root / package
    (pkg / "pipelines").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "pipelines" / "__init__.py").write_text("")
    for folder, source in brainsets.items():
        (pkg / "pipelines" / folder).mkdir()
        (pkg / "pipelines" / folder / "pipeline.py").write_text(source)
    # Ignored folders
    (pkg / "pipelines" / "_template").mkdir()
    (pkg / "pipelines" / "_template" / "pipeline.py").write_text("")
    (pkg / "pipelines" / "no_pipeline").mkdir()


@pytest.fixture
def fake_plugin(tmp_path, monkeypatch):
    """Install a fake 'fake' plugin (package 'fakeplug') with one brainset."""
    root = tmp_path / "site"
    _write_plugin(
        root,
        "fakeplug",
        {"fake_ds": PIPELINE_TEMPLATE.format(brainset_id="fake_ds")},
    )
    monkeypatch.syspath_prepend(str(root))
    entry_points = [EntryPoint("fake", "fakeplug.pipelines", GROUP)]
    monkeypatch.setattr(plugins_mod, "_iter_entry_points", lambda: entry_points)
    yield entry_points
    # Forget the fake packages so the next test imports its own copy
    for name in list(sys.modules):
        if name.split(".")[0] in ("fakeplug", "otherplug"):
            del sys.modules[name]


@pytest.fixture
def no_plugins(monkeypatch):
    monkeypatch.setattr(plugins_mod, "_iter_entry_points", lambda: [])


class TestDiscovery:
    def test_builtin_brainsets_listed(self):
        brainsets = get_builtin_source().brainsets()
        assert "pei_pandarinath_nlb_2021" in brainsets

    def test_plugin_discovered(self, fake_plugin):
        sources = get_plugin_sources()
        assert list(sources) == ["fake"]
        source = sources["fake"]
        assert source.error is None
        assert source.dist_name == "fakeplug"
        assert list(source.brainsets()) == ["fake_ds"]
        assert source.qualified_name("fake_ds") == "fake:fake_ds"

    @pytest.mark.parametrize(
        "name, value, error",
        [
            ("Bad-Name", "fakeplug.pipelines", "invalid namespace"),
            ("torch_brain", "fakeplug.pipelines", "reserved"),
            ("fake", "fakeplug.pipelines:obj", "must be a package name"),
            ("fake", "does_not_exist.pipelines", "could not load package"),
        ],
    )
    def test_broken_plugin_is_reported_not_raised(
        self, fake_plugin, monkeypatch, name, value, error
    ):
        entry_points = [EntryPoint(name, value, GROUP)]
        monkeypatch.setattr(plugins_mod, "_iter_entry_points", lambda: entry_points)
        with pytest.warns(UserWarning, match="Skipping brainsets plugin"):
            sources = get_plugin_sources()
        (source,) = sources.values()
        assert error in source.error
        assert source.brainsets() == {}

    def test_namespace_clash_keeps_first(self, fake_plugin, tmp_path, monkeypatch):
        _write_plugin(tmp_path / "other", "otherplug", {})
        monkeypatch.syspath_prepend(str(tmp_path / "other"))
        entry_points = fake_plugin + [EntryPoint("fake", "otherplug.pipelines", GROUP)]
        monkeypatch.setattr(plugins_mod, "_iter_entry_points", lambda: entry_points)
        sources = get_plugin_sources(warn=False)
        assert sources["fake"].error is None
        clash = [s for s in sources.values() if s.error is not None]
        assert len(clash) == 1
        assert "already used by" in clash[0].error


class TestResolve:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("pei_pandarinath_nlb_2021", ("torch_brain", "pei_pandarinath_nlb_2021")),
            ("openneuro:klinzing", ("openneuro", "klinzing")),
        ],
    )
    def test_split_name(self, name, expected):
        assert split_name(name) == expected

    def test_builtin_brainset(self, no_plugins):
        resolved = resolve_brainset("pei_pandarinath_nlb_2021")
        assert resolved.source.is_builtin
        assert resolved.qualified_name == "pei_pandarinath_nlb_2021"

    def test_builtin_namespace_is_rejected(self, no_plugins):
        with pytest.raises(BrainsetNotFoundError, match="without a namespace"):
            resolve_brainset("torch_brain:pei_pandarinath_nlb_2021")

    def test_plugin_brainset(self, fake_plugin):
        resolved = resolve_brainset("fake:fake_ds")
        assert resolved.source.namespace == "fake"
        assert resolved.brainset_id == "fake_ds"
        assert resolved.qualified_name == "fake:fake_ds"

    def test_bare_plugin_name_suggests_qualified(self, fake_plugin):
        with pytest.raises(BrainsetNotFoundError, match="did you mean 'fake:fake_ds'"):
            resolve_brainset("fake_ds")

    def test_unknown_brainset_in_plugin(self, fake_plugin):
        with pytest.raises(BrainsetNotFoundError, match="not found in the 'fake'"):
            resolve_brainset("fake:nope")

    def test_unknown_namespace(self, no_plugins):
        with pytest.raises(BrainsetNotFoundError, match="No installed plugin"):
            resolve_brainset("nope:x")


class TestListCommand:
    def test_list_groups_by_source(self, fake_plugin):
        result = CliRunner().invoke(cli, ["list"])
        assert result.exit_code == 0, result.output
        assert "torch_brain (built-in):" in result.output
        assert "- pei_pandarinath_nlb_2021" in result.output
        assert "fake (fakeplug" in result.output
        assert "- fake:fake_ds" in result.output

    def test_list_reports_broken_plugin(self, monkeypatch):
        entry_points = [EntryPoint("broken", "does_not_exist.pipelines", GROUP)]
        monkeypatch.setattr(plugins_mod, "_iter_entry_points", lambda: entry_points)
        result = CliRunner().invoke(cli, ["list"])
        assert result.exit_code == 0, result.output
        assert "Plugins that could not be loaded:" in result.output
        assert "- broken (does_not_exist)" in result.output


class TestPreparePlugin:
    @pytest.fixture
    def dirs(self, tmp_path):
        raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
        return raw_dir, processed_dir

    def _invoke(self, args, dirs):
        raw_dir, processed_dir = dirs
        with patch(
            "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
        ) as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = CliRunner().invoke(
                cli,
                [
                    "prepare",
                    *args,
                    f"--raw-dir={raw_dir}",
                    f"--processed-dir={processed_dir}",
                ],
            )
        command = mock_subprocess.call_args[0][0] if mock_subprocess.called else None
        return result, command

    def test_prepare_plugin_brainset(self, fake_plugin, dirs):
        result, command = self._invoke(["fake:fake_ds"], dirs)
        assert result.exit_code == 0, result.output
        assert "Preparing fake:fake_ds (community plugin fakeplug" in result.output
        pipeline_file = command[command.index("torch_brain.pipeline.runner") + 1]
        assert pipeline_file.endswith("fakeplug/pipelines/fake_ds/pipeline.py")

    def test_prepare_unknown_namespace(self, no_plugins, dirs):
        result, command = self._invoke(["nope:x"], dirs)
        assert result.exit_code != 0
        assert "No installed plugin provides the namespace 'nope'" in result.output
        assert command is None
