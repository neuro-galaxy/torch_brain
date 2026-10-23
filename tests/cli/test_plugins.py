"""Tests for brainsets plugins: discovery, the registry, and the list/info/check/prepare
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
from torch_brain.pipeline._cli.cli_prepare import _dist_in_dependencies
from torch_brain.pipeline._cli.plugins import (
    OWNER_MARKER_FILENAME,
    BrainsetNotFoundError,
    check_owner_marker,
    get_builtin_source,
    get_plugin_sources,
    read_owner_marker,
    resolve_brainset,
    split_name,
    write_owner_marker,
)
from torch_brain.pipeline._cli.registry import (
    RegistryError,
    load_registry,
    parse_registry,
    validate_registry,
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

REGISTRY_TOML = """\
[fake]
package = "fakeplug"
description = "A fake plugin for tests."
install = "git+https://github.com/example/fakeplug@v0.1.0"
repository = "https://github.com/example/fakeplug"
license = "MIT"
maintainers = ["@example"]

[[fake.brainsets]]
id = "fake_ds"
description = "A fake dataset."

[remote]
package = "remote_brainsets"
description = "A registered plugin that is not installed."
install = "remote_brainsets==1.0.0"
repository = "https://github.com/example/remote_brainsets"
license = "Apache-2.0"
maintainers = ["@example"]
modalities = ["eeg"]

[[remote.brainsets]]
id = "remote_ds"
description = "A remote dataset."
source = "https://example.org/remote_ds"
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
def registry(tmp_path, monkeypatch):
    path = tmp_path / "plugins.toml"
    path.write_text(REGISTRY_TOML)
    monkeypatch.setattr("torch_brain.pipeline._cli.registry.REGISTRY_PATH", path)
    return path


@pytest.fixture
def empty_registry(tmp_path, monkeypatch):
    path = tmp_path / "empty_plugins.toml"
    path.write_text("")
    monkeypatch.setattr("torch_brain.pipeline._cli.registry.REGISTRY_PATH", path)
    return path


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
            (
                "torch_brain:pei_pandarinath_nlb_2021",
                ("torch_brain", "pei_pandarinath_nlb_2021"),
            ),
            ("openneuro:klinzing", ("openneuro", "klinzing")),
        ],
    )
    def test_split_name(self, name, expected):
        assert split_name(name) == expected

    def test_builtin_alias(self, no_plugins):
        bare = resolve_brainset("pei_pandarinath_nlb_2021")
        alias = resolve_brainset("torch_brain:pei_pandarinath_nlb_2021")
        assert bare.source.is_builtin and alias.source.is_builtin
        assert bare.pipeline_dir == alias.pipeline_dir

    def test_plugin_brainset(self, fake_plugin, empty_registry):
        resolved = resolve_brainset("fake:fake_ds")
        assert resolved.source.namespace == "fake"
        assert resolved.brainset_id == "fake_ds"
        assert resolved.qualified_name == "fake:fake_ds"

    def test_bare_plugin_name_suggests_qualified(self, fake_plugin, empty_registry):
        with pytest.raises(BrainsetNotFoundError, match="did you mean 'fake:fake_ds'"):
            resolve_brainset("fake_ds")

    def test_unknown_brainset_in_plugin(self, fake_plugin, empty_registry):
        with pytest.raises(BrainsetNotFoundError, match="not found in the 'fake'"):
            resolve_brainset("fake:nope")

    def test_registered_but_not_installed_prints_install_command(
        self, no_plugins, registry
    ):
        with pytest.raises(BrainsetNotFoundError) as e:
            resolve_brainset("remote:remote_ds")
        assert "not installed" in str(e.value)
        assert 'uv add "remote_brainsets==1.0.0"' in str(e.value)

    def test_unknown_namespace(self, no_plugins, empty_registry):
        with pytest.raises(BrainsetNotFoundError, match="No installed plugin"):
            resolve_brainset("nope:x")


class TestOwnerMarker:
    def test_unmarked_folder_is_compatible(self, tmp_path, fake_plugin):
        source = get_plugin_sources()["fake"]
        assert check_owner_marker(tmp_path / "ds", source) is None

    def test_write_then_read(self, tmp_path, fake_plugin):
        source = get_plugin_sources()["fake"]
        write_owner_marker(tmp_path / "ds", source)
        marker = read_owner_marker(tmp_path / "ds")
        assert marker["source"] == "fakeplug"
        assert marker["namespace"] == "fake"
        assert check_owner_marker(tmp_path / "ds", source) is None

    def test_existing_marker_not_overwritten(self, tmp_path, fake_plugin):
        write_owner_marker(tmp_path / "ds", get_builtin_source())
        write_owner_marker(tmp_path / "ds", get_plugin_sources()["fake"])
        assert read_owner_marker(tmp_path / "ds")["source"] == "torch_brain"

    def test_torch_brain_folder_is_compatible_with_plugin(self, tmp_path, fake_plugin):
        write_owner_marker(tmp_path / "ds", get_builtin_source())
        source = get_plugin_sources()["fake"]
        assert check_owner_marker(tmp_path / "ds", source) is None

    def test_other_plugin_folder_is_refused(self, tmp_path, fake_plugin):
        (tmp_path / "ds").mkdir()
        (tmp_path / "ds" / OWNER_MARKER_FILENAME).write_text(
            'source = "someone_else"\nversion = "2.0"\n'
        )
        source = get_plugin_sources()["fake"]
        msg = check_owner_marker(tmp_path / "ds", source)
        assert "'someone_else' 2.0" in msg
        assert check_owner_marker(tmp_path / "ds", get_builtin_source()) is not None


class TestRegistry:
    def test_bundled_registry_is_valid(self):
        assert isinstance(load_registry(), dict)

    def test_parse(self, registry):
        entries = load_registry()
        assert set(entries) == {"fake", "remote"}
        assert entries["remote"].modalities == ["eeg"]
        assert entries["remote"].brainsets[0].source == "https://example.org/remote_ds"

    @pytest.mark.parametrize(
        "data, error",
        [
            ({"Bad": {}}, "namespace must only contain"),
            ({"torch_brain": {}}, "is reserved"),
            ({"x": {"description": "d"}}, "missing required key 'install'"),
            ({"x": {"unknown": 1}}, "unknown key 'unknown'"),
            ({"x": {"repository": "http://insecure"}}, "must be an https:// URL"),
            ({"x": {"maintainers": []}}, "'maintainers' must not be empty"),
            ({"x": {"brainsets": []}}, "'brainsets' must not be empty"),
            (
                {"x": {"brainsets": [{"id": "a", "description": "d"}] * 2}},
                "duplicate id 'a'",
            ),
            (
                {"x": {"brainsets": [{"id": "1a", "description": "d"}]}},
                "valid identifier",
            ),
        ],
    )
    def test_validation_errors(self, data, error):
        errors = validate_registry(data)
        assert any(error in e for e in errors), errors

    def test_parse_raises_on_invalid(self):
        with pytest.raises(RegistryError):
            parse_registry({"x": {}})


class TestListCommand:
    def test_list_groups_by_source(self, fake_plugin, registry):
        result = CliRunner().invoke(cli, ["list"])
        assert result.exit_code == 0, result.output
        assert "torch_brain (built-in):" in result.output
        assert "- pei_pandarinath_nlb_2021" in result.output
        assert "fake (fakeplug" in result.output
        assert "- fake:fake_ds" in result.output
        assert "remote" not in result.output

    def test_list_all_shows_registered_plugins(self, fake_plugin, registry):
        result = CliRunner().invoke(cli, ["list", "--all"])
        assert result.exit_code == 0, result.output
        assert "Not installed" in result.output
        assert "- remote:remote_ds" in result.output


class TestInfoCommand:
    def test_info_registered_not_installed(self, no_plugins, registry):
        result = CliRunner().invoke(cli, ["info", "remote"])
        assert result.exit_code == 0, result.output
        assert "Installed:" in result.output and "no" in result.output
        assert 'pip install "remote_brainsets==1.0.0"' in result.output
        assert "- remote:remote_ds: A remote dataset." in result.output

    def test_info_installed_plugin(self, fake_plugin, registry):
        result = CliRunner().invoke(cli, ["info", "fake"])
        assert result.exit_code == 0, result.output
        assert "yes (fakeplug" in result.output
        assert "- fake:fake_ds: A fake dataset." in result.output

    def test_info_plugin_brainset(self, fake_plugin, registry):
        result = CliRunner().invoke(cli, ["info", "fake:fake_ds"])
        assert result.exit_code == 0, result.output
        assert "A fake dataset." in result.output
        assert "community plugin" in result.output
        assert "numpy" in result.output

    def test_info_remote_brainset(self, no_plugins, registry):
        result = CliRunner().invoke(cli, ["info", "remote:remote_ds"])
        assert result.exit_code == 0, result.output
        assert "https://example.org/remote_ds" in result.output
        assert 'uv add "remote_brainsets==1.0.0"' in result.output

    def test_info_builtin(self, no_plugins, empty_registry):
        result = CliRunner().invoke(cli, ["info", "pei_pandarinath_nlb_2021"])
        assert result.exit_code == 0, result.output
        assert "built-in" in result.output

    def test_info_unknown(self, no_plugins, empty_registry):
        result = CliRunner().invoke(cli, ["info", "nope"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestCheckCommand:
    def test_check_valid_plugin(self, fake_plugin, registry):
        result = CliRunner().invoke(cli, ["check", "fake"])
        assert result.exit_code == 0, result.output
        assert "OK       fake:fake_ds" in result.output

    def test_check_unregistered_plugin_warns(self, fake_plugin, empty_registry):
        result = CliRunner().invoke(cli, ["check", "fake"])
        assert result.exit_code == 0, result.output
        assert "not in the plugin registry" in result.output
        strict = CliRunner().invoke(cli, ["check", "fake", "--strict"])
        assert strict.exit_code == 1

    @pytest.mark.parametrize(
        "source, error",
        [
            (
                PIPELINE_TEMPLATE.format(brainset_id="other_name"),
                "must equal the folder name",
            ),
            ("x = 1\n", "does not define 'Pipeline'"),
            ("class Pipeline:\n    brainset_id = 'fake_ds'\n", "must subclass"),
            (
                "from torch_brain.pipeline import BrainsetPipeline\n"
                "class Pipeline(BrainsetPipeline):\n    brainset_id = 'fake_ds'\n",
                "does not implement download, get_manifest, process",
            ),
            (
                "# /// brainset-pipeline\n# python-version = '>=3.10'\n# ///\n"
                + PIPELINE_TEMPLATE.format(brainset_id="fake_ds"),
                "invalid brainset-pipeline metadata",
            ),
            ("raise RuntimeError('boom')\n", "importing pipeline.py failed"),
        ],
    )
    def test_check_reports_errors(self, fake_plugin, empty_registry, source, error):
        pipeline_file = (
            Path(str(get_plugin_sources()["fake"].pipelines_path))
            / "fake_ds"
            / "pipeline.py"
        )
        pipeline_file.write_text(source)
        result = CliRunner().invoke(cli, ["check", "fake"])
        assert result.exit_code == 1, result.output
        assert error in result.output

    def test_check_missing_dependency_falls_back_to_static(
        self, fake_plugin, empty_registry
    ):
        pipeline_file = (
            Path(str(get_plugin_sources()["fake"].pipelines_path))
            / "fake_ds"
            / "pipeline.py"
        )
        pipeline_file.write_text(
            "import not_installed_dependency_xyz\n"
            + PIPELINE_TEMPLATE.format(brainset_id="fake_ds")
        )
        result = CliRunner().invoke(cli, ["check", "fake"])
        assert result.exit_code == 0, result.output
        assert "checked it statically" in result.output

    def test_check_registry_mismatch(self, fake_plugin, tmp_path, monkeypatch):
        path = tmp_path / "plugins.toml"
        path.write_text(
            REGISTRY_TOML.replace('id = "fake_ds"', 'id = "listed_only"').replace(
                'package = "fakeplug"', 'package = "someone_else"'
            )
        )
        monkeypatch.setattr("torch_brain.pipeline._cli.registry.REGISTRY_PATH", path)
        result = CliRunner().invoke(cli, ["check", "fake"])
        assert result.exit_code == 1
        assert "'listed_only' is listed but not shipped" in result.output
        assert "'fake_ds' is shipped but not listed" in result.output
        assert "registered to 'someone_else'" in result.output

    def test_check_builtin(self, no_plugins):
        result = CliRunner().invoke(cli, ["check", "torch_brain"])
        assert result.exit_code == 0, result.output

    def test_check_unknown_namespace(self, no_plugins):
        result = CliRunner().invoke(cli, ["check", "nope"])
        assert result.exit_code != 0
        assert "No installed plugin" in result.output

    def test_check_registry_file(self, registry, tmp_path):
        result = CliRunner().invoke(cli, ["check", "--registry", str(registry)])
        assert result.exit_code == 0, result.output
        bad = tmp_path / "bad.toml"
        bad.write_text('[Bad]\ndescription = "x"\n')
        result = CliRunner().invoke(cli, ["check", "--registry", str(bad)])
        assert result.exit_code == 1
        assert "namespace must only contain" in result.output

    def test_check_bundled_registry(self):
        result = CliRunner().invoke(cli, ["check", "--registry"])
        assert result.exit_code == 0, result.output


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

    def test_prepare_plugin_injects_plugin_and_writes_marker(
        self, fake_plugin, empty_registry, dirs
    ):
        result, command = self._invoke(["fake:fake_ds"], dirs)
        assert result.exit_code == 0, result.output
        assert "Preparing fake:fake_ds (community plugin fakeplug" in result.output

        pipeline_file = command[command.index("torch_brain.pipeline.runner") + 1]
        assert pipeline_file.endswith("fakeplug/pipelines/fake_ds/pipeline.py")
        with_deps = command[command.index("--with") + 1]
        assert "numpy" in with_deps
        # The fake plugin has no distribution metadata, so a bare name is injected
        assert "fakeplug" in with_deps

        marker = read_owner_marker(dirs[1] / "fake_ds")
        assert marker["source"] == "fakeplug"

    def test_prepare_plugin_refuses_foreign_folder(
        self, fake_plugin, empty_registry, dirs
    ):
        dataset_dir = dirs[1] / "fake_ds"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / OWNER_MARKER_FILENAME).write_text('source = "someone_else"\n')
        result, command = self._invoke(["fake:fake_ds"], dirs)
        assert result.exit_code != 0
        assert "was prepared by 'someone_else'" in result.output
        assert command is None

    def test_prepare_list_does_not_write_marker(
        self, fake_plugin, empty_registry, dirs
    ):
        result, _ = self._invoke(["fake:fake_ds", "--list"], dirs)
        assert result.exit_code == 0, result.output
        assert read_owner_marker(dirs[1] / "fake_ds") is None

    def test_prepare_builtin_alias(self, no_plugins, dirs):
        result, command = self._invoke(["torch_brain:pei_pandarinath_nlb_2021"], dirs)
        assert result.exit_code == 0, result.output
        assert "Preparing pei_pandarinath_nlb_2021" in result.output
        assert read_owner_marker(dirs[1] / "pei_pandarinath_nlb_2021")["source"] == (
            "torch_brain"
        )

    def test_prepare_uninstalled_registered_plugin(self, no_plugins, registry, dirs):
        result, command = self._invoke(["remote:remote_ds"], dirs)
        assert result.exit_code != 0
        assert 'uv add "remote_brainsets==1.0.0"' in result.output
        assert command is None


@pytest.mark.parametrize(
    "deps, name, expected",
    [
        (["torch_brain==0.2.1"], "torch_brain", True),
        (["torch-brain>=0.2"], "torch_brain", True),
        (["git+https://github.com/x/torch_brain@main"], "torch_brain", True),
        (["torch_brain_extra"], "torch_brain", False),
        (["openneuro-brainsets @ git+https://x"], "openneuro_brainsets", True),
        (["mne==1.11"], "openneuro_brainsets", False),
    ],
)
def test_dist_in_dependencies(deps, name, expected):
    assert _dist_in_dependencies(deps, name) is expected
