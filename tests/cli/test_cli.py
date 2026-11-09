"""Tests for CLI commands in torch_brain.pipeline._cli module."""

import importlib
import json
from importlib.metadata import PackagePath
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from torch_brain.pipeline._cli.cli import cli
from torch_brain.pipeline._cli.cli_completion import (
    SHELL_COMPLETION_FILENAMES,
    _detect_shell,
)
from torch_brain.pipeline.config import CONFIG_FILE


@pytest.fixture(autouse=True)
def isolated_cli_state(tmp_path, monkeypatch):
    """Isolates tests in this file from the real brainsets state dir."""
    mod = importlib.import_module("torch_brain.pipeline._cli.cli")

    monkeypatch.setattr(mod, "STATE", tmp_path)
    monkeypatch.setattr(mod, "COMPLETION_ATTEMPTED", tmp_path / ".completion_attempted")
    monkeypatch.setattr(mod, "HINT_SHOWN", tmp_path / ".completion_hint_shown")
    monkeypatch.setattr(mod, "attempted", False)
    monkeypatch.setattr(mod.cli, "epilog", mod.TIP)

    return mod


class TestDetectShell:
    """Tests for _detect_shell."""

    @pytest.mark.parametrize(
        "shell_env, expected",
        [
            ("/bin/bash", "bash"),
            ("/usr/bin/zsh", "zsh"),
            ("/bin/sh", "sh"),
        ],
    )
    def test_detects_shell_from_env(self, shell_env, expected):
        with patch.dict("os.environ", {"SHELL": shell_env}):
            assert _detect_shell() == expected


class TestInstallCompletion:
    """Tests for the --install-completion flag via the CLI."""

    def test_installs_bash_completion(self, tmp_path):
        runner = CliRunner()
        comp_dir = tmp_path / "bash-completions"

        with (
            patch(
                "torch_brain.pipeline._cli.cli_completion._detect_shell",
                return_value="bash",
            ),
            patch.dict(
                "torch_brain.pipeline._cli.cli_completion.SHELL_COMPLETION_DIRS",
                {"bash": comp_dir},
            ),
        ):
            result = runner.invoke(cli, ["--install-completion"])

        assert result.exit_code == 0
        assert "Completion installed for bash" in result.output
        assert "Restart your shell" in result.output

        written = (comp_dir / SHELL_COMPLETION_FILENAMES["bash"]).read_text()
        assert "_BRAINSETS_COMPLETE" in written

    def test_installs_zsh_completion(self, tmp_path):
        runner = CliRunner()
        comp_dir = tmp_path / "zfunc"

        with (
            patch(
                "torch_brain.pipeline._cli.cli_completion._detect_shell",
                return_value="zsh",
            ),
            patch.dict(
                "torch_brain.pipeline._cli.cli_completion.SHELL_COMPLETION_DIRS",
                {"zsh": comp_dir},
            ),
        ):
            result = runner.invoke(cli, ["--install-completion"])

        assert result.exit_code == 0
        assert "Completion installed for zsh" in result.output
        assert "fpath" in result.output

        written = (comp_dir / SHELL_COMPLETION_FILENAMES["zsh"]).read_text()
        assert "_BRAINSETS_COMPLETE" in written

    def test_creates_completion_dir_if_missing(self, tmp_path):
        runner = CliRunner()
        comp_dir = tmp_path / "deep" / "nested" / "dir"

        with (
            patch(
                "torch_brain.pipeline._cli.cli_completion._detect_shell",
                return_value="bash",
            ),
            patch.dict(
                "torch_brain.pipeline._cli.cli_completion.SHELL_COMPLETION_DIRS",
                {"bash": comp_dir},
            ),
        ):
            result = runner.invoke(cli, ["--install-completion"])

        assert result.exit_code == 0
        assert comp_dir.exists()
        assert (comp_dir / SHELL_COMPLETION_FILENAMES["bash"]).is_file()


class TestInstallCompletionTip:
    """Tests for the tab completion tip and first-use hint."""

    @pytest.fixture
    def stubbed_tty_cli_module(self, isolated_cli_state, monkeypatch):
        """Interactive-like session with a stubbed completion installer."""
        mod = isolated_cli_state

        def fake_install(ctx, param, value):
            if value and not ctx.resilient_parsing:
                ctx.exit()

        monkeypatch.setattr(mod, "install_completion", fake_install)
        monkeypatch.setattr(
            mod, "sys", SimpleNamespace(stderr=SimpleNamespace(isatty=lambda: True))
        )
        return mod

    def test_hint_shown_only_once(self, stubbed_tty_cli_module):
        runner = CliRunner()
        config = {"raw_dir": "raw", "processed_dir": "processed"}

        with patch(
            "torch_brain.pipeline._cli.cli_config.load_config", return_value=config
        ):
            first = runner.invoke(cli, ["config", "show"])
            second = runner.invoke(cli, ["config", "show"])

        assert "Tip:" in first.output
        assert stubbed_tty_cli_module.HINT_SHOWN.exists()
        assert "Tip:" not in second.output

    def test_help_tip_disappears_after_completion_attempt(self, stubbed_tty_cli_module):
        runner = CliRunner()

        assert "Tip:" in runner.invoke(cli, ["--help"]).output

        runner.invoke(cli, ["--install-completion"])

        assert stubbed_tty_cli_module.COMPLETION_ATTEMPTED.exists()
        assert "Tip:" not in runner.invoke(cli, ["--help"]).output

    def test_resilient_parsing_does_not_record_attempt(self, stubbed_tty_cli_module):
        cli.make_context("brainsets", ["--install-completion"], resilient_parsing=True)

        assert not stubbed_tty_cli_module.COMPLETION_ATTEMPTED.exists()
        assert stubbed_tty_cli_module.attempted is False
        assert cli.epilog == stubbed_tty_cli_module.TIP


class TestConfigCommand:
    """Tests for the 'brainsets config' command group."""

    def test_config_show(self, tmp_path):
        """Test `brainsets config show` displays current configuration."""
        runner = CliRunner()
        raw_dir = str(tmp_path / "raw")
        processed_dir = str(tmp_path / "processed")
        mock_config = {"raw_dir": raw_dir, "processed_dir": processed_dir}

        with patch(
            "torch_brain.pipeline._cli.cli_config.load_config", return_value=mock_config
        ):
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert f"Config file: {CONFIG_FILE}" in result.output
            assert f"Raw data dir: {raw_dir}" in result.output
            assert f"Processed data dir: {processed_dir}" in result.output

    def test_config_show_no_config(self):
        """Test `brainsets config show` errors when no config exists."""
        runner = CliRunner()

        with patch(
            "torch_brain.pipeline._cli.cli_config.load_config", return_value=None
        ):
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code != 0
            assert "Config not found" in result.output

    def test_config_set_with_options(self, tmp_path):
        """Test `brainsets config set --raw-dir ... --processed-dir ...`."""
        runner = CliRunner()
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"

        with (
            patch(
                "torch_brain.pipeline._cli.cli_config.load_config", return_value=None
            ),
            patch(
                "torch_brain.pipeline._cli.cli_config.save_config",
                return_value=CONFIG_FILE,
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "config",
                    "set",
                    "--raw-dir",
                    str(raw_dir),
                    "--processed-dir",
                    str(processed_dir),
                ],
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert f"Raw data dir: {raw_dir}" in result.output
            assert f"Processed data dir: {processed_dir}" in result.output


class TestPrepareCommand:
    """Tests for the 'brainsets prepare' command."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()
        return {
            "raw_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        }

    @pytest.fixture
    def make_mock_dist(self, tmp_path):
        """Mock torch_brain distribution."""

        def _make(install_type: dict | None):

            if install_type == "pypi":
                return SimpleNamespace(files=[], locate_file=lambda f: f)

            if install_type == "directory":
                src = tmp_path / "torch_brain"
                src.mkdir()
                url = src.as_uri()
            elif install_type == "directory_with_spaces":
                # as_uri() percent-encodes the space (%20), url2pathname decodes it
                src = tmp_path / "my folder" / "torch_brain"
                src.mkdir(parents=True)
                url = src.as_uri()
            elif install_type == "directory_two_slashes":
                # Hand-written Windows form: file://C:/...
                src = tmp_path / "torch_brain"
                src.mkdir()
                url = "file://" + src.as_posix()
            elif install_type == "wheel":
                archive = tmp_path / "torch_brain-0.1.0-py3-none-any.whl"
                archive.touch()
                url = archive.as_uri()
            elif install_type == "tarball":
                archive = tmp_path / "torch_brain-0.1.0.tar.gz"
                archive.touch()
                url = archive.as_uri()

            path = tmp_path / "direct_url.json"
            path.write_text(json.dumps({"url": url}))
            return SimpleNamespace(
                files=[PackagePath("direct_url.json")],
                locate_file=lambda _f: path,
            )

        return _make

    @pytest.mark.parametrize(
        "install_type",
        [
            "pypi",
            "directory",
            "directory_with_spaces",
            "directory_two_slashes",
            "wheel",
            "tarball",
        ],
    )
    def test_prepare_valid_brainset(self, mock_config, make_mock_dist, install_type):
        """Test prepare command with valid brainset constructs correct subprocess call.
        Ensuring it passes through the inline metadata correctly.
        """
        runner = CliRunner()

        dist = make_mock_dist(install_type)

        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch("importlib.metadata.distribution", return_value=dist),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            # Expected command: strings must match exactly, tuple components must be present
            if "directory" in install_type:
                dependency_args = [
                    "--with-editable",
                    ("torch_brain",),
                    "--with",
                    "dandi==0.74.0",
                ]
            else:
                dependency_args = ["--with", ("torch_brain", "dandi==0.74.0")]

            expected_command = [
                "uv",
                "run",
                "--directory",
                ("pei_pandarinath_nlb_2021", "pipelines"),
                "--isolated",
                "--no-project",
                "--python",
                "3.11",
                *dependency_args,
                "python",
                "-m",
                "torch_brain.pipeline.runner",
                ("pipeline.py",),
                f"--raw-dir={mock_config['raw_dir']}",
                f"--processed-dir={mock_config['processed_dir']}",
                "-c4",  # default cores
            ]

            assert len(command) == len(expected_command)

            for actual, expected in zip(command, expected_command, strict=True):
                if isinstance(expected, tuple):
                    assert all(part in actual for part in expected)
                else:
                    assert expected == actual

    def test_cli_raw_processed_dirs_override(self, tmp_path):
        """Test prepare command with raw and processed dirs overridden."""
        runner = CliRunner()

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli,
                [
                    "prepare",
                    "pei_pandarinath_nlb_2021",
                    "--raw-dir",
                    str(raw_dir),
                    "--processed-dir",
                    str(processed_dir),
                ],
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert f"--raw-dir={str(raw_dir)}" in command
            assert f"--processed-dir={str(processed_dir)}" in command

    def test_download_only_flag(self, mock_config):
        """Test prepare command with --download-only flag
        Ensure it is forwarded to the runner subprocess."""
        runner = CliRunner()

        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "--download-only"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--download-only" in command

    def test_list_flag(self, mock_config):
        """Test prepare command with --list flag
        Ensure it is forwarded to the runner subprocess."""
        runner = CliRunner()

        # --list passed if present in cli args
        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "--list"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--list" in command

        # --list not passed if absent in cli args
        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--list" not in command

    def test_single_option(self, mock_config):
        """Test passthrough of the --single/-s option to the runner subprocess."""
        runner = CliRunner()

        # --single=test passed if "--single test" present in cli args
        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "--single", "test-index"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--single=test-index" in command
            for part in command:
                assert part != "-s"
                assert not part.startswith("-s=")

        # --single=test passed if "-s test" test present in cli args
        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "-s", "test-index"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--single=test-index" in command
            for part in command:
                assert part != "-s"
                assert not part.startswith("-s=")

        # --single not passed if absent in cli args
        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"

            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            for part in command:
                assert part != "--single"
                assert part != "-s"
                assert not part.startswith("--single=")
                assert not part.startswith("-s=")

    def test_extra_option_passthrough(self, mock_config):
        """Test that extra options are passed through to the subprocess."""
        runner = CliRunner()

        with (
            patch(
                "torch_brain.pipeline._cli.cli_prepare.load_config",
                return_value=mock_config,
            ),
            patch(
                "torch_brain.pipeline._cli.cli_prepare.subprocess.run"
            ) as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(
                cli, ["prepare", "pei_pandarinath_nlb_2021", "--unknown", "option"]
            )
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert "--unknown" in command[-2]
            assert "option" in command[-1]
