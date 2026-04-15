"""Tests for CLI commands in brainsets._cli module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from brainsets._cli.cli import cli
from brainsets._cli.cli_completion import (
    _detect_shell,
    SHELL_COMPLETION_FILENAMES,
)
from brainsets.config import CONFIG_FILE


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
            patch("brainsets._cli.cli_completion._detect_shell", return_value="bash"),
            patch.dict(
                "brainsets._cli.cli_completion.SHELL_COMPLETION_DIRS",
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
            patch("brainsets._cli.cli_completion._detect_shell", return_value="zsh"),
            patch.dict(
                "brainsets._cli.cli_completion.SHELL_COMPLETION_DIRS",
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
            patch("brainsets._cli.cli_completion._detect_shell", return_value="bash"),
            patch.dict(
                "brainsets._cli.cli_completion.SHELL_COMPLETION_DIRS",
                {"bash": comp_dir},
            ),
        ):
            result = runner.invoke(cli, ["--install-completion"])

        assert result.exit_code == 0
        assert comp_dir.exists()
        assert (comp_dir / SHELL_COMPLETION_FILENAMES["bash"]).is_file()


class TestConfigCommand:
    """Tests for the 'brainsets config' command group."""

    def test_config_show(self, tmp_path):
        """Test `brainsets config show` displays current configuration."""
        runner = CliRunner()
        raw_dir = str(tmp_path / "raw")
        processed_dir = str(tmp_path / "processed")
        mock_config = {"raw_dir": raw_dir, "processed_dir": processed_dir}

        with patch("brainsets._cli.cli_config.load_config", return_value=mock_config):
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert f"Config file: {CONFIG_FILE}" in result.output
            assert f"Raw data dir: {raw_dir}" in result.output
            assert f"Processed data dir: {processed_dir}" in result.output

    def test_config_show_no_config(self):
        """Test `brainsets config show` errors when no config exists."""
        runner = CliRunner()

        with patch("brainsets._cli.cli_config.load_config", return_value=None):
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code != 0
            assert "Config not found" in result.output

    def test_config_set_with_options(self, tmp_path):
        """Test `brainsets config set --raw-dir ... --processed-dir ...`."""
        runner = CliRunner()
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"

        with (
            patch("brainsets._cli.cli_config.load_config", return_value=None),
            patch("brainsets._cli.cli_config.save_config", return_value=CONFIG_FILE),
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

    def test_prepare_valid_brainset(self, mock_config):
        """Test prepare command with valid brainset constructs correct subprocess call.
        Ensuring it passes through the inline metadata correctly.
        """
        runner = CliRunner()

        with (
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
        ):
            mock_subprocess.return_value = MagicMock(returncode=0)
            result = runner.invoke(cli, ["prepare", "pei_pandarinath_nlb_2021"])
            assert result.exit_code == 0, f"CLI failed with: {result.output}"
            assert "Preparing pei_pandarinath_nlb_2021" in result.output

            # Verify subprocess was called with correct arguments
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args
            command = call_args[1].get("command") or call_args[0][0]

            assert command[0] == "uv"
            assert command[1] == "run"
            assert command[2] == "--directory"
            assert "pei_pandarinath_nlb_2021" in command[3]
            assert "brainsets_pipelines" in command[3]
            assert command[4] == "--isolated"
            assert command[5] == "--no-project"
            assert command[6] == "--python"
            assert command[7] == "3.11"
            assert command[8] == "--with-editable"
            assert ("brainsets" in command[9]) and ("file://" in command[9])
            assert command[10] == "--with"
            assert command[11] == "dandi==0.74.0"
            assert command[12] == "python"
            assert command[13] == "-m"
            assert command[14] == "brainsets.runner"
            assert "pipeline.py" in command[15]
            assert f"--raw-dir={mock_config['raw_dir']}" in command
            assert f"--processed-dir={mock_config['processed_dir']}" in command
            assert "-c4" in command  # default cores

    def test_cli_raw_processed_dirs_override(self, tmp_path):
        """Test prepare command with raw and processed dirs overridden."""
        runner = CliRunner()

        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        raw_dir.mkdir()
        processed_dir.mkdir()

        with (patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,):
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
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
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

    def test_extra_option_passthrough(self, mock_config):
        """Test that extra options are passed through to the subprocess."""
        runner = CliRunner()

        with (
            patch("brainsets._cli.cli_prepare.load_config", return_value=mock_config),
            patch("brainsets._cli.cli_prepare.subprocess.run") as mock_subprocess,
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
