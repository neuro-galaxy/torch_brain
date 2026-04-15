from typing import Optional
import click
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.completion import PathCompleter

from brainsets.config import CONFIG_FILE, load_config, save_config

from .utils import expand_path


@click.group(invoke_without_command=True)
@click.option(
    "--raw-dir",
    help="[Deprecated] Path for storing raw data. Use `brainsets config set` instead.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--processed-dir",
    help="[Deprecated] Path for storing processed brainsets. Use `brainsets config set` instead.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.pass_context
def config(ctx, raw_dir, processed_dir):
    """Manage brainsets configuration."""
    if ctx.invoked_subcommand is None:
        # Deprecated: forward to `set` subcommand for backward compatibility
        click.echo(
            "Warning: `brainsets config [--raw-dir ... --processed-dir ...]` is "
            "deprecated. Use `brainsets config set` instead.",
            err=True,
        )
        ctx.invoke(set_config, raw_dir=raw_dir, processed_dir=processed_dir)


@config.command(name="set")
@click.option(
    "--raw-dir",
    help="Path for storing raw data.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--processed-dir",
    help="Path for storing processed brainsets.",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
def set_config(raw_dir: Optional[Path], processed_dir: Optional[Path]):
    """Set raw and processed data directories."""

    # Get missing args from user prompts
    if raw_dir is None or processed_dir is None:
        raw_dir = prompt(
            "Enter raw data directory: ",
            completer=PathCompleter(only_directories=True),
            complete_style=CompleteStyle.READLINE_LIKE,
        )
        processed_dir = prompt(
            "Enter processed data directory: ",
            completer=PathCompleter(only_directories=True),
            complete_style=CompleteStyle.READLINE_LIKE,
        )

    raw_dir = expand_path(raw_dir)
    processed_dir = expand_path(processed_dir)

    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = load_config()
    config_exists = cfg is not None
    if not config_exists:
        cfg = {}
    cfg["raw_dir"] = str(raw_dir)
    cfg["processed_dir"] = str(processed_dir)
    config_filepath = save_config(cfg)

    if not config_exists:
        click.echo(f"Created config file at {config_filepath}")
    else:
        click.echo(f"Updated config file at {config_filepath}")

    click.echo(f"Raw data dir: {cfg['raw_dir']}")
    click.echo(f"Processed data dir: {cfg['processed_dir']}")


@config.command()
def show():
    """Display current configuration."""
    cfg = load_config()
    if cfg is None:
        raise click.ClickException(
            f"Config not found at {CONFIG_FILE}. Please run `brainsets config set`."
        )

    click.echo(f"Config file: {CONFIG_FILE}")
    click.echo(f"Raw data dir: {cfg['raw_dir']}")
    click.echo(f"Processed data dir: {cfg['processed_dir']}")
