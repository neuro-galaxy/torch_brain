import sys
from pathlib import Path

import click

from .cli_completion import install_completion
from .cli_config import config
from .cli_list import cli_list
from .cli_prepare import prepare

# For handling tab completion tip
STATE = Path(click.get_app_dir("brainsets"))
COMPLETION_ATTEMPTED = STATE / ".completion_attempted"
HINT_SHOWN = STATE / ".completion_hint_shown"
TIP = "Tip: enable tab completion with `brainsets --install-completion`"

attempted = COMPLETION_ATTEMPTED.exists()


def _touch(path: Path) -> None:
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError:
        pass  # don't break CLI for tab completion tip


def _install_completion(ctx, param, value):
    if value:
        _touch(COMPLETION_ATTEMPTED)
    return install_completion(ctx, param, value)


@click.group(epilog=None if attempted else TIP)
@click.option(
    "--install-completion",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_install_completion,
    help="Install shell tab completion and exit.",
)
def cli():
    """
    Brainsets CLI\n
    A command line interface for downloading and preparing brainsets.

    \b
    Documentation: https://torch-brain.readthedocs.io
    Project page: https://github.com/neuro-galaxy/torch_brain
    """
    # Tab completion tip
    if attempted or not sys.stderr.isatty() or HINT_SHOWN.exists():
        return
    click.secho(TIP, fg="yellow", err=True)
    _touch(HINT_SHOWN)


cli.add_command(prepare)
cli.add_command(cli_list, name="list")
cli.add_command(config)


if __name__ == "__main__":
    cli()
