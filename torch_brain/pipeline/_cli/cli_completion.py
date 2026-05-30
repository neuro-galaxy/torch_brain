import os
from pathlib import Path

import click
from click.shell_completion import get_completion_class


SUPPORTED_SHELLS = {"bash", "zsh"}

SHELL_COMPLETION_DIRS = {
    "bash": Path.home() / ".local" / "share" / "bash-completion" / "completions",
    "zsh": Path.home() / ".zfunc",
}

SHELL_COMPLETION_FILENAMES = {
    "bash": "brainsets",
    "zsh": "_brainsets",
}


def _detect_shell():
    shell_path = os.environ.get("SHELL", "")
    return Path(shell_path).name or None


def _get_completion_script(cli_group, shell):
    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(
            f"Shell '{shell}' is not supported. Supported: bash, zsh."
        )
    comp = comp_cls(cli_group, {}, "brainsets", "_BRAINSETS_COMPLETE")
    return comp.source()


def install_completion(ctx, param, value):
    """Add shell completion for brainsets CLI."""
    if not value or ctx.resilient_parsing:
        return

    shell = _detect_shell()
    if shell not in SUPPORTED_SHELLS:
        raise click.ClickException(
            f"Could not detect a supported shell (got '{shell}'). "
            "Supported shells: {SUPPORTED_SHELLS}."
        )

    script = _get_completion_script(ctx.command, shell)

    completion_dir = SHELL_COMPLETION_DIRS[shell]
    completion_dir.mkdir(parents=True, exist_ok=True)
    completion_file = completion_dir / SHELL_COMPLETION_FILENAMES[shell]
    completion_file.write_text(script)

    click.echo(f"Completion installed for {shell} at {completion_file}")
    if shell == "zsh":
        click.echo("Make sure ~/.zfunc is in your fpath.")
    click.echo("Restart your shell to activate.")
    ctx.exit()
