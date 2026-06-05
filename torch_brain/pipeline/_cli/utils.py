import os
from importlib.resources import files
from pathlib import Path

import click

PIPELINES_PATH = files("torch_brain.pipeline") / "brainsets-pipelines"


def expand_path(path: str | Path) -> Path:
    """
    Convert string path to absolute Path, expanding environment variables and user.
    """
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


def get_available_brainsets():
    ret = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]
    ret = [name for name in ret if not name.startswith((".", "_"))]
    return ret


def debug_echo(msg: str, enable: bool):
    if enable:
        click.echo(f"DEBUG: {msg}")
