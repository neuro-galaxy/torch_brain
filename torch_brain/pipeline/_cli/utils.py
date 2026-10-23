import os
from pathlib import Path

import click

from .plugins import BUILTIN_PIPELINES_PATH, get_all_sources

PIPELINES_PATH = BUILTIN_PIPELINES_PATH


def expand_path(path: str | Path) -> Path:
    """
    Convert string path to absolute Path, expanding environment variables and user.
    """
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


def get_available_brainsets() -> list[str]:
    """Names of every installed brainset, as typed on the command line.

    Built-in brainsets have bare names, plugin brainsets are ``<namespace>:<id>``.
    """
    ret = []
    for source in get_all_sources(warn=False):
        ret += [source.qualified_name(b) for b in source.brainsets()]
    return ret


def debug_echo(msg: str, enable: bool):
    if enable:
        click.echo(f"DEBUG: {msg}")
