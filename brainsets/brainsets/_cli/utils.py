import os
from typing import Callable, List, Optional, Union
import click
import yaml
from pathlib import Path
import brainsets_pipelines
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion

CONFIG_FILE = Path.home() / ".brainsets.yaml"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])


def expand_path(path: Union[str, Path]) -> Path:
    """
    Convert string path to absolute Path, expanding environment variables and user.
    """
    return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(path))))


def load_config(path: Path = CONFIG_FILE, raise_cli_error: bool = True):
    if path.exists():
        with open(path, "r") as f:
            ret = yaml.safe_load(f)

        if raise_cli_error:
            _validate_config(ret)
        else:
            try:
                _validate_config(ret)
            except:
                return None

        return ret
    elif raise_cli_error:
        raise click.ClickException(
            f"Config not found at {path}. Please run `brainsets config`"
        )
    else:
        return None


def _validate_config(config: dict):
    if "raw_dir" not in config:
        raise click.ClickException(
            "'raw_dir' missing in config. Please run `brainsets config`."
        )
    if "processed_dir" not in config:
        raise click.ClickException(
            "'processed_dir' missing in config. Please run `brainsets config`."
        )


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return CONFIG_FILE


def get_available_brainsets():
    ret = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]
    ret = [name for name in ret if not name.startswith((".", "_"))]
    return ret


def debug_echo(msg: str, enable: bool):
    if enable:
        click.echo(f"DEBUG: {msg}")
