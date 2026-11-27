import importlib.util
import inspect
import warnings
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path

CONF_FILENAME = "conf.py"


@dataclass
class PipelineConf:
    """Pipeline configuration, read from the ``conf.py`` next to ``pipeline.py``.

    ``conf.py`` is executed by the ``brainsets`` CLI in the user's environment,
    *before* the pipeline's isolated environment exists. So it must only import
    the standard library (and **torch_brain**), never the pipeline's dependencies.
    """

    description: str | None = None
    """Module docstring of ``conf.py``."""
    python_version: str | None = None
    """Python version for the isolated environment, e.g. ``"3.11"``."""
    dependencies: list[str] = field(default_factory=list)
    """pip-installable requirement specifiers."""
    parser: ArgumentParser | None = None
    """Parser for pipeline-specific command-line arguments."""


_KNOWN_NAMES = {"python_version", "dependencies", "parser"}


def load_conf(pipeline_dir: Path) -> PipelineConf | None:
    """Load ``conf.py`` from ``pipeline_dir``. Returns ``None`` if there is none."""
    path = Path(pipeline_dir) / CONF_FILENAME
    if not path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("brainset_pipeline_conf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, value in vars(module).items():
        if name.startswith("_") or name in _KNOWN_NAMES:
            continue
        if inspect.ismodule(value) or inspect.isclass(value) or callable(value):
            continue  # imports and helpers
        warnings.warn(f"Ignoring unknown name '{name}' in {path}", stacklevel=2)

    return PipelineConf(
        description=inspect.getdoc(module),
        python_version=getattr(module, "python_version", None),
        dependencies=list(getattr(module, "dependencies", [])),
        parser=getattr(module, "parser", None),
    )
