"""Discovery of brainset pipelines: built-in ones and the ones shipped by plugins.

A plugin is an installed distribution that declares an entry point in the
``brainsets.pipelines`` group. The entry point name is the plugin's namespace and its
value is an importable package holding one ``<brainset_id>/pipeline.py`` folder per
brainset. Plugin brainsets are always addressed as ``<namespace>:<brainset_id>``.
"""

import ast
import re
import warnings
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from importlib.resources import files
from pathlib import Path

try:  # Python 3.11+
    from importlib.resources.abc import Traversable

    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python <3.11
    from importlib.abc import Traversable

    import tomli as tomllib  # type: ignore[import-not-found]

ENTRY_POINT_GROUP = "brainsets.pipelines"
BUILTIN_NAMESPACE = "torch_brain"
BUILTIN_PIPELINES_PATH = files("torch_brain.pipeline") / "brainsets-pipelines"
NAMESPACE_PATTERN = re.compile(r"^[a-z0-9_]+$")

OWNER_MARKER_FILENAME = ".brainsets_source"


@dataclass
class PipelineSource:
    """A folder of ``<brainset_id>/pipeline.py`` pipelines, built-in or from a plugin."""

    namespace: str
    """``torch_brain`` for built-ins, otherwise the plugin's entry point name."""
    dist_name: str
    """Name of the distribution that ships the pipelines."""
    dist_version: str | None
    pipelines_path: Traversable | None
    """Package holding the pipeline folders; ``None`` if the plugin failed to load."""
    entry_point_value: str | None = None
    error: str | None = None
    """Why the plugin could not be loaded, if it could not."""

    @property
    def is_builtin(self) -> bool:
        return self.namespace == BUILTIN_NAMESPACE

    def brainsets(self) -> dict[str, Traversable]:
        """Map each brainset id to its pipeline folder."""
        if self.pipelines_path is None:
            return {}
        return list_pipeline_dirs(self.pipelines_path)

    def qualified_name(self, brainset_id: str) -> str:
        """Name used on the command line for ``brainset_id``."""
        if self.is_builtin:
            return brainset_id
        return f"{self.namespace}:{brainset_id}"


@dataclass
class ResolvedBrainset:
    """A brainset name from the command line, resolved to its pipeline folder."""

    source: PipelineSource
    brainset_id: str
    pipeline_dir: Traversable = field(repr=False)

    @property
    def qualified_name(self) -> str:
        return self.source.qualified_name(self.brainset_id)


class BrainsetNotFoundError(LookupError):
    """Raised when a brainset name cannot be resolved to an installed pipeline."""


def list_pipeline_dirs(path: Traversable) -> dict[str, Traversable]:
    """Map folder name to folder for every ``<name>/pipeline.py`` under ``path``.

    Folders starting with ``.`` or ``_`` are ignored.
    """
    ret = {}
    for d in sorted(path.iterdir(), key=lambda d: d.name):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        if (d / "pipeline.py").is_file():
            ret[d.name] = d
    return ret


def read_pipeline_brainset_id(pipeline_file: Traversable) -> str | None:
    """Read ``Pipeline.brainset_id`` from ``pipeline.py`` without importing it.

    Returns ``None`` if there is no ``Pipeline`` class or its ``brainset_id`` is not
    a string literal.
    """
    tree = ast.parse(pipeline_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Pipeline"):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "brainset_id"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    return value.value
    return None


def get_builtin_source() -> PipelineSource:
    from torch_brain import __version__

    return PipelineSource(
        namespace=BUILTIN_NAMESPACE,
        dist_name="torch_brain",
        dist_version=__version__,
        pipelines_path=BUILTIN_PIPELINES_PATH,
    )


def _iter_entry_points() -> list[EntryPoint]:
    return list(entry_points(group=ENTRY_POINT_GROUP))


def _source_from_entry_point(ep: EntryPoint) -> PipelineSource:
    dist = getattr(ep, "dist", None)
    dist_name = dist.name if dist is not None else ep.value.split(".")[0]
    dist_version = dist.version if dist is not None else None
    source = PipelineSource(
        namespace=ep.name,
        dist_name=dist_name,
        dist_version=dist_version,
        pipelines_path=None,
        entry_point_value=ep.value,
    )

    if not NAMESPACE_PATTERN.match(ep.name):
        source.error = (
            f"invalid namespace '{ep.name}' (only lowercase letters, digits and "
            "underscores are allowed)"
        )
    elif ep.name == BUILTIN_NAMESPACE:
        source.error = f"the namespace '{BUILTIN_NAMESPACE}' is reserved"
    elif ":" in ep.value:
        source.error = (
            f"entry point value '{ep.value}' must be a package name, "
            "not an object reference"
        )
    else:
        try:
            path = files(ep.value)
            if not path.is_dir():
                raise NotADirectoryError(f"'{ep.value}' is not a package")
            source.pipelines_path = path
        except Exception as e:  # never let a broken plugin break the CLI
            source.error = f"could not load package '{ep.value}': {e}"
    return source


def get_plugin_sources(warn: bool = True) -> dict[str, PipelineSource]:
    """Return installed plugins by namespace, including the ones that failed to load.

    If two installed distributions claim the same namespace, the first one wins and
    the other is reported as broken.
    """
    sources: dict[str, PipelineSource] = {}
    for ep in _iter_entry_points():
        source = _source_from_entry_point(ep)
        if source.namespace in sources:
            other = sources[source.namespace]
            if other.dist_name == source.dist_name:
                continue  # same distribution seen twice (e.g. duplicated sys.path)
            source.error = (
                f"namespace '{source.namespace}' is already used by '{other.dist_name}'"
            )
            source.namespace = f"{source.namespace} ({source.dist_name})"
        sources[source.namespace] = source

        if warn and source.error is not None:
            warnings.warn(
                f"Skipping brainsets plugin '{source.dist_name}': {source.error}",
                stacklevel=2,
            )
    return sources


def get_all_sources(warn: bool = True) -> list[PipelineSource]:
    """Built-in pipelines first, then installed plugins sorted by namespace."""
    plugins = get_plugin_sources(warn=warn)
    return [get_builtin_source()] + [plugins[ns] for ns in sorted(plugins)]


def split_name(name: str) -> tuple[str, str]:
    """Split ``[namespace:]brainset_id`` into ``(namespace, brainset_id)``.

    Bare names refer to built-in brainsets.
    """
    if ":" in name:
        namespace, brainset_id = name.split(":", 1)
        return namespace, brainset_id
    return BUILTIN_NAMESPACE, name


def resolve_brainset(name: str) -> ResolvedBrainset:
    """Find the pipeline folder for a brainset name given on the command line.

    Raises:
        BrainsetNotFoundError: with a message that says what to do instead.
    """
    namespace, brainset_id = split_name(name)

    if ":" not in name:
        source = get_builtin_source()
        brainsets = source.brainsets()
        if brainset_id in brainsets:
            return ResolvedBrainset(source, brainset_id, brainsets[brainset_id])

        # Plugin brainsets must be qualified; point to the right name if one exists.
        matches = [
            s.qualified_name(brainset_id)
            for s in get_plugin_sources(warn=False).values()
            if s.error is None and brainset_id in s.brainsets()
        ]
        msg = f"Brainset '{name}' not found."
        if matches:
            msg += " Plugin brainsets need their namespace, did you mean " + (
                " or ".join(f"'{m}'" for m in matches) + "?"
            )
        else:
            msg += " Run 'brainsets list' to get the available list of brainsets."
        raise BrainsetNotFoundError(msg)

    if namespace == BUILTIN_NAMESPACE:
        raise BrainsetNotFoundError(
            f"Built-in brainsets are named without a namespace: use '{brainset_id}'."
        )

    plugins = get_plugin_sources(warn=False)
    if namespace not in plugins:
        raise BrainsetNotFoundError(
            f"No installed plugin provides the namespace '{namespace}'. "
            "Run 'brainsets list' to see installed plugins."
        )

    source = plugins[namespace]
    if source.error is not None:
        raise BrainsetNotFoundError(
            f"The '{namespace}' plugin ('{source.dist_name}') could not be loaded: "
            f"{source.error}"
        )
    brainsets = source.brainsets()
    if brainset_id not in brainsets:
        raise BrainsetNotFoundError(
            f"Brainset '{brainset_id}' not found in the '{namespace}' plugin. "
            f"Run 'brainsets list' to get the available list of brainsets."
        )
    return ResolvedBrainset(source, brainset_id, brainsets[brainset_id])


def read_owner_marker(dataset_dir: Path) -> dict | None:
    """Read ``<dataset_dir>/.brainsets_source``, or return ``None`` if there is none."""
    marker = Path(dataset_dir) / OWNER_MARKER_FILENAME
    if not marker.is_file():
        return None
    try:
        with open(marker, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _normalize_dist_name(name: str) -> str:
    """Normalize a distribution name as in PEP 503 (``My_Pkg`` == ``my-pkg``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def check_owner_marker(dataset_dir: Path, source: PipelineSource) -> str | None:
    """Return why ``source`` may not write into ``dataset_dir``, or ``None`` if it may.

    A folder is compatible when it has no marker (made before markers existed), is
    owned by torch_brain, or is owned by the same distribution as ``source``.
    """
    marker = read_owner_marker(dataset_dir)
    if marker is None:
        return None
    owner = marker.get("source")
    if owner is None or _normalize_dist_name(owner) in (
        _normalize_dist_name(BUILTIN_NAMESPACE),
        _normalize_dist_name(source.dist_name),
    ):
        return None
    owner_version = marker.get("version")
    owner_str = f"'{owner}'" + (f" {owner_version}" if owner_version else "")
    return (
        f"{dataset_dir} was prepared by {owner_str}, not by '{source.dist_name}'. "
        "Two sources must not write into the same brainset folder: use another "
        "--processed-dir, or remove the folder first."
    )


def write_owner_marker(dataset_dir: Path, source: PipelineSource) -> None:
    """Write the owner marker, unless the folder already has one."""
    dataset_dir = Path(dataset_dir)
    marker = dataset_dir / OWNER_MARKER_FILENAME
    if read_owner_marker(dataset_dir) is not None:
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'source = "{source.dist_name}"']
    if source.dist_version:
        lines.append(f'version = "{source.dist_version}"')
    lines.append(f'namespace = "{source.namespace}"')
    marker.write_text("\n".join(lines) + "\n")
