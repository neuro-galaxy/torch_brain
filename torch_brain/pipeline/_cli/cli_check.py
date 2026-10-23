import ast
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click

from .plugins import (
    BUILTIN_NAMESPACE,
    PipelineSource,
    get_builtin_source,
    get_plugin_sources,
)
from .registry import (
    RegistryError,
    load_registry,
    load_registry_data,
    validate_registry,
)


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _static_pipeline_brainset_id(pipeline_file: Path) -> tuple[bool, str | None]:
    """Find ``class Pipeline`` and its literal ``brainset_id`` without importing."""
    tree = ast.parse(pipeline_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Pipeline":
            for stmt in node.body:
                targets = []
                if isinstance(stmt, ast.Assign):
                    targets, value = stmt.targets, stmt.value
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    targets, value = [stmt.target], stmt.value
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "brainset_id"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        return True, value.value
            return True, None
    return False, None


def _import_pipeline_cls(pipeline_file: Path):
    module_name = f"_brainsets_check_{pipeline_file.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_file)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return getattr(module, "Pipeline", None)


def check_pipeline_dir(
    pipeline_dir: Path, report: CheckReport, label: str | None = None
) -> None:
    """Check one ``<brainset_id>/pipeline.py`` folder against the pipeline rules."""
    from torch_brain.pipeline import BrainsetPipeline

    from .cli_prepare import _read_inline_metadata, _validate_python_version

    brainset_id = pipeline_dir.name
    label = label or brainset_id
    pipeline_file = pipeline_dir / "pipeline.py"
    n_errors = len(report.errors)

    if not brainset_id.isidentifier():
        report.errors.append(f"{label}: folder name is not a valid identifier")

    try:
        inline_md = _read_inline_metadata(pipeline_file)
        if inline_md is not None and "python-version" in inline_md:
            _validate_python_version(inline_md["python-version"])
    except Exception as e:
        report.errors.append(f"{label}: invalid brainset-pipeline metadata: {e}")

    try:
        pipeline_cls = _import_pipeline_cls(pipeline_file)
    except ModuleNotFoundError as e:
        # Pipelines run in their own environment, so their dependencies may be
        # missing here. Fall back to reading the source.
        report.warnings.append(
            f"{label}: could not import pipeline.py ({e}); checked it statically"
        )
        has_class, static_id = _static_pipeline_brainset_id(pipeline_file)
        if not has_class:
            report.errors.append(f"{label}: pipeline.py does not define 'Pipeline'")
        elif static_id is not None and static_id != brainset_id:
            report.errors.append(
                f"{label}: Pipeline.brainset_id is '{static_id}', "
                f"but must equal the folder name '{brainset_id}'"
            )
    except Exception as e:
        report.errors.append(f"{label}: importing pipeline.py failed: {e!r}")
    else:
        if pipeline_cls is None:
            report.errors.append(f"{label}: pipeline.py does not define 'Pipeline'")
        elif not (
            inspect.isclass(pipeline_cls) and issubclass(pipeline_cls, BrainsetPipeline)
        ):
            report.errors.append(
                f"{label}: Pipeline must subclass torch_brain.pipeline.BrainsetPipeline"
            )
        else:
            if getattr(pipeline_cls, "brainset_id", None) != brainset_id:
                report.errors.append(
                    f"{label}: Pipeline.brainset_id is "
                    f"{getattr(pipeline_cls, 'brainset_id', None)!r}, "
                    f"but must equal the folder name '{brainset_id}'"
                )
            if inspect.isabstract(pipeline_cls):
                missing = ", ".join(sorted(pipeline_cls.__abstractmethods__))
                report.errors.append(f"{label}: Pipeline does not implement {missing}")

    if len(report.errors) == n_errors:
        report.passed.append(label)


def check_source(source: PipelineSource) -> CheckReport:
    """Check an installed plugin (or the built-in pipelines) against the plugin rules."""
    report = CheckReport()
    if source.error is not None:
        report.errors.append(f"plugin could not be loaded: {source.error}")
        return report

    brainsets = source.brainsets()
    if not brainsets:
        report.errors.append(
            f"package '{source.entry_point_value}' has no <brainset_id>/pipeline.py "
            "folders (are they shipped as package data?)"
        )
    for brainset_id, pipeline_dir in brainsets.items():
        check_pipeline_dir(
            Path(str(pipeline_dir)), report, source.qualified_name(brainset_id)
        )

    if source.is_builtin:
        return report

    try:
        registry = load_registry()
    except RegistryError as e:
        report.warnings.append(f"could not read the plugin registry: {e}")
        return report
    entry = registry.get(source.namespace)
    if entry is None:
        report.warnings.append(
            f"'{source.namespace}' is not in the plugin registry, so it is not "
            "listed in 'brainsets list --all' or the docs"
        )
        return report
    if entry.package != source.dist_name:
        report.errors.append(
            f"registry: namespace '{source.namespace}' is registered to "
            f"'{entry.package}', but provided by '{source.dist_name}'"
        )
    registered = {b.id for b in entry.brainsets}
    shipped = set(brainsets)
    for brainset_id in sorted(registered - shipped):
        report.errors.append(f"registry: '{brainset_id}' is listed but not shipped")
    for brainset_id in sorted(shipped - registered):
        report.errors.append(f"registry: '{brainset_id}' is shipped but not listed")
    return report


def _echo_report(report: CheckReport) -> None:
    for label in report.passed:
        click.echo(f"OK       {label}")
    for msg in report.warnings:
        click.echo(f"WARNING  {msg}")
    for msg in report.errors:
        click.echo(f"ERROR    {msg}")


@click.command()
@click.argument("namespace", required=False)
@click.option(
    "--registry",
    "registry_path",
    is_flag=False,
    flag_value="",
    default=None,
    type=str,
    help=(
        "Validate the plugin registry instead of a plugin. Takes an optional path to "
        "a plugins.toml file (default: the registry bundled with torch_brain)."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings as errors.",
)
def check(namespace: str | None, registry_path: str | None, strict: bool):
    """Check that an installed plugin follows the brainsets plugin rules.

    NAMESPACE is the plugin's namespace (its entry point name), or 'torch_brain' to
    check the built-in pipelines. This imports each pipeline.py but never downloads
    or processes data.

    \b
    Examples:
    $ brainsets check openneuro
    $ brainsets check --registry
    $ brainsets check --registry path/to/plugins.toml
    """
    if registry_path is not None:
        if namespace is not None:
            raise click.UsageError("Pass either NAMESPACE or --registry, not both.")
        if registry_path:
            path = Path(registry_path)
            click.echo(f"Checking registry {path}")
        else:
            path = None
            click.echo("Checking the bundled plugin registry")
        try:
            errors = validate_registry(load_registry_data(path))
        except Exception as e:
            errors = [f"could not parse the registry: {e}"]
        for msg in errors:
            click.echo(f"ERROR    {msg}")
        if errors:
            sys.exit(1)
        click.echo("Registry is valid.")
        return

    if namespace is None:
        raise click.UsageError("Missing NAMESPACE (or pass --registry).")

    if namespace == BUILTIN_NAMESPACE:
        source = get_builtin_source()
    else:
        plugins = get_plugin_sources(warn=False)
        if namespace not in plugins:
            raise click.ClickException(
                f"No installed plugin provides the namespace '{namespace}'. "
                "Run 'brainsets list' to see installed plugins."
            )
        source = plugins[namespace]

    version = f" {source.dist_version}" if source.dist_version else ""
    click.echo(f"Checking '{namespace}' ({source.dist_name}{version})")
    report = check_source(source)
    _echo_report(report)

    failed = not report.ok or (strict and report.warnings)
    n = len(report.passed)
    click.echo(
        f"\n{n} brainset(s) passed, {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)."
    )
    if failed:
        sys.exit(1)
