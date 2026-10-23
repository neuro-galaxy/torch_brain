import click

from .plugins import (
    BUILTIN_NAMESPACE,
    BrainsetNotFoundError,
    get_builtin_source,
    get_plugin_sources,
    resolve_brainset,
    split_name,
)
from .registry import install_hint, load_registry


def _echo_field(name: str, value) -> None:
    if value:
        click.echo(f"{name + ':':<14}{value}")


def _plugin_info(namespace: str) -> None:
    plugins = get_plugin_sources(warn=False)
    registry = load_registry()
    source = plugins.get(namespace)
    entry = registry.get(namespace)

    click.echo(f"Plugin: {namespace}")
    if entry is not None:
        _echo_field("Description", entry.description)
        _echo_field("Package", entry.package)
        _echo_field("Repository", entry.repository)
        _echo_field("Docs", entry.docs)
        _echo_field("License", entry.license)
        _echo_field("Maintainers", ", ".join(entry.maintainers))
        _echo_field("Modalities", ", ".join(entry.modalities))
    else:
        click.echo("Not in the plugin registry.")

    if source is None:
        _echo_field("Installed", "no")
        click.echo("\nBrainsets:")
        for brainset in entry.brainsets:
            click.echo(f"- {namespace}:{brainset.id}: {brainset.description}")
        click.echo(f"\n{install_hint(entry)}")
        return

    version = f" {source.dist_version}" if source.dist_version else ""
    _echo_field("Installed", f"yes ({source.dist_name}{version})")
    if source.error is not None:
        _echo_field("Error", source.error)
        return
    descriptions = {b.id: b.description for b in entry.brainsets} if entry else {}
    click.echo("\nBrainsets:")
    for brainset_id in source.brainsets():
        desc = descriptions.get(brainset_id)
        line = f"- {namespace}:{brainset_id}"
        click.echo(f"{line}: {desc}" if desc else line)
    if entry is not None:
        click.echo(
            f"\n'{namespace}' is a community plugin, maintained by its authors, "
            "not by torch_brain."
        )


def _brainset_info(name: str) -> None:
    from .cli_prepare import _read_inline_metadata

    namespace, brainset_id = split_name(name)
    registry = load_registry()
    entry = registry.get(namespace)
    registered = (
        next((b for b in entry.brainsets if b.id == brainset_id), None)
        if entry
        else None
    )

    try:
        resolved = resolve_brainset(name)
    except BrainsetNotFoundError as e:
        if registered is None:
            raise click.ClickException(str(e)) from e
        click.echo(f"Brainset: {namespace}:{brainset_id}")
        _echo_field("Description", registered.description)
        _echo_field("Source", registered.source)
        _echo_field("Plugin", f"{namespace} ({entry.package})")
        _echo_field("Installed", "no")
        click.echo(f"\n{install_hint(entry)}")
        return

    source = resolved.source
    click.echo(f"Brainset: {resolved.qualified_name}")
    if registered is not None:
        _echo_field("Description", registered.description)
        _echo_field("Source", registered.source)
    version = f" {source.dist_version}" if source.dist_version else ""
    origin = "built-in" if source.is_builtin else "community plugin"
    _echo_field("Provided by", f"{source.dist_name}{version} ({origin})")
    _echo_field("Pipeline", str(resolved.pipeline_dir / "pipeline.py"))
    _echo_field("Output folder", f"<processed_dir>/{brainset_id}")
    inline_md = _read_inline_metadata(resolved.pipeline_dir / "pipeline.py")
    if inline_md:
        _echo_field("Python", inline_md.get("python-version"))
        _echo_field("Dependencies", ", ".join(inline_md.get("dependencies", [])))


@click.command()
@click.argument("name")
def info(name: str):
    """Show details about a plugin or a brainset.

    NAME is a plugin namespace (e.g. 'openneuro'), a plugin brainset
    ('<namespace>:<brainset_id>') or a built-in brainset. Plugins from the
    registry are shown even if they are not installed.

    \b
    Examples:
    $ brainsets info openneuro
    $ brainsets info openneuro:klinzing_sleep_ds005555
    $ brainsets info pei_pandarinath_nlb_2021
    """
    if ":" not in name:
        if name == BUILTIN_NAMESPACE:
            source = get_builtin_source()
            click.echo(f"Built-in brainsets (torch_brain {source.dist_version}):")
            for brainset_id in source.brainsets():
                click.echo(f"- {brainset_id}")
            return
        is_builtin = name in get_builtin_source().brainsets()
        if not is_builtin and (
            name in get_plugin_sources(warn=False) or name in load_registry()
        ):
            _plugin_info(name)
            return
    _brainset_info(name)
