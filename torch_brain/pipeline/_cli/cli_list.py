import click

from .plugins import get_all_sources
from .registry import load_registry


def _source_header(source) -> str:
    if source.is_builtin:
        return f"{source.namespace} (built-in):"
    version = f" {source.dist_version}" if source.dist_version else ""
    return f"{source.namespace} ({source.dist_name}{version}, community plugin):"


@click.command()
@click.option(
    "--all",
    "all_",
    is_flag=True,
    default=False,
    help="Also list community plugins from the registry that are not installed.",
)
def cli_list(all_: bool):
    """List available brainsets."""
    sources = get_all_sources(warn=False)
    broken = [s for s in sources if s.error is not None]

    click.echo("Available brainsets:")
    for source in sources:
        if source.error is not None:
            continue
        click.echo(f"\n{_source_header(source)}")
        for brainset_id in source.brainsets():
            click.echo(f"- {source.qualified_name(brainset_id)}")

    if broken:
        click.echo("\nPlugins that could not be loaded:")
        for source in broken:
            click.echo(f"- {source.namespace} ({source.dist_name}): {source.error}")

    if all_:
        installed = {s.namespace for s in sources}
        missing = [p for ns, p in load_registry().items() if ns not in installed]
        click.echo("\nNot installed (community plugins from the registry):")
        if not missing:
            click.echo("  none")
        for plugin in missing:
            click.echo(f"\n{plugin.namespace} ({plugin.package}): {plugin.description}")
            for brainset in plugin.brainsets:
                click.echo(f"- {plugin.namespace}:{brainset.id}")
        if missing:
            click.echo(
                "\nRun 'brainsets info <namespace>' for details and the install command."
            )
