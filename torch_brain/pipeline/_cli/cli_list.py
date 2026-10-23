import click

from .plugins import get_all_sources


def _source_header(source) -> str:
    if source.is_builtin:
        return f"{source.namespace} (built-in):"
    version = f" {source.dist_version}" if source.dist_version else ""
    return f"{source.namespace} ({source.dist_name}{version}, community plugin):"


@click.command()
def cli_list():
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
