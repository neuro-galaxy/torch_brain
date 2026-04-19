import click

from .utils import get_available_brainsets


@click.command()
def cli_list():
    """List available brainsets."""
    click.echo("Available brainsets:")
    for brainset in get_available_brainsets():
        click.echo(f"- {brainset}")
