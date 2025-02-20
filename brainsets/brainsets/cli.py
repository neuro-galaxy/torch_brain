import click
import json
from pathlib import Path
import subprocess
import brainsets_pipelines


CONFIG_FILE = Path.home() / ".brainsets_config.json"
PIPELINES_PATH = Path(brainsets_pipelines.__path__[0])
DATASETS = [d.name for d in PIPELINES_PATH.iterdir() if d.is_dir()]


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"raw_dir": None, "processed_dir": None}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


@click.group()
def cli():
    """Brainsets CLI tool."""
    pass


@cli.command()
@click.argument("dataset", type=click.Choice(DATASETS, case_sensitive=False))
@click.option("-c", "--cores", default=4, help="Number of cores to use")
def prepare(dataset, cores):
    """Download and process a specific dataset."""
    click.echo(f"Preparing {dataset}...")

    # Get config to check if directories are set
    config = load_config()
    if not config["raw_dir"] or not config["processed_dir"]:
        click.echo(
            "Error: Please set raw and processed directories first using 'brainsets config'"
        )
        return

    # Run snakemake workflow for dataset download with live output
    try:
        process = subprocess.run(
            [
                "snakemake",
                "-s",
                str(Path(__file__).parent.parent / "brainsets_pipelines" / "Snakefile"),
                "--config",
                f"raw_dir={config['raw_dir']}",
                f"processed_dir={config['processed_dir']}",
                f"-c{cores}",
                f"{dataset}",
            ],
            check=True,
            capture_output=False,
            text=True,
        )

        if process.returncode == 0:
            click.echo(f"Successfully downloaded {dataset}")
        else:
            click.echo("Error downloading dataset")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Command failed with return code {e.returncode}")
    except Exception as e:
        click.echo(f"Error: {str(e)}")


@cli.command()
def list():
    """List available datasets."""
    click.echo("Available datasets:")
    for dataset in DATASETS:
        click.echo(f"- {dataset}")


@cli.command()
@click.option(
    "--raw",
    prompt="Enter raw data directory",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
@click.option(
    "--processed",
    prompt="Enter processed data directory",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
)
def config(raw, processed):
    """Set raw and processed data directories."""
    # Create directories if they don't exist
    import os

    # If no arguments provided, prompt for input
    if raw is None or processed is None:
        if raw is None:
            raw = click.prompt(
                "Enter raw data directory",
                type=click.Path(file_okay=False, dir_okay=True),
            )
        if processed is None:
            processed = click.prompt(
                "Enter processed data directory",
                type=click.Path(file_okay=False, dir_okay=True),
            )

    raw = Path(os.path.expanduser(raw))
    processed = Path(os.path.expanduser(processed))

    # Create directories
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # Save config
    raw = str(raw)
    processed = str(processed)

    config = load_config()
    config["raw_dir"] = raw
    config["processed_dir"] = processed
    save_config(config)
    click.echo("Configuration updated successfully.")
    click.echo(f"Raw data directory: {raw}")
    click.echo(f"Processed data directory: {processed}")


if __name__ == "__main__":
    cli()
