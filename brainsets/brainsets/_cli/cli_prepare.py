from pathlib import Path
from typing import Optional
import click
import subprocess
from prompt_toolkit import prompt

from .utils import (
    PIPELINES_PATH,
    load_config,
    get_available_brainsets,
    expand_path,
)


@click.command()
@click.argument("brainset", type=str)
@click.option("-c", "--cores", default=4, help="Number of cores to use. (default 4)")
@click.option(
    "--raw-dir",
    type=click.Path(file_okay=False),
    help="Path for storing raw data. Overrides config.",
)
@click.option(
    "--processed-dir",
    type=click.Path(file_okay=False),
    help="Path for storing processed brainset. Overrides config.",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help=(
        "Prepare brainset with from a local pipeline. "
        "BRAINSET must then be set to the path of the local brainset pipeline directory."
    ),
)
@click.option(
    "--use-active-env",
    is_flag=True,
    default=False,
    help=(
        "Developer flag. If set, will not create an isolated environment. "
        "Only set if you know what you're doing."
    ),
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print debugging information.",
)
def prepare(
    brainset: str,
    cores: int,
    verbose: bool,
    use_active_env: bool,
    raw_dir: Optional[str],
    processed_dir: Optional[str],
    local: bool,
):
    """Download and process a single brainset.

    Run 'brainsets list' to get a list of available brainsets.

    \b
    Examples:
    $ brainsets prepare pei_pandarinath_nlb_2021
    $ brainsets prepare pei_pandarinath_nlb_2021 --cores 8 --raw-dir ~/data/raw --processed-dir ~/data/processed
    $ brainsets prepare ./my_local_brainsets_pipeline --local
    """

    # Get raw and processed dirs
    if raw_dir is None or processed_dir is None:
        config = load_config()
        raw_dir = expand_path(raw_dir or config["raw_dir"])
        processed_dir = expand_path(processed_dir or config["processed_dir"])
    else:
        raw_dir = expand_path(raw_dir)
        processed_dir = expand_path(processed_dir)

    if not local:
        # Preparing using an OG pipeline
        available_brainsets = get_available_brainsets()
        if brainset not in available_brainsets:
            raise click.ClickException(
                f"Brainset '{brainset}' not found. "
                f"Run 'brainsets list' to get the available list of brainsets."
            )
        # Find snakefile
        snakefile_filepath = PIPELINES_PATH / brainset / "Snakefile"
        reqs_filepath = PIPELINES_PATH / brainset / "requirements.txt"

        _validate_snakefile(snakefile_filepath)
        click.echo(f"Preparing {brainset}...")
    else:
        # Preparing using a local pipeline
        pipeline_dir = expand_path(brainset)
        snakefile_filepath = pipeline_dir / "Snakefile"
        reqs_filepath = pipeline_dir / "requirements.txt"

        _validate_snakefile(snakefile_filepath)
        click.echo(f"Preparing local pipeline: {pipeline_dir}")

    click.echo(f"Raw data directory: {raw_dir}")
    click.echo(f"Processed data directory: {processed_dir}")

    # Construct base Snakemake command with configuration
    command = [
        "snakemake",
        "-s",
        str(snakefile_filepath),
        "--config",
        f"RAW_DIR={raw_dir}",
        f"PROCESSED_DIR={processed_dir}",
        f"-c{cores}",
        "all",
        "--verbose" if verbose else "--quiet",
    ]

    if use_active_env:
        click.echo(
            "WARNING: Working in active environment due to --use-active-env.\n"
            "         This mode is only intended for brainset development purposes."
        )
        if reqs_filepath.exists():
            click.echo(
                f"WARNING: {reqs_filepath} found.\n"
                f"         These will not be installed automatically due to --use-active-env usage.\n"
                f"         Make sure to install necessary requirements manually."
            )
    elif reqs_filepath.exists():
        # If dataset has additional requirements, prefix command with uv package manager
        if not use_active_env:
            uv_prefix_command = [
                "uv",
                "run",
                "--with-requirements",
                str(reqs_filepath),
                "--isolated",
                "--no-project",
            ]
            if verbose:
                uv_prefix_command.append("--verbose")

            command = uv_prefix_command + command
            click.echo(
                "Building temporary virtual environment using"
                f" requirements from {reqs_filepath}"
            )

    # Run snakemake workflow for dataset download with live output
    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True,
        )

        if process.returncode == 0:
            click.echo(f"Successfully downloaded {brainset}")
        else:
            click.echo("Error downloading dataset")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Command failed with return code {e.returncode}")
    except Exception as e:
        click.echo(f"Error: {str(e)}")


def _validate_snakefile(filepath: Path) -> bool:

    # Check if Snakefile exists
    if not filepath.exists():
        raise click.ClickException(
            f"Missing {filepath}. A pipeline must have a Snakefile."
        )

    # Check if rule "all" exists in the Snakefile
    try:
        result = subprocess.run(
            [
                "snakemake",
                "-s",
                str(filepath),
                "--list-target-rules",
                "--config",
                "RAW_DIR=test",
                "PROCESSED_DIR=test",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if "all" not in result.stdout.splitlines():
            raise click.ClickException(
                f"Rule 'all' not found in {filepath}. "
                " A valid Snakefile must have an 'all' rule."
            )
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Error validating Snakefile: {e}")
