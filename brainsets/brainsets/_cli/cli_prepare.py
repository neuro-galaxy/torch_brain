import os
import sys
from pathlib import Path
from typing import Optional
import click
import subprocess
from prompt_toolkit import prompt
import re
import json

from .utils import (
    PIPELINES_PATH,
    load_config,
    get_available_brainsets,
    expand_path,
)


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
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
@click.pass_context
def prepare(
    ctx: click.Context,
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
        pipeline_dir = PIPELINES_PATH / brainset
        prepare_filepath = pipeline_dir / "pipeline.py"
        reqs_filepath = pipeline_dir / "requirements.txt"

        click.echo(f"Preparing {brainset}...")
    else:
        # Preparing using a local pipeline
        pipeline_dir = expand_path(brainset)
        prepare_filepath = pipeline_dir / "pipeline.py"
        reqs_filepath = pipeline_dir / "requirements.txt"

        click.echo(f"Preparing local pipeline: {pipeline_dir}")

    click.echo(f"Raw data directory: {raw_dir}")
    click.echo(f"Processed data directory: {processed_dir}")

    # Construct base Snakemake command with configuration
    command = [
        "python",
        "-m",
        "brainsets.runner",
        str(prepare_filepath),
        f"--raw-dir={raw_dir}",
        f"--processed-dir={processed_dir}",
        f"-c{cores}",
        *ctx.args,  # extra arguments
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
                "--directory",
                str(pipeline_dir),
                "--isolated",
                "--no-project",
            ]

            has_brainsets = _brainsets_in_requirements(reqs_filepath)
            if not has_brainsets:
                brainsets_spec = _determine_brainsets_spec()
                click.echo(f"Detected brainsets installation from {brainsets_spec}")
                if brainsets_spec.startswith("file://"):
                    # UV can be weird about caching local packages
                    # So, if we want to recreate a local version of the package,
                    # it is safer to do so in editable mode, which does not go
                    # through UV's caching.
                    uv_prefix_command.extend(["--with-editable", brainsets_spec])
                else:
                    uv_prefix_command.extend(["--with", brainsets_spec])

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

    except subprocess.CalledProcessError as e:
        click.echo(f"Error: Command failed with return code {e.returncode}")
        sys.exit(e.returncode or 1)
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


def _brainsets_in_requirements(reqs_filepath: Path) -> tuple[list[str], bool]:
    with open(reqs_filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if stripped and re.search(r"\bbrainsets\b", stripped, re.IGNORECASE):
            return True

    return False


def _determine_brainsets_spec() -> str:
    """
    Determine how to install brainsets when not specified in requirements.txt.

    Priority:
    1. CI environment (install from current branch)
    2. Detect current installation source (git, local)
    3. Default (assume downloaded from PyPI)
    """

    # First, check if we're in CI
    if os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true":
        repo_url = os.environ.get("GITHUB_REPOSITORY", "neuro-galaxy/brainsets")
        commit_sha = os.environ.get("GITHUB_SHA")
        return f"git+https://github.com/{repo_url}.git@{commit_sha}"

    # Second, try to detect if brainsets was installed via a URL or local file
    url_source = _detect_brainsets_installation_url()
    if url_source:
        return url_source

    # Default: install from PyPI (latest or read version from installed package)
    try:
        import brainsets

        return f"brainsets=={brainsets.__version__}"
    except (ImportError, AttributeError):
        return "brainsets"


def _detect_brainsets_installation_url() -> Optional[str]:
    """
    Detect if the current brainsets package was installed via something like
    pip install <url>.
    """

    from importlib.metadata import distribution, PackageNotFoundError

    try:
        dist = distribution("brainsets")
    except PackageNotFoundError:
        return None

    # Check direct_url.json for installation source (PEP 610)
    direct_url_file = dist.files and next(
        (f for f in dist.files if f.name.endswith("direct_url.json")), None
    )

    if direct_url_file:
        direct_url_path = Path(dist.locate_file(direct_url_file))
        with open(direct_url_path, "r") as f:
            direct_url = json.load(f)

        url_info = direct_url.get("url", "")
        if len(url_info) > 0:
            # Check if installed from git
            if url_info.startswith("https://github.com"):
                # Use requested_revision (branch/tag) if available, otherwise commit_id
                vcs_info = direct_url.get("vcs_info", {})
                commit_id = vcs_info.get("commit_id")
                requested_revision = vcs_info.get("requested_revision")  # brain/tag
                ref = requested_revision or commit_id or "main"
                return f"git+{url_info}@{ref}"
            else:
                return url_info

    return None
