import json
import re
import subprocess
import sys
from pathlib import Path

import click

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-fount]
except ModuleNotFoundError:  # Python <3.11
    import tomli as tomllib  # type: ignore[import-not-found]

from click.shell_completion import CompletionItem

from torch_brain.pipeline.config import CONFIG_FILE, load_config

from .plugins import (
    BrainsetNotFoundError,
    check_owner_marker,
    resolve_brainset,
    write_owner_marker,
)
from .utils import (
    expand_path,
    get_available_brainsets,
)


def complete_brainset(ctx, param, incomplete):
    if ctx.params.get("local"):
        return [CompletionItem(incomplete, type="dir")]

    try:
        names = get_available_brainsets()
    except Exception:
        return []
    return [CompletionItem(name) for name in names if name.startswith(incomplete)]


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("brainset", type=str, shell_complete=complete_brainset)
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
    "--list",
    "list_flag_",
    is_flag=True,
    default=False,
    help="Print the manifest for the brainset and exit.",
)
@click.option(
    "--single",
    "-s",
    type=str,
    default=None,
    help="Prepare a single item from the manifest if provided. Value: manifest index id for the item.",
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
    "--download-only",
    is_flag=True,
    default=False,
    help="Only download raw data, skip processing.",
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
    download_only: bool,
    list_flag_: bool,
    single: str,
    use_active_env: bool,
    raw_dir: str | None,
    processed_dir: str | None,
    local: bool,
):
    """Download and process a single brainset.

    Run 'brainsets list' to get a list of available brainsets. Brainsets from
    plugins are named '<namespace>:<brainset_id>'.

    \b
    Examples:
    $ brainsets prepare pei_pandarinath_nlb_2021
    $ brainsets prepare pei_pandarinath_nlb_2021 --download-only
    $ brainsets prepare pei_pandarinath_nlb_2021 --cores 8 --raw-dir ~/data/raw --processed-dir ~/data/processed
    $ brainsets prepare pei_pandarinath_nlb_2021 --list
    $ brainsets prepare pei_pandarinath_nlb_2021 --single jenkins_maze_train
    $ brainsets prepare openneuro:klinzing_sleep_ds005555
    $ brainsets prepare ./my_local_brainsets_pipeline --local
    """

    # Get raw and processed dirs
    if raw_dir is None or processed_dir is None:
        config = load_config()
        if config is None:
            raise click.ClickException(
                f"Config not found or invalid at {CONFIG_FILE}. "
                "Please run `brainsets config set`."
            )
        raw_dir = expand_path(raw_dir or config["raw_dir"])
        processed_dir = expand_path(processed_dir or config["processed_dir"])
    else:
        raw_dir = expand_path(raw_dir)
        processed_dir = expand_path(processed_dir)

    source = None
    if not local:
        # Preparing using a built-in or plugin pipeline
        try:
            resolved = resolve_brainset(brainset)
        except BrainsetNotFoundError as e:
            raise click.ClickException(str(e)) from e
        source = resolved.source
        pipeline_dir = resolved.pipeline_dir
        if source.is_builtin:
            click.echo(f"Preparing {resolved.brainset_id}...")
        else:
            version = f" {source.dist_version}" if source.dist_version else ""
            click.echo(
                f"Preparing {resolved.qualified_name} "
                f"(community plugin {source.dist_name}{version})..."
            )

        # Refuse to mix the output of two different sources in one folder
        dataset_dir = processed_dir / resolved.brainset_id
        conflict = check_owner_marker(dataset_dir, source)
        if conflict is not None:
            raise click.ClickException(conflict)
        if not list_flag_:
            write_owner_marker(dataset_dir, source)
    else:
        # Preparing using a local pipeline
        pipeline_dir = expand_path(brainset)
        click.echo(f"Preparing local pipeline: {pipeline_dir}")

    pipeline_filepath = pipeline_dir / "pipeline.py"

    click.echo(f"Raw data directory: {raw_dir}")
    click.echo(f"Processed data directory: {processed_dir}")

    inline_md = _read_inline_metadata(pipeline_filepath)
    if verbose:
        click.echo(f"Inline metadata: {inline_md}")

    # Construct command to run pipeline through runner
    command = [
        "python",
        "-m",
        "torch_brain.pipeline.runner",
        str(pipeline_filepath),
        f"--raw-dir={raw_dir}",
        f"--processed-dir={processed_dir}",
        f"-c{cores}",
        *(["--download-only"] if download_only else []),
        *(["--list"] if list_flag_ else []),
        *([f"--single={single}"] if single is not None else []),
        *ctx.args,  # extra arguments
    ]

    if use_active_env:
        click.echo(
            "WARNING: Working in active environment due to --use-active-env.\n"
            "         This mode is only intended for brainset development purposes."
        )
    elif inline_md is not None:
        # Additional dependencies / Python version specified in inline metadata
        # We need to create an isolated environment for the pipeline to ensure
        # that the pipeline runs with the correct dependencies and Python version.

        uv_prefix_command = [
            "uv",
            "run",
            "--directory",
            str(pipeline_dir),
            "--isolated",
            "--no-project",
        ]

        if "python-version" in inline_md:
            python_version = inline_md["python-version"]
            try:
                _validate_python_version(python_version)
            except ValueError as e:
                raise click.ClickException(str(e)) from e
            uv_prefix_command.extend(["--python", python_version])

        deps = inline_md.get("dependencies", [])

        # Ensure a reasonable brainsets install spec is added to the dependencies.
        # If not already present in the inline dependencies list, make a best guess
        # at the install spec to use.
        # The typical case is for brainsets to not be in the inline dependencies list.
        # Plugin pipelines also need the plugin itself, at the version the user
        # installed, since they import from it.
        dist_names = ["torch_brain"]
        if source is not None and not source.is_builtin:
            dist_names.append(source.dist_name)

        for dist_name in dist_names:
            if _dist_in_dependencies(deps, dist_name):
                continue
            spec = _determine_install_spec(dist_name)
            click.echo(f"Detected {dist_name} installation from {spec}")
            if spec.startswith("file://"):
                # UV can be weird about caching local packages
                # So, if we want to recreate a local version of the package,
                # it is safer to do so in editable mode, which does not go
                # through UV's caching.
                uv_prefix_command.extend(["--with-editable", spec])
            else:
                deps.append(spec)

        if len(deps) > 0:
            uv_prefix_command.extend(["--with", ",".join(deps)])

        if verbose:
            uv_prefix_command.append("--verbose")

        command = uv_prefix_command + command
        click.echo(f"Building temporary virtual environment for {pipeline_filepath}")

    if verbose:
        click.echo(f"Command: {command}")

    try:
        subprocess.run(
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


def _validate_python_version(python_version) -> None:
    """Check that python_version is a single version number (e.g., "3.10").

    Version ranges are rejected because uv run sometimes fails (stalls) with them.
    """
    if (
        not isinstance(python_version, str)
        or not python_version.strip().replace(".", "", 1).isdigit()
    ):
        raise ValueError(
            f"Invalid python version in inline metadata: '{python_version}'. "
            "Only a single version (e.g., '3.10') is allowed, not a range."
        )


def _dist_in_dependencies(dependencies: list[str], dist_name: str) -> bool:
    """Check if any dependency refers to the given distribution.

    Distribution names are compared as in PEP 503 ('-', '_' and '.' are equivalent).
    """
    parts = re.split(r"[-_.]+", dist_name)
    pattern = r"(?<![\w.-])" + r"[-_.]+".join(map(re.escape, parts)) + r"(?![\w.-])"
    for line in dependencies:
        stripped = line.strip()
        if stripped and re.search(pattern, stripped, re.IGNORECASE):
            return True

    return False


def _determine_install_spec(dist_name: str) -> str:
    """
    Determine how to install a distribution when not specified in the pipeline's
    inline metadata.

    Priority:
    1. Detect current installation source (git, local) from package metadata
    2. Default (assume downloaded from PyPI, at the installed version)
    """

    # Try to detect if the distribution was installed via a URL or local file
    url_source = _detect_installation_url(dist_name)
    if url_source:
        return url_source

    # Default: install from PyPI (latest or read version from installed package)
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"{dist_name}=={version(dist_name)}"
    except PackageNotFoundError:
        return dist_name


def _detect_installation_url(dist_name: str) -> str | None:
    """
    Detect if the given distribution was installed via something like
    pip install <url>.
    """

    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution(dist_name)
    except PackageNotFoundError:
        return None

    # Check direct_url.json for installation source (PEP 610)
    direct_url_file = dist.files and next(
        (f for f in dist.files if f.name.endswith("direct_url.json")), None
    )

    if direct_url_file:
        direct_url_path = Path(dist.locate_file(direct_url_file))
        with open(direct_url_path) as f:
            direct_url = json.load(f)

        url_info = direct_url.get("url", "")
        if len(url_info) > 0:
            # Check if installed from git
            if url_info.startswith("https://github.com") or "vcs_info" in direct_url:
                # Use requested_revision (branch/tag) if available, otherwise commit_id
                vcs_info = direct_url.get("vcs_info", {})
                commit_id = vcs_info.get("commit_id")
                requested_revision = vcs_info.get("requested_revision")  # brain/tag
                ref = requested_revision or commit_id or "main"
                return f"git+{url_info}@{ref}"
            else:
                return url_info

    return None


_ALLOWED_INLINE_MD_KEYS = {
    "python-version",
    "dependencies",
}


def _read_inline_metadata(filepath: Path) -> dict | None:
    """
    Extract and parse the inline metadata block of type '# /// brainset-pipeline' from the given script file.

    Searches the specified file for a '# /// brainset-pipeline' block delimited by lines starting with '# /// brainset-pipeline'
    and ending with '# ///'. The contents between these markers are parsed as TOML and returned as a dict.
    If no such block is found, returns None.
    Raises ValueError if multiple 'brainset-pipeline' metadata blocks are found (following PEP 723).

    Implementation adapted from reference implementation in PEP 723:
    https://peps.python.org/pep-0723/#reference-implementation
    """
    with open(filepath, encoding="utf-8") as f:
        script = f.read()

    REGEX = r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$"
    name = "brainset-pipeline"
    matches = list(
        filter(lambda m: m.group("type") == name, re.finditer(REGEX, script))
    )
    if len(matches) > 1:
        raise ValueError(f"Multiple {name} blocks found")
    elif len(matches) == 1:
        content = "".join(
            line[2:] if line.startswith("# ") else line[1:]
            for line in matches[0].group("content").splitlines(keepends=True)
        )

        parsed = tomllib.loads(content)
        if not set(parsed.keys()).issubset(_ALLOWED_INLINE_MD_KEYS):
            unsupported_keys = set(parsed.keys()) - _ALLOWED_INLINE_MD_KEYS
            raise ValueError(
                f"Unsupported key(s) in brainset-pipeline metadata block: "
                f"{', '.join(unsupported_keys)}"
            )

        return parsed
    else:
        return None
