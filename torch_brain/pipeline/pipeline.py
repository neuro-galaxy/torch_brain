import logging
import shutil
import sys
import traceback
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import ray.actor
from rich.console import Console

logger = logging.getLogger(__name__)


class BrainsetPipeline(ABC):
    r"""Abstract base class for defining processing pipelines.
    Pipelines are subclasses of this class. Pipelines are either run through
    the CLI or through :mod:`brainsets.runner` module.

    **Subclasses must implement:**
        - Set the :attr:`brainset_id`
        - :meth:`get_manifest()`: Generate a :obj:`pd.DataFrame` listing all assets to process
        - :meth:`download()`: Download a single asset from the manifest
        - :meth:`process()`: Transform downloaded data into standardized format

    **The pipeline workflow consists of:**
        1. Generating a manifest (list of assets to process) via :meth:`get_manifest()`. This happens on the root process.
        2. Downloading each asset via :meth:`download()`. This happens in parallel for multiple rows of the manifest.
        3. Processing each downloaded asset via :meth:`process()`. This also happens in parallel, on the same process as :meth:`download()`.

    **Handling pipeline-specific command line arguments:**
        - Subclasses can define pipeline-specific command-line arguments by setting :attr:`parser`.
        - The runner will automatically parse any extra CLI arguments using this parser.
        - The parsed arguments are passed to the :meth:`get_manifest()` as the `args` method parameter, and to the :meth:`download` and :meth:`process` methods via class attribute :attr:`args`.

    Example:

        >>> from argparse import ArgumentParser
        >>> parser = ArgumentParser()
        >>> parser.add_argument("--redownload", action="store_true")  # doctest: +SKIP
        >>> parser.add_argument("--reprocess", action="store_true")  # doctest: +SKIP
        >>>
        >>> class Pipeline(BrainsetPipeline):
        ...     brainset_id = "my_brainset"
        ...     parser = parser
        ...
        ...     @classmethod
        ...     def get_manifest(cls, raw_dir, processed_dir, args):
        ...         # Return DataFrame of assets to process
        ...         return pd.DataFrame(...)
        ...
        ...     def download(self, manifest_item):
        ...         # Download the asset
        ...         # return filepath or handle of downloaded data
        ...         ...
        ...
        ...     def process(self, download_output):
        ...         # Process the downloaded data
        ...         ...
    """

    brainset_id: str
    """Unique identifier for the brainset. Must be set by the Pipeline subclass."""
    parser: ArgumentParser | None = None
    """Optional :obj:`argparse.ArgumentParser` object for pipeline-specific
    command-line arguments.
    If set by a subclass, the runner will automatically parse any extra
    command-line arguments using this parser. The parsed arguments are then
    passed to `get_manifest()` as a method argument, and to the `download()` and
    `process()` methods via `self.args`.
    """
    args: Namespace | None
    """Pipeline-specific arguments parsed from the command line. Set by the runner
    if :attr:`parser` is defined by subclass.
    """
    raw_dir: Path
    """Raw data directory assigned to this brainset by the pipeline runner.
    """
    processed_dir: Path
    """Processed data directory assigned to this brainset by the pipeline runner.
    """
    _asset_id: str
    """Identifier for the current asset being processed. Set automatically in
    :meth:`run_item()`.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        args: Namespace | None,
        tracker_handle: ray.actor.ActorHandle | None = None,
        download_only: bool = False,
        delete_raw: bool = False,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.args = args
        self._tracker_handle = tracker_handle
        self._download_only = download_only
        self._delete_raw = delete_raw

    @classmethod
    @abstractmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Namespace | None,
    ) -> pd.DataFrame:
        r"""Returns a :obj:`pandas.DataFrame`, which is a table of assets to be
        downloaded and processed. Each row will be passed individually to the
        :meth:`download` and :meth:`process` methods.

        The index of this DataFrame will be used to identify assets for when user wants
        to process a single asset.

        Parameters
        ----------
        raw_dir: Path
            Raw data directory assigned to this brainset by the pipeline runner.
        args: Optional[Namespace]
            Pipeline-specific arguments parsed from the command line. Set by the runner
            if :attr:`parser` is defined by subclass.

        Returns
        -------
        pandas.DataFrame
        """
        ...

    @abstractmethod
    def download(self, manifest_item: NamedTuple) -> Any:
        r"""Download the asset indicated by `manifest_item`.
        All return values will be passed to :meth:`process()`.

        Parameters
        ----------
        manifest_item: typing.NamedTuple
            This is a single row of the manifest returned by :meth:`get_manifest()`.
        """
        ...

    @abstractmethod
    def process(self, download_output: Any):
        r"""
        Process and save the dataset.

        Parameters
        ----------
        download_output : Any
            This will be the return value of the :meth:`download()` method.
        """
        ...

    def update_status(self, status: str):
        """
        Update the current status of the pipeline for a given asset.
        This will be shown on the terminal.
        """
        from .runner import get_style

        if self._tracker_handle is not None:
            self._tracker_handle.update_status.remote(self._asset_id, status)

        Console().print(f"[bold][Status][/] [{get_style(status)}]{status}[/]")

    def cleanup_raw(self, download_output: Any) -> None:
        r"""Delete the raw file(s) for a manifest item, once it has been
        successfully processed. Called automatically after :meth:`process()`
        when the pipeline is run with ``delete_raw=True``.

        The default implementation deletes every :class:`~pathlib.Path` found
        in `download_output` (including inside lists/tuples, or behind a
        ``.path`` attribute). Pipelines whose :meth:`download` return value
        doesn't map directly to the raw files it created (e.g. it returns
        parsed metadata instead of a path, or writes files that aren't part
        of its return value) must override this method to correctly delete
        their own raw files.

        Parameters
        ----------
        download_output : Any
            The return value of :meth:`download()` for this manifest item.
        """
        paths = self._extract_raw_paths(download_output)
        if not paths:
            logger.warning(
                f"{type(self).__name__}: delete_raw is enabled, but no raw file "
                "paths could be inferred from the download() return value. "
                "Override cleanup_raw() in this pipeline to support delete_raw."
            )
            return
        self.update_status("Deleting Raw Files")
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()

    @staticmethod
    def _extract_raw_paths(value: Any) -> list[Path]:
        """Recursively collect every :class:`~pathlib.Path` found in `value`."""
        if isinstance(value, Path):
            return [value]
        if isinstance(value, (list, tuple, set)):
            paths = []
            for item in value:
                paths.extend(BrainsetPipeline._extract_raw_paths(item))
            return paths
        path_attr = getattr(value, "path", None)
        if isinstance(path_attr, Path):
            return [path_attr]
        return []

    def _run_item(self, manifest_item):
        """Run download and process for a manifest item."""
        self._asset_id = manifest_item.Index
        output = self.download(manifest_item)
        if not self._download_only:
            self.process(output)
            if self._delete_raw:
                self.cleanup_raw(output)
        self.update_status("DONE")

    def _run_item_on_parallel_worker(self, manifest_item):
        """Run item in parallel worker, capturing logs and status."""
        self._asset_id = manifest_item.Index
        log_dir = self.processed_dir / "pipeline_logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        log_out_path = log_dir / f"{self._asset_id}.out"
        log_err_path = log_dir / f"{self._asset_id}.err"

        with _redirect_stdio(log_out_path, log_err_path):
            try:
                self._run_item(manifest_item)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                self.update_status("FAILED")


@contextmanager
def _redirect_stdio(log_out_path, log_err_path):
    """Context manager to redirect stdout/stderr to files.
    This is useful when running pipelines in parallel."""
    stdout_prev = sys.stdout
    stderr_prev = sys.stderr
    with (
        open(log_out_path, "w", encoding="utf-8") as log_out,
        open(log_err_path, "w", encoding="utf-8") as log_err,
    ):
        sys.stdout = log_out
        sys.stderr = log_err
        try:
            yield
        finally:
            sys.stdout = stdout_prev
            sys.stderr = stderr_prev
