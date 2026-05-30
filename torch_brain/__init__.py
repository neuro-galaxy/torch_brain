from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torch_brain")
except PackageNotFoundError:  # pragma: no cover
    # This can happen if someone is importing the package without installing
    __version__ = "unknown"  # pragma: no cover
