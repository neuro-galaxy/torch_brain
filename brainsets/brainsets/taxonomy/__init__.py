# pragma: no cover: start

_MOVED = [
    "Species",
    "Sex",
    "Task",
    "Orientation_8_Classes",
    "Macaque",
    "Cre_line",
    "RecordingTech",
    "Hemisphere",
]


def __getattr__(name):
    if name in _MOVED:
        raise ImportError(
            f"`brainsets.taxonomy.{name}` has been deprecated."
            f" Please directly encode the metadata."
        )
    raise AttributeError(f"module 'brainsets.taxonomy' has no attribute {name!r}")


# pragma: no cover: stop
