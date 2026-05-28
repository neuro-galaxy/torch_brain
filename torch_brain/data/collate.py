_MOVED = {
    "collate",
    "chain",
    "pad",
    "pad8",
    "pad2d",
    "pad2d8",
    "track_batch",
    "track_mask",
    "track_mask8",
    "track_mask2d",
    "track_mask2d8",
}


def __getattr__(name):
    if name in _MOVED:
        raise ImportError(
            f"`torch_brain.data.collate.{name}` has moved to `torch_brain.batching`. "
            f"Use `from torch_brain.batching import {name}` instead."
        )
    raise AttributeError(f"module 'torch_brain.data.collate' has no attribute {name!r}")
