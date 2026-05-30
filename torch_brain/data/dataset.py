_MOVED = {
    "Dataset",
}


def __getattr__(name):
    if name == "Dataset":
        raise ImportError(
            f"`torch_brain.data.dataset.Dataset` is deprecated."
            "Please use torch_brain.dataset.Dataset"
        )
    raise AttributeError(f"module 'torch_brain.data.dataset' has no attribute {name!r}")
