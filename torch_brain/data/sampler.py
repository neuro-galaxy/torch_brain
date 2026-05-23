def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    raise ImportError(
        f"`torch_brain.data.sampler.{name}` has moved to `torch_brain.samplers`. "
        f"Use `from torch_brain.samplers import {name}` instead."
    )
