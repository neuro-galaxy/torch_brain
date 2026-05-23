_MOVED = {
    "RandomFixedWindowSampler",
    "SequentialFixedWindowSampler",
    "TrialSampler",
    "DistributedEvaluationSamplerWrapper",
    "DistributedStitchingFixedWindowSampler",
}


def __getattr__(name):
    if name in _MOVED:
        raise ImportError(
            f"`torch_brain.data.sampler.{name}` has moved to `torch_brain.sampler`. "
            f"Use `from torch_brain.sampler import {name}` instead."
        )
    raise AttributeError(f"module 'torch_brain.data.sampler' has no attribute {name!r}")
