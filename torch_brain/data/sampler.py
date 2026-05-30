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
            f"`torch_brain.data.sampler.{name}` has moved to `torch_brain.samplers`. "
            f"Use `from torch_brain.samplers import {name}` instead."
        )
    raise AttributeError(f"module 'torch_brain.data.sampler' has no attribute {name!r}")
