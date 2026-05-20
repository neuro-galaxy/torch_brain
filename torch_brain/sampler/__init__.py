from .random_fixed_window import RandomFixedWindowSampler
from .sequential_fixed_window import SequentialFixedWindowSampler
from .trial_sampler import TrialSampler
from .distributed_sampler import (
    DistributedEvaluationSamplerWrapper,
    DistributedStitchingFixedWindowSampler,
)

__all__ = [
    "RandomFixedWindowSampler",
    "SequentialFixedWindowSampler",
    "TrialSampler",
    "DistributedEvaluationSamplerWrapper",
    "DistributedStitchingFixedWindowSampler",
]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": "See :ref:`sampling` for further details.",
    "sections": [{"autosummary": __all__}],
}
