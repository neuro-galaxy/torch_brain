from .binning import bin_spikes
from .misc import calculate_sampling_rate as calculate_sampling_rate
from .misc import np_string_prefix
from .weights import isin_interval

# from .stitcher import stitch


def seed_everything(*args, **kwargs):
    raise ImportError(
        "`seed_everything` has been removed from `torch_brain.utils`. "
        "You can find a reference implementation in `examples/poyo/utils.py`"
    )


__all__ = [
    # "stitch",
    "isin_interval",
    "np_string_prefix",
    "bin_spikes",
]

__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}
