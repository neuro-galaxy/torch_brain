from .binning import bin_spikes
from .misc import calculate_sampling_rate, np_string_prefix
from .tokenizers import create_linspace_latent_tokens, create_start_end_unit_tokens
from .weights import isin_interval, resolve_weights_based_on_interval_membership

# from .stitcher import stitch


def seed_everything(*args, **kwargs):
    raise ImportError(
        "`seed_everything` has been removed from `torch_brain.utils`. "
        "You can find a reference implementation in `examples/poyo/utils.py`"
    )


__all__ = [
    # "stitch",
    "create_linspace_latent_tokens",
    "create_start_end_unit_tokens",
    "resolve_weights_based_on_interval_membership",
    "isin_interval",
    "np_string_prefix",
    "bin_spikes",
]

__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}
