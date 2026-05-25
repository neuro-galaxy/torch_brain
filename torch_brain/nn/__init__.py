# embedding layers
from .embedding import Embedding
from .infinite_vocab_embedding import InfiniteVocabEmbedding

# rotary attention-based models
from .position_embeddings import RotaryTimeEmbedding, SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention

# readout layers
from .multitask_readout import (
    MultitaskReadout,
    prepare_for_multitask_readout,
)


def __getattr__(name):
    if name == "loss":
        raise ImportError("`torch_brain.nn.loss` has been removed")
    raise AttributeError(f"module 'torch_brain.nn' has no attribute {name!r}")


__all__ = [
    "Embedding",
    "InfiniteVocabEmbedding",
    "RotaryTimeEmbedding",
    "SinusoidalTimeEmbedding",
    "RotaryCrossAttention",
    "RotarySelfAttention",
    "MultitaskReadout",
    "prepare_for_multitask_readout",
]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Embedding modules",
            "autosummary": [
                "Embedding",
                "InfiniteVocabEmbedding",
                "RotaryTimeEmbedding",
                "SinusoidalTimeEmbedding",
            ],
        },
        {
            "title": "Transformer related modules",
            "autosummary": [
                "RotaryCrossAttention",
                "RotarySelfAttention",
            ],
        },
        {
            "title": "Readout",
            "autosummary": [
                "MultitaskReadout",
                "prepare_for_multitask_readout",
            ],
        },
    ],
}
