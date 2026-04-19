# embedding layers
from .embedding import Embedding
from .infinite_vocab_embedding import InfiniteVocabEmbedding

# rotary attention-based models
from .position_embeddings import RotaryTimeEmbedding, SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention
from .feedforward import FeedForward

# loss
from .loss import Loss, MSELoss, CrossEntropyLoss, MallowDistanceLoss

# readout layers
from .multitask_readout import (
    MultitaskReadout,
    prepare_for_multitask_readout,
)

__all__ = [
    "Embedding",
    "InfiniteVocabEmbedding",
    "RotaryTimeEmbedding",
    "SinusoidalTimeEmbedding",
    "RotaryCrossAttention",
    "RotarySelfAttention",
    "FeedForward",
    "Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "MallowDistanceLoss",
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
                "FeedForward",
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
        {
            "title": "Losses",
            "autosummary": [
                "Loss",
                "MSELoss",
                "CrossEntropyLoss",
                "MallowDistanceLoss",
            ],
        },
    ],
}
