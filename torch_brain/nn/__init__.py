# embedding layers
from .infinite_vocab_embedding import InfiniteVocabEmbedding

# rotary attention-based models
from .position_embeddings import RotaryTimeEmbedding, SinusoidalTimeEmbedding
from .rotary_attention import RotaryCrossAttention, RotarySelfAttention

__all__ = [
    "InfiniteVocabEmbedding",
    "RotaryTimeEmbedding",
    "SinusoidalTimeEmbedding",
    "RotaryCrossAttention",
    "RotarySelfAttention",
]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "title": "Embedding modules",
            "autosummary": [
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
    ],
}
