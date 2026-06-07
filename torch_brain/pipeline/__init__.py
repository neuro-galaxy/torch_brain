__all__ = [
    "BrainsetPipeline",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
    "submodules": [
        "torch_brain.pipeline.openneuro",
    ],
}

from .pipeline import BrainsetPipeline
