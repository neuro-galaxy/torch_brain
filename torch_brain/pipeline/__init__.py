"""Tools for defining pipelines that create brainsets.

**User guide.** See :ref:`brainsets_guide` guide for more information.
"""
__all__ = [
    "BrainsetPipeline",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}

from .pipeline import BrainsetPipeline
