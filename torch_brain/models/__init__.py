from .poyo import POYO, poyo_mp
from .poyo_plus import POYOPlus
from .capoyo import CaPOYO

__all__ = ["POYO", "POYOPlus", "CaPOYO", "poyo_mp"]

# see docs/source/api_reference.py
__api_ref__ = {
    "description": None,
    "sections": [
        {
            "autosummary": __all__,
        }
    ],
}
