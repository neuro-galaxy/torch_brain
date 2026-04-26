"""
Preprocessor registry for automatic preprocessor discovery and instantiation.
"""
import importlib
import os
from pathlib import Path
from omegaconf import DictConfig

PREPROCESSOR_REGISTRY = {}


def register_preprocessor(name):
    """Decorator to register a preprocessor class."""
    def register_preprocessor_cls(cls):
        if name in PREPROCESSOR_REGISTRY:
            raise ValueError(f'{name} already in registry')
        PREPROCESSOR_REGISTRY[name] = cls
        return cls
    return register_preprocessor_cls


def build_preprocessor(cfg: DictConfig):
    """
    Build a preprocessor from configuration.
    
    Args:
        cfg: Preprocessor configuration (DictConfig from Hydra)
        
    Returns:
        Instantiated preprocessor
    """
    preprocessor_name = cfg.name
    if preprocessor_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f'Preprocessor {preprocessor_name} not found in registry. '
            f'Available: {list(PREPROCESSOR_REGISTRY.keys())}'
        )
    
    preprocessor_class = PREPROCESSOR_REGISTRY[preprocessor_name]
    preprocessor = preprocessor_class(cfg)
    return preprocessor


def import_preprocessors():
    """Auto-import all preprocessor files to register them."""
    preprocessors_dir = os.path.dirname(__file__)
    for file in os.listdir(preprocessors_dir):
        if file.endswith(".py") and not file.startswith("_") and file != "base_preprocessor.py":
            module_name = str(Path(file).with_suffix(""))
            # Try relative import first, then absolute
            try:
                importlib.import_module(f'.{module_name}', package='preprocessors')
            except (ImportError, ValueError):
                # Fallback: construct absolute import path
                parent_dir = os.path.basename(os.path.dirname(preprocessors_dir))
                importlib.import_module(f'{parent_dir}.preprocessors.{module_name}')


# Import all preprocessors to register them
import_preprocessors()

