"""
Model registry for automatic model discovery and instantiation.
"""
import importlib
import os
from pathlib import Path
from omegaconf import DictConfig

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model class."""
    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f'{name} already in registry')
        MODEL_REGISTRY[name] = cls
        return cls
    return register_model_cls


def build_model(cfg: DictConfig):
    """
    Build a model from configuration.
    
    Args:
        cfg: Model configuration (DictConfig from Hydra)
        
    Returns:
        Instantiated model
    """
    model_name = cfg.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Model {model_name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}')
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(cfg)
    return model


def import_models():
    """Auto-import all model files to register them."""
    models_dir = os.path.dirname(__file__)
    for file in os.listdir(models_dir):
        if file.endswith(".py") and not file.startswith("_") and file != "base_model.py":
            module_name = str(Path(file).with_suffix(""))
            # Try relative import first, then absolute
            try:
                importlib.import_module(f'.{module_name}', package='models')
            except (ImportError, ValueError):
                # Fallback: construct absolute import path
                parent_dir = os.path.basename(os.path.dirname(models_dir))
                importlib.import_module(f'{parent_dir}.models.{module_name}')


# Import all models to register them
import_models()

