"""
Multi-Layer Perceptron model (PyTorch).
"""
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from omegaconf import DictConfig
from .base_model import BaseModel
from . import register_model


class TorchBaseModel(BaseModel):
    """Base class for PyTorch models with common functionality."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.device = self._get_device(cfg)
        self.model = None
        self.random_state = cfg.get("random_state", 42)
        torch.manual_seed(self.random_state)
    
    def _get_device(self, cfg):
        """Get device from config or auto-detect."""
        device_str = cfg.get("device", "auto")
        
        if device_str == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device('cuda')
        elif device_str == "cpu":
            return torch.device('cpu')
        else:
            # Allow specific device strings like 'cuda:0'
            return torch.device(device_str)
    
    @abstractmethod
    def _create_network(self, input_shape, n_classes):
        """Create the neural network architecture."""
        pass
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit is implemented by trainers, but models need this for interface."""
        raise NotImplementedError("PyTorch models should be trained via trainers")
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        self.model.eval()
        all_probs = []
        batch_size = self.cfg.get("batch_size", 64)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def get_device(self):
        """Get the device (CPU or CUDA)."""
        return self.device


@register_model("mlp")
class MLPModel(TorchBaseModel):
    """Multi-Layer Perceptron classifier."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.hidden_dims = cfg.get("hidden_dims", [1024, 1024])
    
    def _create_network(self, input_size, n_classes):
        """Create MLP network."""
        class MLP(nn.Module):
            def __init__(self, input_size, n_classes, hidden_dims):
                super().__init__()
                layers = []
                
                if len(hidden_dims) == 0:
                    # Linear model (logistic regression)
                    layers.append(nn.Linear(input_size, n_classes))
                else:
                    # MLP with hidden layers
                    prev_dim = input_size
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.2))
                        prev_dim = hidden_dim
                    
                    # Output layer
                    layers.append(nn.Linear(prev_dim, n_classes))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                # Flatten all dimensions except batch
                x = x.view(x.size(0), -1)
                return self.network(x)
        
        return MLP(input_size, n_classes, self.hidden_dims)
    
    def build_model(self, input_shape, n_classes):
        """Build the model with given input shape and number of classes."""
        input_size = np.prod(input_shape)
        self.model = self._create_network(input_size, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

