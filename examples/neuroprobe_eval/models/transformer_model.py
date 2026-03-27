"""
Transformer model (PyTorch).
"""
import torch
import torch.nn as nn
import numpy as np
import math
from omegaconf import DictConfig
from .mlp_model import TorchBaseModel
from . import register_model


@register_model("transformer")
class TransformerModel(TorchBaseModel):
    """Transformer classifier."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.d_model = cfg.get("d_model", 64)
        self.nhead = cfg.get("nhead", 8)
        self.dim_feedforward = cfg.get("dim_feedforward", 256)
        self.dropout = cfg.get("dropout", 0.1)
        self.num_layers = cfg.get("num_layers", 3)
    
    def _create_network(self, input_shape, n_classes):
        """Create Transformer network."""
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                return x + self.pe[:, :x.size(1)]
        
        class Transformer(nn.Module):
            def __init__(self, input_shape, n_classes, d_model, nhead, dim_feedforward, dropout, num_layers):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.dim_feedforward = dim_feedforward
                self.dropout = dropout
                self.num_layers = num_layers
                
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    # (channels, time)
                    self.input_proj = nn.Linear(input_shape[0], self.d_model)
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[1])
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = nn.Linear(self.d_model, n_classes)
                else:  # 3D input (channels, freq, time)
                    # (channels, freq, time)
                    self.input_proj = nn.Linear(input_shape[0] * input_shape[1], self.d_model)
                    self.pos_encoder = PositionalEncoding(self.d_model, max_len=input_shape[2])
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=self.nhead,
                        dim_feedforward=self.dim_feedforward,
                        dropout=self.dropout,
                        batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.fc = nn.Linear(self.d_model, n_classes)
            
            def forward(self, x):
                # Reshape input for transformer
                if len(x.shape) == 3:  # (batch, channels, time)
                    x = x.transpose(1, 2)  # (batch, time, channels)
                    x = self.input_proj(x)  # (batch, time, d_model)
                else:  # (batch, channels, freq, time)
                    batch_size, channels, freq, time = x.shape
                    x = x.transpose(1, 3)  # (batch, time, channels, freq)
                    x = x.reshape(batch_size, time, channels * freq)
                    x = self.input_proj(x)  # (batch, time, d_model)
                
                # Add positional encoding
                x = self.pos_encoder(x)
                
                # Apply transformer
                x = self.transformer_encoder(x)
                
                # Global average pooling over time dimension
                x = x.mean(dim=1)
                
                # Final classification layer
                x = self.fc(x)
                return x
        
        return Transformer(
            input_shape, n_classes,
            self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.num_layers
        )
    
    def build_model(self, input_shape, n_classes):
        """Build the model with given input shape and number of classes."""
        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

