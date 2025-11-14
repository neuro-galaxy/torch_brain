"""
Convolutional Neural Network model (PyTorch).
"""
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from .mlp_model import TorchBaseModel
from . import register_model


@register_model("cnn")
class CNNModel(TorchBaseModel):
    """Convolutional Neural Network classifier."""
    
    def _create_network(self, input_shape, n_classes):
        """Create CNN network."""
        class CNN(nn.Module):
            def __init__(self, input_shape, n_classes):
                super().__init__()
                # Assuming input shape is (channels, time) or (channels, freq, time)
                if len(input_shape) == 2:
                    # 1D CNN for (channels, time)
                    self.conv1 = nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool1d(2)
                    self.dropout = nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = input_shape[1] // 8 * 128
                    
                    self.fc1 = nn.Linear(conv_output_size, 256)
                    self.fc2 = nn.Linear(256, n_classes)
                    
                else:  # 3D input (channels, freq, time)
                    # 2D CNN for (channels, freq, time)
                    self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool2d(2)
                    self.dropout = nn.Dropout(0.5)
                    
                    # Calculate the size after convolutions and pooling
                    conv_output_size = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
                    
                    self.fc1 = nn.Linear(conv_output_size, 256)
                    self.fc2 = nn.Linear(256, n_classes)
                
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.dropout(x)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return CNN(input_shape, n_classes)
    
    def build_model(self, input_shape, n_classes):
        """Build the model with given input shape and number of classes."""
        self.model = self._create_network(input_shape, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model

