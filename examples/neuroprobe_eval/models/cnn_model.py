"""
Convolutional Neural Network model (PyTorch).
"""

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from .mlp_model import TorchBaseModel
from . import register_model
from .model_utils import to_int_list


def prepare_cnn_batch(batch, **kwargs):
    """Prepare collated batch for CNN-style input convention."""
    _ = kwargs
    if not isinstance(batch, dict):
        raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
    if "x" not in batch:
        raise KeyError("Batch must include key 'x'.")

    x = batch["x"]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float()
    if x.ndim == 4:
        # Keep parity with legacy reshape_data_for_model for CNN-style models.
        x = x.permute(0, 1, 3, 2).contiguous()

    out = dict(batch)
    out["x"] = x
    return out


@register_model("cnn")
class CNNModel(TorchBaseModel):
    """Convolutional Neural Network classifier."""

    def prepare_batch(self, batch, **kwargs):
        """Prepare collated batch for CNN input convention."""
        return prepare_cnn_batch(batch, **kwargs)

    def _create_network(self, input_shape, n_classes, hidden_dims=None):
        """Create CNN network.

        Args:
            input_shape: Input shape tuple
            n_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions for FC layers after conv layers.
                        Defaults to [256] if None.
        """
        if hidden_dims is None:
            hidden_dims = [256]
        elif isinstance(hidden_dims, (int, float)):
            hidden_dims = [int(hidden_dims)]
        elif not isinstance(hidden_dims, (list, tuple)):
            hidden_dims = [256]

        class CNN(nn.Module):
            def __init__(self, input_shape, n_classes, hidden_dims):
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

                else:  # 3D input (channels, freq, time)
                    # 2D CNN for (channels, freq, time)
                    self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                    self.pool = nn.MaxPool2d(2)
                    self.dropout = nn.Dropout(0.5)

                    # Calculate the size after convolutions and pooling
                    conv_output_size = (
                        (input_shape[1] // 8) * (input_shape[2] // 8) * 128
                    )

                # Build FC layers from hidden_dims
                fc_layers = []
                prev_dim = conv_output_size
                for hidden_dim in hidden_dims:
                    fc_layers.append(nn.Linear(prev_dim, hidden_dim))
                    fc_layers.append(nn.ReLU())
                    fc_layers.append(nn.Dropout(0.5))
                    prev_dim = hidden_dim

                # Output layer
                fc_layers.append(nn.Linear(prev_dim, n_classes))
                self.fc = nn.Sequential(*fc_layers)

                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return CNN(input_shape, n_classes, hidden_dims)

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)
        hidden_dims = to_int_list(
            self.cfg.get("hidden_dims", [256]),
            [256],
            allow_empty=True,
        )
        # Store hidden_dims as attribute for logging
        self.hidden_dims = hidden_dims
        self.model = self._create_network(input_shape, n_classes, hidden_dims)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model
