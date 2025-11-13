"""
Low-Rank Adaptation (LoRA) implementation for torch_brain_private.

This module provides flexible LoRA layers that can be applied to various neural network
components, enabling efficient fine-tuning with reduced parameter count.

References:
    - Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
    - https://arxiv.org/abs/2106.09685
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torchtyping import TensorType


class LoRALayer(nn.Module):
    r"""Base LoRA layer that can be applied to any linear transformation.

    LoRA decomposes the weight update ΔW into two low-rank matrices A and B:
    ΔW = α/r * A @ B, where A ∈ R^(d×r), B ∈ R^(r×k), and r << min(d,k)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        r"""
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA scaling factor (α)
            dropout: Dropout rate for LoRA layers
            init_scale: Initialization scale for matrix A
        """
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * init_scale)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Register as buffer to avoid being included in state_dict when disabled
        self.register_buffer("enabled", torch.tensor(True))

    def forward(
        self, x: TensorType["batch", "seq_len", "in_features"]
    ) -> TensorType["batch", "seq_len", "out_features"]:
        """Forward pass through LoRA layer.

        Args:
            x: Input tensor

        Returns:
            LoRA output: α/r * x @ A @ B
        """
        if not self.enabled:
            return torch.zeros_like(x[..., : self.lora_B.shape[1]])

        # Apply LoRA: x @ A @ B * scaling
        result = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return result

    def enable(self):
        """
        Enable the LoRA layer.

        This method sets the internal 'enabled' buffer to True, which allows the LoRA
        layer to contribute its output during the forward pass. When enabled, the LoRA
        adaptation is active and its output is added to the main layer's output.
        """
        self.enabled.fill_(True)

    def disable(self):
        """
        Disable the LoRA layer.

        This method sets the internal 'enabled' buffer to False, which causes the LoRA
        layer to output zeros during the forward pass. When disabled, the LoRA adaptation
        has no effect on the model's output.
        """
        self.enabled.fill_(False)

    def get_effective_rank(self) -> int:
        """Get the effective rank of the LoRA layer."""
        return self.rank if self.enabled else 0


class LoRALinear(nn.Module):
    r"""Linear layer with LoRA adaptation.

    This wraps a standard linear layer and adds LoRA adaptation capability.
    The forward pass computes: original_output + lora_output
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        r"""
        Args:
            linear_layer: The original linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA layers
            init_scale: Initialization scale for matrix A
        """
        super().__init__()

        self.linear = linear_layer
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            init_scale=init_scale,
        )

    def forward(
        self, x: TensorType["batch", "seq_len", "in_features"]
    ) -> TensorType["batch", "seq_len", "out_features"]:
        """Forward pass: original + LoRA."""
        original_output = self.linear(x)
        lora_output = self.lora(x)
        return original_output + lora_output

    def enable_lora(self):
        """Enable LoRA adaptation."""
        self.lora.enable()

    def disable_lora(self):
        """Disable LoRA adaptation."""
        self.lora.disable()

    def freeze_original(self):
        """Freeze the original linear layer parameters."""
        for param in self.linear.parameters():
            param.requires_grad = False

    def unfreeze_original(self):
        """Unfreeze the original linear layer parameters."""
        for param in self.linear.parameters():
            param.requires_grad = True


class LoRALinearCombined(nn.Module):
    r"""LoRA adaptation for combined linear layers with selective projection targeting.

    This class can handle any combined linear layer (like to_qkv, to_kv, etc.) and apply
    LoRA selectively to specific projections within the combined layer.

    The projection names are automatically deduced from the provided name:
    - Names ending with 'to_qkv' -> ['q', 'k', 'v']
    - Names ending with 'to_kv' -> ['k', 'v']
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        name: str,
        target_projections: Optional[List[str]] = None,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        r"""
        Args:
            linear_layer: The original linear layer
            name: The name of the linear layer
            target_projections: List of projections to apply LoRA to ('q', 'k', 'v')
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA layers
            init_scale: Initialization scale for LoRA matrix A
        """
        super().__init__()

        # Always deduce projection_names from name
        projection_names = _deduce_projection_names_from_module_name(name)

        # Set default target_projections to all projections if not specified
        if target_projections is None:
            target_projections = projection_names.copy()

        if not any(proj in projection_names for proj in target_projections):
            raise ValueError(
                f"At least one target_projection {target_projections} must be in projection_names {projection_names}"
            )

        self.projection_names = projection_names
        num_projections = len(projection_names)

        # Get original layer parameters
        original_weight = linear_layer.weight.data
        original_bias = linear_layer.bias

        # Calculate dimensions
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        if out_features % num_projections != 0:
            raise ValueError(
                f"Output features ({out_features}) must be divisible by number of projections ({self.num_projections})"
            )

        projection_dim = out_features // num_projections

        # Create separate projections
        self.projections = nn.ModuleDict()
        self.lora_projections = nn.ModuleDict()
        for i, proj_name in enumerate(self.projection_names):
            # Create individual projection
            proj = nn.Linear(
                in_features, projection_dim, bias=original_bias is not None
            )

            # Initialize weights from original combined projection
            start_idx = i * projection_dim
            end_idx = (i + 1) * projection_dim
            proj.weight.data = original_weight[start_idx:end_idx, :]

            if original_bias is not None:
                proj.bias.data = original_bias[start_idx:end_idx]

            # if in target_projections, apply LoRA
            if proj_name in target_projections:
                self.lora_projections[proj_name] = LoRALinear(
                    proj, rank, alpha, dropout, init_scale
                )
            else:
                self.projections[proj_name] = proj

    def forward(self, x):
        """Forward pass that combines all projections with LoRA applied where specified."""
        outputs = []

        for proj_name in self.projection_names:
            if proj_name in self.lora_projections:
                # Use LoRA version
                output = self.lora_projections[proj_name](x)
            else:
                # Use original projection
                output = self.projections[proj_name](x)
            outputs.append(output)

        # Concatenate to match original combined layer output format
        return torch.cat(outputs, dim=-1)

    def enable_lora(self):
        """Enable LoRA for all targeted projections."""
        for lora_proj in self.lora_projections.values():
            lora_proj.enable_lora()

    def disable_lora(self):
        """Disable LoRA for all targeted projections."""
        for lora_proj in self.lora_projections.values():
            lora_proj.disable_lora()

    def freeze_original(self):
        """Freeze original projection parameters."""
        for lora_proj in self.lora_projections.values():
            lora_proj.freeze_original()

    def unfreeze_original(self):
        """Unfreeze original projection parameters."""
        for lora_proj in self.lora_projections.values():
            lora_proj.unfreeze_original()


class LoRAModelWrapper(nn.Module):
    r"""Wrapper to apply LoRA to an entire model.

    This class provides utilities to apply LoRA to specific layers in a model
    and manage training of LoRA parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        target_projections: List[str] = ["q", "k", "v"],
    ):
        r"""
        Args:
            model: The base model to adapt
            target_modules: List of module names to apply LoRA to
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA layers
            init_scale: Initialization scale for LoRA matrix A
            target_projections: List of projections to apply LoRA to in attention layers or linear layers
        """
        super().__init__()

        self.base_model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.init_scale = init_scale
        self.target_projections = target_projections

        # Store original modules and their LoRA adaptations
        self.lora_modules: Dict[
            str,
            Union[
                LoRALinear,
                LoRALinearCombined,
            ],
        ] = {}

        # Apply LoRA to target modules
        self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to target modules."""
        for name, module in self.base_model.named_modules():
            logging.debug(f"\nModule {name} is a {type(module).__name__}")
            if any(target in name for target in self.target_modules) and isinstance(
                module, nn.Linear
            ):
                logging.debug(f"\nModule {name} identified as target")
                projection_names = _deduce_projection_names_from_module_name(name)

                # if the module is a linear layer and has one projection
                # and it is in the target projections, apply LoRALinear
                if len(projection_names) == 1 and any(
                    proj in self.target_projections for proj in projection_names
                ):
                    logging.debug(
                        f"Applying LoRALinear to {name} ({type(module).__name__}) with projections {projection_names}"
                    )
                    lora_module = LoRALinear(
                        module,
                        self.rank,
                        self.alpha,
                        self.dropout,
                        self.init_scale,
                    )
                    self.lora_modules[name] = lora_module
                    self._replace_module(name, lora_module)

                # if the module is a linear layer and has multiple projections
                # and any of the projections are in the target projections,
                # apply LoRALinearCombined
                elif len(projection_names) > 1 and any(
                    proj in self.target_projections for proj in projection_names
                ):
                    logging.debug(
                        f"Applying LoRALinearCombined to {name} ({type(module).__name__}) with projections {projection_names}"
                    )
                    lora_module = LoRALinearCombined(
                        module,
                        name,
                        self.target_projections,
                        self.rank,
                        self.alpha,
                        self.dropout,
                        self.init_scale,
                    )
                    self.lora_modules[name] = lora_module
                    self._replace_module(name, lora_module)

                # if the module is a linear layer and has no projections
                # and the name contains 'net', apply LoRALinear
                elif len(projection_names) == 0 and (
                    "net" in name and "net" in self.target_projections
                ):
                    logging.debug(
                        f"Applying LoRALinear to {name} ({type(module).__name__}) with no projections"
                    )
                    lora_module = LoRALinear(
                        module,
                        self.rank,
                        self.alpha,
                        self.dropout,
                        self.init_scale,
                    )
                    self.lora_modules[name] = lora_module
                    self._replace_module(name, lora_module)

    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the model by name."""
        parts = module_name.split(".")
        parent = self.base_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.base_model(*args, **kwargs)

    def enable_lora(self):
        """Enable LoRA for all adapted modules."""
        for module in self.lora_modules.values():
            if hasattr(module, "enable_lora"):
                module.enable_lora()
            elif hasattr(module, "enable"):
                module.enable()

    def disable_lora(self):
        """Disable LoRA for all adapted modules."""
        for module in self.lora_modules.values():
            if hasattr(module, "disable_lora"):
                module.disable_lora()
            elif hasattr(module, "disable"):
                module.disable()

    def freeze_base_model(self, freeze_lora_original: bool = True):
        """Freeze all base model parameters except LoRA, readout layers, and embeddings."""
        for name, param in self.base_model.named_parameters():
            if (
                "lora" not in name.lower()
                and "readout" not in name.lower()
                and "unit_emb" not in name.lower()
                and "session_emb" not in name.lower()
                and "modality_emb" not in name.lower()
                and "task_emb" not in name.lower()
            ):
                param.requires_grad = False

        if freeze_lora_original:
            self.freeze_lora_original()

    def unfreeze_base_model(self, unfreeze_lora_original: bool = True):
        """Unfreeze all base model parameters except LoRA, readout layers, and embeddings."""
        for name, param in self.base_model.named_parameters():
            if (
                "lora" not in name.lower()
                and "readout" not in name.lower()
                and "unit_emb" not in name.lower()
                and "session_emb" not in name.lower()
                and "modality_emb" not in name.lower()
                and "task_emb" not in name.lower()
            ):
                param.requires_grad = True

        if unfreeze_lora_original:
            self.unfreeze_lora_original()

    def freeze_lora_original(self):
        """Freeze original parameters in LoRA modules."""
        for module in self.lora_modules.values():
            if hasattr(module, "freeze_original"):
                module.freeze_original()

    def unfreeze_lora_original(self):
        """Unfreeze original parameters in LoRA modules."""
        for module in self.lora_modules.values():
            if hasattr(module, "unfreeze_original"):
                module.unfreeze_original()

    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Get all LoRA parameters."""
        lora_params = []
        for module in self.lora_modules.values():
            if hasattr(module, "lora"):
                # LoRALinear
                lora_params.extend([module.lora.lora_A, module.lora.lora_B])
            elif hasattr(module, "lora_projections"):
                # LoRALinearCombined
                for lora_proj in module.lora_projections.values():
                    lora_params.extend([lora_proj.lora.lora_A, lora_proj.lora.lora_B])
        return lora_params

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters (LoRA + unfrozen base model)."""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for different components."""
        total_params = sum(p.numel() for p in self.parameters())
        lora_params = sum(p.numel() for p in self.get_lora_parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())

        return {
            "total": total_params,
            "lora": lora_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
        }

    def print_parameter_summary(self):
        """Print a summary of parameter counts."""
        counts = self.get_parameter_count()
        print(f"Parameter Summary:")
        print(f"  Total parameters: {counts['total']:,}")
        print(f"  LoRA parameters: {counts['lora']:,}")
        print(f"  Trainable parameters: {counts['trainable']:,}")
        print(f"  Frozen parameters: {counts['frozen']:,}")
        print(f"  LoRA % of total: {100 * counts['lora'] / counts['total']:.2f}%")
        print(
            f"  Trainable % of total: {100 * counts['trainable'] / counts['total']:.2f}%"
        )

    def __getattr__(self, name: str):
        """Get an attribute from the model."""

        # If the attribute is base_model, return the base_model
        if name == "base_model":
            return object.__getattribute__(self, "__dict__")["_modules"]["base_model"]

        # Otherwise delegate to base_model
        try:
            base_model = object.__getattribute__(self, "__dict__")["_modules"][
                "base_model"
            ]
            return getattr(base_model, name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


def _deduce_projection_names_from_module_name(name: str) -> List[str]:
    """Deduce projection names from layer name.

    Args:
        name: The layer name (e.g., 'to_qkv', 'to_kv')

    Returns:
        List of projection names

    Raises:
        ValueError: If the name doesn't match known patterns
    """
    name_suffix = name.split(".")[-1]
    if name_suffix.startswith("to_"):
        projection = name_suffix.split("_")[1]
        if name_suffix.endswith("out"):
            return [projection]
        else:
            return list(projection)
    else:
        return []


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    init_scale: float = 0.01,
    target_projections: List[str] = ["q", "k", "v"],
) -> LoRAModelWrapper:
    """Convenience function to apply LoRA to a model.

    Args:
        model: The base model to adapt
        target_modules: List of module names to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA layers
        init_scale: Initialization scale for LoRA matrix A
        target_projections: List of projections to apply LoRA to in attention layers

    Returns:
        LoRA-wrapped model
    """
    return LoRAModelWrapper(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        init_scale=init_scale,
        target_projections=target_projections,
    )