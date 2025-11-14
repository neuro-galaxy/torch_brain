"""
Tests for LoRA (Low-Rank Adaptation) functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_brain.nn import (
    FeedForward,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
)
from torch_brain_private.nn import (
    LoRALayer,
    LoRALinear,
    LoRALinearCombined,
    LoRAModelWrapper,
)
from torch_brain_private.nn.lora import _deduce_projection_names_from_module_name


class SimpleAttentionModel(nn.Module):
    """Simple model with encoding, processing, and decoding attention layers for testing LoRA."""

    def __init__(
        self,
        dim=64,
        dim_input=64,
        dim_head=16,
        cross_heads=1,
        self_heads=2,
        depth=2,
        ffn_dropout=0.2,
        lin_dropout=0.4,
        atn_dropout=0.0,
        t_min=1e-4,
        t_max=4.0,
    ):
        super().__init__()

        self.dim = dim
        self.dim_input = dim_input

        # Rotary time embedding
        self.rotary_emb = RotaryTimeEmbedding(dim_head, dim_head, t_min, t_max)

        # Encoding cross-attention layer (latents attend to inputs)
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=dim_input,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Processing self-attention layers (2 layers)
        self.proc_layers = nn.ModuleList([])
        for _ in range(depth):
            self.proc_layers.append(
                nn.Sequential(
                    RotarySelfAttention(
                        dim=dim,
                        heads=self_heads,
                        dropout=atn_dropout,
                        dim_head=dim_head,
                        rotate_value=True,
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        FeedForward(dim=dim, dropout=ffn_dropout),
                    ),
                )
            )

        # Decoding cross-attention layer (outputs attend to latents)
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )

        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # Simple output projection
        self.readout = nn.Linear(dim, 2)  # Output 2 classes for simplicity

    def forward(
        self,
        inputs,
        latents,
        outputs,
        input_timestamps,
        latent_timestamps,
        output_timestamps,
        input_mask=None,
    ):
        """
        Args:
            inputs: (batch, n_in, dim_input)
            latents: (batch, n_latent, dim)
            outputs: (batch, n_out, dim)
            input_timestamps: (batch, n_in)
            latent_timestamps: (batch, n_latent)
            output_timestamps: (batch, n_out)
            input_mask: (batch, n_in) optional

        Returns:
            logits: (batch, n_out, 2)
        """
        # Get rotary embeddings
        input_timestamp_emb = self.rotary_emb(input_timestamps)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # Encoding: latents attend to inputs
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # Processing: self-attention on latents
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # Decoding: outputs attend to latents
        outputs = outputs + self.dec_atn(
            outputs,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
        )
        outputs = outputs + self.dec_ffn(outputs)

        # Output projection
        logits = self.readout(outputs)

        return logits


class TestLoRALayer:
    """Test the base LoRALayer class."""

    def test_lora_layer_initialization(self):
        """Test LoRA layer initialization."""
        lora = LoRALayer(
            in_features=128,
            out_features=64,
            rank=16,
            alpha=16.0,
            dropout=0.05,
            init_scale=0.01,
        )

        assert lora.dropout.p == 0.05
        assert lora.rank == 16
        assert lora.alpha == 16.0
        assert lora.scaling == 1.0  # alpha / rank
        assert lora.lora_A.shape == (128, 16)
        assert lora.lora_B.shape == (16, 64)
        assert lora.enabled.item() == True

        # A should be initialized with small random values
        assert torch.std(lora.lora_A) > 0

        # B should be initialized with zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_lora_scaling_factor(self):
        """Test that LoRA scaling factor is applied correctly."""
        lora = LoRALayer(
            in_features=128, out_features=64, rank=8, alpha=16.0, init_scale=0.01
        )
        x = torch.randn(2, 10, 128)

        # Manual computation
        manual_output = x @ lora.lora_A @ lora.lora_B * lora.scaling

        # LoRA output
        lora_output = lora(x)

        assert torch.allclose(manual_output, lora_output)
        assert lora.scaling == 2.0  # alpha / rank = 16.0 / 8

    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass."""
        lora = LoRALayer(in_features=128, out_features=64, rank=16, alpha=16.0)
        x = torch.randn(2, 10, 128)

        output = lora(x)
        assert output.shape == (2, 10, 64)

    def test_lora_gradient_flow(self):
        """Test that gradients flow correctly through LoRA layers."""
        lora = LoRALayer(
            in_features=128,
            out_features=64,
            rank=16,
            alpha=16.0,
            dropout=0.05,
            init_scale=0.01,
        )
        x = torch.randn(2, 10, 128, requires_grad=True)

        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert x.grad is not None

    def test_lora_layer_enable_disable(self):
        """Test enabling and disabling LoRA layer with actual functionality verification."""
        # Create a simple model with LoRA
        base_layer = nn.Linear(128, 64)
        lora = LoRALayer(in_features=128, out_features=64, rank=16, alpha=16.0)

        # Create input and target
        x = torch.randn(2, 10, 128, requires_grad=True)
        target = torch.randn(2, 10, 64)

        # Test enabled LoRA - should contribute to gradients
        lora.enable()
        base_output = base_layer(x)
        lora_output = lora(x)
        combined_output = base_output + lora_output

        # Compute loss and backward pass
        loss_enabled = F.mse_loss(combined_output, target)
        loss_enabled.backward()

        # Check that LoRA parameters received gradients when enabled
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        # Note: lora_A may have zero gradients initially because lora_B starts at zero
        # but lora_B should definitely have non-zero gradients
        assert not torch.allclose(lora.lora_B.grad, torch.zeros_like(lora.lora_B.grad))

        # Clear gradients
        lora.zero_grad()
        base_layer.zero_grad()
        x.grad = None

        # Test disabled LoRA - should not contribute to gradients
        lora.disable()
        base_output = base_layer(x)
        lora_output = lora(x)
        combined_output = base_output + lora_output

        # Compute loss and backward pass
        loss_disabled = F.mse_loss(combined_output, target)
        loss_disabled.backward()

        # Check that LoRA parameters did NOT receive gradients when disabled
        # When disabled, LoRA outputs zeros so there's no gradient flow
        # The gradients should be None because the computation graph doesn't include LoRA
        assert lora.lora_A.grad is None
        assert lora.lora_B.grad is None

        # Test that enable/disable actually changes the enabled state
        assert lora.enabled.item() == False
        lora.enable()
        assert lora.enabled.item() == True

    def test_lora_layer_validation(self):
        """Test LoRA layer input validation."""
        with pytest.raises(ValueError, match="LoRA rank must be positive"):
            LoRALayer(in_features=128, out_features=64, rank=0)

        with pytest.raises(ValueError, match="LoRA alpha must be positive"):
            LoRALayer(in_features=128, out_features=64, rank=16, alpha=0)


class TestLoRALinear:
    """Test the LoRALinear class."""

    def test_lora_linear_initialization(self):
        """Test LoRA linear layer initialization."""
        linear = nn.Linear(128, 64)
        lora_linear = LoRALinear(linear, rank=16, alpha=16.0)

        assert lora_linear.linear is linear
        assert lora_linear.lora.rank == 16
        assert lora_linear.lora.alpha == 16.0

    def test_lora_linear_forward(self):
        """Test LoRA linear layer forward pass."""
        linear = nn.Linear(128, 64)
        lora_linear = LoRALinear(linear, rank=16, alpha=16.0)
        x = torch.randn(2, 10, 128)

        output = lora_linear(x)
        assert output.shape == (2, 10, 64)

        # Output should be original + LoRA
        original_output = linear(x)
        lora_output = lora_linear.lora(x)
        expected_output = original_output + lora_output
        assert torch.allclose(output, expected_output)

    def test_lora_linear_enable_disable(self):
        """Test LoRA linear layer enable/disable."""
        linear = nn.Linear(128, 64)
        lora_linear = LoRALinear(linear, rank=16, alpha=16.0)
        x = torch.randn(2, 10, 128)

        # Test enabled
        lora_linear.enable_lora()
        output_enabled = lora_linear(x)

        # Test disabled
        lora_linear.disable_lora()
        output_disabled = lora_linear(x)

        # Disabled output should equal original output
        original_output = linear(x)
        assert torch.allclose(output_disabled, original_output)

    def test_lora_linear_freeze_unfreeze(self):
        """Test freezing and unfreezing original parameters, including after forward and backward pass."""
        linear = nn.Linear(128, 64)
        lora_linear = LoRALinear(linear, rank=16, alpha=16.0)
        x = torch.randn(2, 10, 128)
        target = torch.randn(2, 10, 64)

        # Test freeze
        lora_linear.freeze_original()
        for param in linear.parameters():
            assert not param.requires_grad

        # Forward and backward pass while frozen
        output = lora_linear(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        for param in linear.parameters():
            assert not param.requires_grad

        # Test unfreeze
        lora_linear.unfreeze_original()
        for param in linear.parameters():
            assert param.requires_grad

        # Forward and backward pass after unfreezing
        linear.zero_grad()
        lora_linear.zero_grad()
        output = lora_linear(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        for param in linear.parameters():
            assert param.requires_grad


class TestLoRALinearCombined:
    """Test the LoRALinearCombined class."""

    def test_lora_linear_combined_initialization(self):
        """Test LoRA linear combined layer initialization."""
        # Create a combined linear layer (like to_qkv)
        linear = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined = LoRALinearCombined(
            linear,
            name="to_qkv",
            target_projections=["q", "k", "v"],
            rank=16,
            alpha=16.0,
        )

        # Check that projection names are correctly deduced
        assert lora_linear_combined.projection_names == ["q", "k", "v"]

        # Check that LoRA projections are created for all target projections
        assert "q" in lora_linear_combined.lora_projections
        assert "k" in lora_linear_combined.lora_projections
        assert "v" in lora_linear_combined.lora_projections

        # Check that each LoRA projection has correct rank and alpha
        for proj_name in ["q", "k", "v"]:
            lora_proj = lora_linear_combined.lora_projections[proj_name]
            assert isinstance(lora_proj, LoRALinear)
            assert lora_proj.lora.rank == 16
            assert lora_proj.lora.alpha == 16.0

    def test_lora_linear_combined_partial_targets(self):
        """Test LoRA linear combined with partial target projections."""
        # Create a combined linear layer (like to_qkv)
        linear = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined = LoRALinearCombined(
            linear, name="to_qkv", target_projections=["q", "v"], rank=16, alpha=16.0
        )

        # Check that only q and v have LoRA
        assert "q" in lora_linear_combined.lora_projections
        assert "v" in lora_linear_combined.lora_projections
        assert "k" not in lora_linear_combined.lora_projections

        # Check that k is in regular projections
        assert "k" in lora_linear_combined.projections

    def test_lora_linear_combined_forward(self):
        """Test LoRA linear combined layer forward pass."""
        # Create a combined linear layer (like to_qkv)
        linear = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined = LoRALinearCombined(
            linear,
            name="to_qkv",
            target_projections=["q", "k", "v"],
            rank=16,
            alpha=16.0,
        )
        x = torch.randn(2, 10, 128)

        output = lora_linear_combined(x)
        assert output.shape == (2, 10, 192)

        # Check that output is composed of q, k, v projections
        # Each should have dimension 64
        q_output = output[..., :64]
        k_output = output[..., 64:128]
        v_output = output[..., 128:]

        assert q_output.shape == (2, 10, 64)
        assert k_output.shape == (2, 10, 64)
        assert v_output.shape == (2, 10, 64)

    def test_lora_linear_combined_enable_disable(self):
        """Test LoRA linear combined layer enable/disable."""
        # Create a combined linear layer (like to_qkv)
        linear = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined = LoRALinearCombined(
            linear,
            name="to_qkv",
            target_projections=["q", "k", "v"],
            rank=16,
            alpha=16.0,
        )
        x = torch.randn(2, 10, 128)
        target = torch.randn(2, 10, 192)

        # Train LoRA to ensure it has non-zero effect
        lora_linear_combined.enable_lora()
        for _ in range(5):
            output = lora_linear_combined(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            # Update LoRA parameters
            with torch.no_grad():
                for lora_proj in lora_linear_combined.lora_projections.values():
                    lora_proj.lora.lora_A -= 0.1 * lora_proj.lora.lora_A.grad
                    lora_proj.lora.lora_B -= 0.1 * lora_proj.lora.lora_B.grad
                    lora_proj.lora.lora_A.grad.zero_()
                    lora_proj.lora.lora_B.grad.zero_()

        # Test enabled
        lora_linear_combined.enable_lora()
        output_enabled = lora_linear_combined(x)

        # Test disabled
        lora_linear_combined.disable_lora()
        output_disabled = lora_linear_combined(x)

        # Outputs should be different when LoRA is enabled vs disabled
        assert not torch.allclose(output_enabled, output_disabled)

        # When disabled, each projection should match the original linear layer output
        # for that projection's slice
        for i, proj_name in enumerate(["q", "k", "v"]):
            lora_proj = lora_linear_combined.lora_projections[proj_name]
            start_idx = i * 64
            end_idx = (i + 1) * 64
            disabled_slice = output_disabled[..., start_idx:end_idx]
            original_slice = lora_proj.linear(x)
            assert torch.allclose(disabled_slice, original_slice)

    def test_lora_linear_combined_freeze_unfreeze(self):
        """Test freezing and unfreezing original parameters, including after forward and backward pass."""
        # Create a combined linear layer (like to_qkv)
        linear = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined = LoRALinearCombined(
            linear,
            name="to_qkv",
            target_projections=["q", "k", "v"],
            rank=16,
            alpha=16.0,
        )
        x = torch.randn(2, 10, 128)
        target = torch.randn(2, 10, 192)

        # Test freeze
        lora_linear_combined.freeze_original()
        for lora_proj in lora_linear_combined.lora_projections.values():
            for param in lora_proj.linear.parameters():
                assert not param.requires_grad

        # Forward and backward pass while frozen
        output = lora_linear_combined(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        for lora_proj in lora_linear_combined.lora_projections.values():
            for param in lora_proj.linear.parameters():
                assert not param.requires_grad

        # Test unfreeze
        lora_linear_combined.unfreeze_original()
        for lora_proj in lora_linear_combined.lora_projections.values():
            for param in lora_proj.linear.parameters():
                assert param.requires_grad

        # Forward and backward pass after unfreezing
        for lora_proj in lora_linear_combined.lora_projections.values():
            lora_proj.linear.zero_grad()
            lora_proj.lora.zero_grad()
        output = lora_linear_combined(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        for lora_proj in lora_linear_combined.lora_projections.values():
            for param in lora_proj.linear.parameters():
                assert param.requires_grad

    def test_lora_linear_combined_output_dimensions_match_original(self):
        """Test that LoRALinearCombined forward output has same dimensions as original linear layer."""
        # Test with to_qkv (3 projections)
        linear_qkv = nn.Linear(128, 192)  # 192 = 64 * 3 for q, k, v
        lora_linear_combined_qkv = LoRALinearCombined(
            linear_qkv,
            name="to_qkv",
            target_projections=["q", "k", "v"],
            rank=16,
            alpha=16.0,
        )
        x_qkv = torch.randn(2, 10, 128)

        # Get outputs
        original_output_qkv = linear_qkv(x_qkv)
        lora_output_qkv = lora_linear_combined_qkv(x_qkv)

        # Check dimensions match
        assert (
            original_output_qkv.shape == lora_output_qkv.shape
        ), f"Output shapes don't match: original {original_output_qkv.shape} vs LoRA {lora_output_qkv.shape}"
        assert lora_output_qkv.shape == (2, 10, 192)

        # Test with to_kv (2 projections)
        linear_kv = nn.Linear(128, 128)  # 128 = 64 * 2 for k, v
        lora_linear_combined_kv = LoRALinearCombined(
            linear_kv, name="to_kv", target_projections=["k", "v"], rank=16, alpha=16.0
        )
        x_kv = torch.randn(2, 10, 128)

        # Get outputs
        original_output_kv = linear_kv(x_kv)
        lora_output_kv = lora_linear_combined_kv(x_kv)

        # Check dimensions match
        assert (
            original_output_kv.shape == lora_output_kv.shape
        ), f"Output shapes don't match: original {original_output_kv.shape} vs LoRA {lora_output_kv.shape}"
        assert lora_output_kv.shape == (2, 10, 128)

        # Test with partial projections (only q and v)
        linear_partial = nn.Linear(128, 192)
        lora_linear_combined_partial = LoRALinearCombined(
            linear_partial,
            name="to_qkv",
            target_projections=["q", "v"],
            rank=16,
            alpha=16.0,
        )
        x_partial = torch.randn(2, 10, 128)

        # Get outputs
        original_output_partial = linear_partial(x_partial)
        lora_output_partial = lora_linear_combined_partial(x_partial)

        # Check dimensions match
        assert (
            original_output_partial.shape == lora_output_partial.shape
        ), f"Output shapes don't match: original {original_output_partial.shape} vs LoRA {lora_output_partial.shape}"
        assert lora_output_partial.shape == (2, 10, 192)

        # Test with different batch sizes and sequence lengths
        for batch_size in [1, 4, 8]:
            for seq_len in [5, 20]:
                x_test = torch.randn(batch_size, seq_len, 128)
                original_output_test = linear_qkv(x_test)
                lora_output_test = lora_linear_combined_qkv(x_test)
                assert original_output_test.shape == lora_output_test.shape

    def test_lora_linear_combined_validation(self):
        """Test LoRA linear combined validation."""
        # Test that target_projections must be in projection_names
        linear = nn.Linear(128, 192)
        with pytest.raises(ValueError, match="At least one target_projection"):
            LoRALinearCombined(
                linear,
                name="to_qkv",
                target_projections=["x", "y"],
                rank=16,
                alpha=16.0,
            )


class TestLoRAModelWrapper:
    """Test the LoRAModelWrapper class."""

    def create_test_model(self):
        """Create a SimpleAttentionModel for testing."""
        model = SimpleAttentionModel(
            dim=64,
            dim_input=64,
            dim_head=16,
            cross_heads=1,
            self_heads=2,
            atn_dropout=0.0,
        )
        return model

    @pytest.mark.parametrize(
        ("target_modules", "target_projections", "expected_lora_modules"),
        [
            # Test specific linear layers
            (
                [
                    "enc_atn.to_q",
                    "enc_atn.to_kv",
                    "proc_layers.0.0.to_qkv",
                    "dec_atn.to_q",
                    "dec_atn.to_kv",
                ],
                ["q", "k", "v"],
                [
                    "enc_atn.to_q",
                    "enc_atn.to_kv",
                    "proc_layers.0.0.to_qkv",
                    "dec_atn.to_q",
                    "dec_atn.to_kv",
                ],
            ),
            # Test attention modules with linear layers and q, k, v projections
            (
                ["enc_atn", "proc_layers", "dec_atn"],
                ["q", "k", "v"],
                [
                    "enc_atn.to_q",
                    "enc_atn.to_kv",
                    "proc_layers.0.0.to_qkv",
                    "proc_layers.1.0.to_qkv",
                    "dec_atn.to_q",
                    "dec_atn.to_kv",
                ],
            ),
            # Test attention modules with linear layers and only q projections
            (
                ["enc_atn", "proc_layers", "dec_atn"],
                ["q"],
                [
                    "enc_atn.to_q",
                    "proc_layers.0.0.to_qkv",
                    "proc_layers.1.0.to_qkv",
                    "dec_atn.to_q",
                ],
            ),
            # Test mix of feedforward and attention modules with linear layers and only
            # out and net projections
            (
                ["enc_atn", "enc_ffn", "proc_layers", "dec_atn", "dec_ffn"],
                ["out", "net"],
                [
                    "enc_atn.to_out",
                    "enc_ffn.1.net.0",
                    "enc_ffn.1.net.3",
                    "proc_layers.0.0.to_out",
                    "proc_layers.0.1.1.net.0",
                    "proc_layers.0.1.1.net.3",
                    "proc_layers.1.0.to_out",
                    "proc_layers.1.1.1.net.0",
                    "proc_layers.1.1.1.net.3",
                    "dec_atn.to_out",
                    "dec_ffn.1.net.0",
                    "dec_ffn.1.net.3",
                ],
            ),
        ],
    )
    def test_lora_model_wrapper_initialization(
        self, target_modules, target_projections, expected_lora_modules
    ):
        """Test LoRA model wrapper initialization."""
        model = self.create_test_model()

        for name, module in model.named_modules():
            print(f"Module {name} is a {type(module).__name__}")

        lora_model = LoRAModelWrapper(
            model=model,
            target_modules=target_modules,
            rank=16,
            alpha=16.0,
            target_projections=target_projections,
        )

        assert lora_model.base_model is model
        assert len(lora_model.lora_modules) == len(expected_lora_modules)
        for expected_lora_module in expected_lora_modules:
            assert expected_lora_module in list(
                lora_model.lora_modules.keys()
            ), f"Expected {expected_lora_module} in {list(lora_model.lora_modules.keys())}"

        # Check that LoRA modules are correctly applied based on target_projections
        for module_name, lora_module in lora_model.lora_modules.items():
            projection_names = _deduce_projection_names_from_module_name(module_name)
            if len(projection_names) == 1:
                assert isinstance(lora_module, LoRALinear)
                assert projection_names[0] in target_projections
            elif len(projection_names) > 1:
                assert isinstance(lora_module, LoRALinearCombined)
                assert all(
                    proj in target_projections
                    for proj in lora_module.lora_projections.keys()
                )

    def test_lora_model_wrapper_forward(self):
        """Test LoRA model wrapper forward pass."""
        model = self.create_test_model()
        lora_model = LoRAModelWrapper(
            model=model,
            target_modules=["enc_atn", "proc_layers", "dec_atn"],
            target_projections=["q", "k", "v"],
        )

        lora_model.enable_lora()

        # Create test inputs that match SimpleAttentionModel's expected format
        batch_size = 2
        n_in = 16
        n_latent = 8
        n_out = 4

        inputs = torch.randn(batch_size, n_in, model.dim_input)
        latents = torch.randn(batch_size, n_latent, model.dim)
        outputs = torch.randn(batch_size, n_out, model.dim)
        input_timestamps = torch.randn(batch_size, n_in)
        latent_timestamps = torch.randn(batch_size, n_latent)
        output_timestamps = torch.randn(batch_size, n_out)
        input_mask = torch.ones(batch_size, n_in, dtype=torch.bool)

        # Create target values
        target = torch.randint(0, 2, (batch_size, n_out))

        # Test that the model can run forward pass
        logits = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )

        # Check that outputs are returned with correct shapes
        assert logits is not None
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (batch_size, n_out, 2)

        # Test that we can compute gradients through the model
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        # Check that LoRA parameters received gradients
        lora_params_with_grad = 0
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                if lora_module.lora.lora_A.grad is not None:
                    lora_params_with_grad += 1
                if lora_module.lora.lora_B.grad is not None:
                    lora_params_with_grad += 1
            elif isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    if lora_layer.lora.lora_A.grad is not None:
                        lora_params_with_grad += 1
                    if lora_layer.lora.lora_B.grad is not None:
                        lora_params_with_grad += 1

        # Estimate total number of LoRA parameters
        total_lora_A_params = 0
        total_lora_B_params = 0
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                if lora_module.lora.lora_A is not None:
                    total_lora_A_params += 1
                if lora_module.lora.lora_B is not None:
                    total_lora_B_params += 1
            elif isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    if lora_layer.lora.lora_A is not None:
                        total_lora_A_params += 1
                    if lora_layer.lora.lora_B is not None:
                        total_lora_B_params += 1

        # All LoRA parameters should have gradients
        assert (
            lora_params_with_grad == total_lora_A_params + total_lora_B_params
        ), f"Expected {total_lora_A_params + total_lora_B_params} LoRA parameters with gradients, got {lora_params_with_grad}"

    def test_lora_model_wrapper_enable_disable(self):
        """Test LoRA model wrapper enable/disable."""
        model = self.create_test_model()
        lora_model = LoRAModelWrapper(
            model=model,
            target_modules=["enc_atn", "proc_layers", "dec_atn"],
            target_projections=["k", "v"],
        )

        # Create test inputs that match SimpleAttentionModel's expected format
        batch_size = 2
        n_in = 16
        n_latent = 8
        n_out = 4

        inputs = torch.randn(batch_size, n_in, model.dim_input)
        latents = torch.randn(batch_size, n_latent, model.dim)
        outputs = torch.randn(batch_size, n_out, model.dim)
        input_timestamps = torch.randn(batch_size, n_in)
        latent_timestamps = torch.randn(batch_size, n_latent)
        output_timestamps = torch.randn(batch_size, n_out)
        input_mask = torch.ones(batch_size, n_in, dtype=torch.bool)

        # Train LoRA parameters a bit so they have non-zero effect
        lora_model.enable_lora()
        target = torch.randint(0, 2, (batch_size, n_out))
        for _ in range(5):
            logits = lora_model(
                inputs=inputs,
                latents=latents,
                outputs=outputs,
                input_timestamps=input_timestamps,
                latent_timestamps=latent_timestamps,
                output_timestamps=output_timestamps,
                input_mask=input_mask,
            )
            loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
            loss.backward()
            # Simple gradient descent step
            with torch.no_grad():
                for lora_module in lora_model.lora_modules.values():
                    if isinstance(lora_module, LoRALinear):
                        if lora_module.lora.lora_A.grad is not None:
                            lora_module.lora.lora_A -= (
                                0.1 * lora_module.lora.lora_A.grad
                            )
                            lora_module.lora.lora_A.grad.zero_()
                        if lora_module.lora.lora_B.grad is not None:
                            lora_module.lora.lora_B -= (
                                0.1 * lora_module.lora.lora_B.grad
                            )
                            lora_module.lora.lora_B.grad.zero_()
                    elif isinstance(lora_module, LoRALinearCombined):
                        for lora_layer in lora_module.lora_projections.values():
                            if lora_layer.lora.lora_A.grad is not None:
                                lora_layer.lora.lora_A -= (
                                    0.1 * lora_layer.lora.lora_A.grad
                                )
                                lora_layer.lora.lora_A.grad.zero_()
                            if lora_layer.lora.lora_B.grad is not None:
                                lora_layer.lora.lora_B -= (
                                    0.1 * lora_layer.lora.lora_B.grad
                                )
                                lora_layer.lora.lora_B.grad.zero_()

        # Test that the model can run forward pass with LoRA enabled
        lora_model.enable_lora()
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                assert lora_module.lora.enabled.item() == True
            elif isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    assert lora_layer.lora.enabled.item() == True

        logits_enabled = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )

        # Test that the model can run forward pass
        lora_model.disable_lora()
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                assert lora_module.lora.enabled.item() == False
            elif isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    assert lora_layer.lora.enabled.item() == False

        logits_disabled = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )

        # Check that outputs with LoRA enabled and disabled are different
        assert not torch.allclose(logits_enabled, logits_disabled)

        # Check that LoRA layers return zeros when disabled
        lora_model.enable_lora()
        for name, lora_module in lora_model.lora_modules.items():
            if isinstance(lora_module, LoRALinear):
                # Get a sample input for this layer
                sample_input = torch.randn(batch_size, lora_module.linear.in_features)

                # Test with LoRA disabled - should return zeros from LoRA part
                lora_module.disable_lora()
                output_disabled = lora_module(sample_input)

                # The LoRA contribution should be zero when disabled
                original_output = lora_module.linear(sample_input)
                lora_contribution = output_disabled - original_output
                assert torch.allclose(
                    lora_contribution, torch.zeros_like(lora_contribution), atol=1e-6
                ), f"LoRA layer {name} should return zero contribution when disabled"

            elif isinstance(lora_module, LoRALinearCombined):
                for name, lora_layer in lora_module.lora_projections.items():
                    # Get a sample input for this layer
                    sample_input = torch.randn(
                        batch_size, lora_layer.linear.in_features
                    )

                    # Test with LoRA disabled - should return zeros from LoRA part
                    lora_layer.disable_lora()
                    output_disabled = lora_layer(sample_input)

                    # The LoRA contribution should be zero when disabled
                    original_output = lora_layer.linear(sample_input)
                    lora_contribution = output_disabled - original_output
                    assert torch.allclose(
                        lora_contribution,
                        torch.zeros_like(lora_contribution),
                        atol=1e-6,
                    ), f"LoRA layer {name} should return zero contribution when disabled"

    def test_lora_model_wrapper_freeze_unfreeze_original(self):
        """Test freezing and unfreezing original parameters."""
        model = self.create_test_model()
        lora_model = LoRAModelWrapper(
            model=model,
            target_modules=["enc_atn", "proc_layers", "dec_atn"],
            target_projections=["q", "k", "v"],
        )

        # Create test inputs that match SimpleAttentionModel's expected format
        batch_size = 2
        n_in = 16
        n_latent = 8
        n_out = 4

        inputs = torch.randn(batch_size, n_in, model.dim_input)
        latents = torch.randn(batch_size, n_latent, model.dim)
        outputs = torch.randn(batch_size, n_out, model.dim)
        input_timestamps = torch.randn(batch_size, n_in)
        latent_timestamps = torch.randn(batch_size, n_latent)
        output_timestamps = torch.randn(batch_size, n_out)
        input_mask = torch.ones(batch_size, n_in, dtype=torch.bool)

        lora_model.enable_lora()

        # Test freezing of original parameters before forward pass
        lora_model.freeze_lora_original()
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                for param in lora_module.linear.parameters():
                    assert not param.requires_grad
                for param in lora_module.lora.parameters():
                    assert param.requires_grad
            if isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    for param in lora_layer.linear.parameters():
                        assert not param.requires_grad
                    for param in lora_layer.lora.parameters():
                        assert param.requires_grad
                for layer in lora_module.projections.values():
                    for param in layer.parameters():
                        assert param.requires_grad

        # Test freezing of original parameters after forward/backward pass
        target = torch.randint(0, 2, (batch_size, n_out))

        logits = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        # Check that linear layers in LoRALInear require_grad is False while LoRA parameters require_grad is True
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                for param in lora_module.linear.parameters():
                    assert param.grad is None
                for param in lora_module.lora.parameters():
                    assert param.grad is not None
            if isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    for param in lora_layer.linear.parameters():
                        assert param.grad is None
                    for param in lora_layer.lora.parameters():
                        assert param.grad is not None
                for layer in lora_module.projections.values():
                    for param in layer.parameters():
                        assert param.grad is not None

        # Test unfreezing of original parameters after backward pass
        lora_model.zero_grad()
        lora_model.unfreeze_lora_original()
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                for param in lora_module.linear.parameters():
                    assert param.requires_grad
                for param in lora_module.lora.parameters():
                    assert param.requires_grad
            if isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    for param in lora_layer.linear.parameters():
                        assert param.requires_grad
                    for param in lora_layer.lora.parameters():
                        assert param.requires_grad
                for layer in lora_module.projections.values():
                    for param in layer.parameters():
                        assert param.requires_grad

        # Test unfreezing of original parameters after forward/backward pass
        target = torch.randint(0, 2, (batch_size, n_out))

        logits = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        # Check that linear layers in LoRALInear require_grad is True while LoRA parameters require_grad is True
        for lora_module in lora_model.lora_modules.values():
            if isinstance(lora_module, LoRALinear):
                for param in lora_module.linear.parameters():
                    assert param.grad is not None
                for param in lora_module.lora.parameters():
                    assert param.grad is not None
            if isinstance(lora_module, LoRALinearCombined):
                for lora_layer in lora_module.lora_projections.values():
                    for param in lora_layer.linear.parameters():
                        assert param.grad is not None
                    for param in lora_layer.lora.parameters():
                        assert param.grad is not None
                for layer in lora_module.projections.values():
                    for param in layer.parameters():
                        assert param.grad is not None

    def test_lora_model_wrapper_freeze_unfreeze_base_model(self):
        """Test freezing and unfreezing base model parameters (excluding LoRA)."""
        model = self.create_test_model()
        lora_model = LoRAModelWrapper(
            model=model,
            target_modules=["enc_atn", "proc_layers", "dec_atn"],
            target_projections=["q", "k", "v"],
        )

        # Create test inputs that match SimpleAttentionModel's expected format
        batch_size = 2
        n_in = 16
        n_latent = 8
        n_out = 4

        inputs = torch.randn(batch_size, n_in, model.dim_input)
        latents = torch.randn(batch_size, n_latent, model.dim)
        outputs = torch.randn(batch_size, n_out, model.dim)
        input_timestamps = torch.randn(batch_size, n_in)
        latent_timestamps = torch.randn(batch_size, n_latent)
        output_timestamps = torch.randn(batch_size, n_out)
        input_mask = torch.ones(batch_size, n_in, dtype=torch.bool)

        lora_model.enable_lora()

        params_requiring_grad_before_freeze = sum(
            1 for p in lora_model.base_model.parameters() if p.requires_grad
        )

        # Test freezing of base model parameters before forward pass
        lora_model.freeze_base_model()
        for name, param in lora_model.base_model.named_parameters():
            if ("lora" not in name.lower() and "readout" not in name.lower()) or (
                "lora" in name.lower() and "linear" in name.lower()
            ):
                assert not param.requires_grad, f"Parameter {name} should be frozen"
            else:
                assert param.requires_grad, f"Parameter {name} should not be frozen"

        # Test freezing of base model parameters after forward/backward pass
        target = torch.randint(0, 2, (batch_size, n_out))

        logits = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        # Check that frozen parameters have no gradients
        for name, param in lora_model.base_model.named_parameters():
            if ("lora" not in name.lower() and "readout" not in name.lower()) or (
                "lora" in name.lower() and "linear" in name.lower()
            ):
                assert param.grad is None, f"Parameter {name} should have no gradient"
            else:
                assert param.grad is not None, f"Parameter {name} should have gradient"

        # Test unfreezing of base model parameters after backward pass
        lora_model.zero_grad()
        lora_model.unfreeze_base_model()
        for name, param in lora_model.base_model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be unfrozen"

        params_requiring_grad_after_unfreeze = sum(
            1 for p in lora_model.base_model.parameters() if p.requires_grad
        )

        # After unfreezing, all parameters should require grad again
        assert (
            params_requiring_grad_after_unfreeze == params_requiring_grad_before_freeze
        )

        # Test unfreezing of base model parameters after forward/backward pass
        target = torch.randint(0, 2, (batch_size, n_out))

        logits = lora_model(
            inputs=inputs,
            latents=latents,
            outputs=outputs,
            input_timestamps=input_timestamps,
            latent_timestamps=latent_timestamps,
            output_timestamps=output_timestamps,
            input_mask=input_mask,
        )
        loss = F.cross_entropy(logits.view(-1, 2), target.view(-1))
        loss.backward()

        # Check that all parameters with requires_grad now have gradients
        for name, param in lora_model.base_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradient"


if __name__ == "__main__":
    pytest.main([__file__])
