import pytest
import torch

from torch_brain.nn.loss import MSELoss, CrossEntropyLoss, MallowDistanceLoss


def test_mse_loss():
    loss_fn = MSELoss()
    input = torch.randn(10, 5)
    target = torch.randn(10, 5)

    # Test without weights
    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Test with weights
    weights = torch.ones(10)
    loss = loss_fn(input, target, weights)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_cross_entropy_loss():
    loss_fn = CrossEntropyLoss()
    input = torch.randn(10, 5)  # logits
    target = torch.randint(0, 5, (10,))  # class indices

    print(input.shape, target.shape)

    # Test without weights
    loss = loss_fn(input, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Test with weights
    weights = torch.ones(10)
    loss = loss_fn(input, target, weights)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_mallow_distance_loss():
    loss_fn = MallowDistanceLoss()
    num_classes = 5
    batch_size = 10

    input = torch.randn(batch_size, num_classes)  # logits
    target = torch.randint(0, num_classes, (batch_size,))  # class indices
    weights = torch.ones(batch_size)

    loss = loss_fn(input, target, weights)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Test that loss is non-negative
    assert loss >= 0

    # Test perfect prediction case
    perfect_input = torch.zeros(batch_size, num_classes)
    perfect_input.scatter_(1, target.unsqueeze(1), 1.0)
    perfect_input = torch.log(perfect_input)  # convert to logits
    perfect_loss = loss_fn(perfect_input, target, weights)
    assert torch.allclose(perfect_loss, torch.tensor(0.0), atol=1e-6)
