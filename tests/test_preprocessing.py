import torch
from torch_brain.utils.preprocessing import (
    z_score_normalize,
    mean_center,
    min_max_normalize,
    robust_scale,
)


def test_z_score_normalize():
    x = torch.randn(4, 10)
    y = z_score_normalize(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_mean_center():
    x = torch.randn(4, 10)
    y = mean_center(x)
    assert y.shape == x.shape
    # Mean should be approx zero for each channel
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4), atol=1e-6)


def test_min_max_normalize():
    x = torch.randn(4, 10)
    y = min_max_normalize(x, 0.0, 1.0)
    assert y.shape == x.shape
    # Range should be between 0 and 1
    assert (y >= 0).all() and (y <= 1).all()


def test_robust_scale():
    x = torch.randn(4, 10)
    y = robust_scale(x)
    assert y.shape == x.shape
    # Should not output NaNs
    assert not torch.isnan(y).any()
