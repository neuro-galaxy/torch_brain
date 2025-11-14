import pytest
import torch
from torch_brain.nn.attention import CrossAttention, SelfAttention


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_length():
    return 16


@pytest.fixture
def dim():
    return 128


@pytest.fixture
def heads():
    return 4


@pytest.fixture
def dim_head():
    return 32


def test_cross_attention_shape(device, batch_size, seq_length, dim, heads):
    model = CrossAttention(dim=dim, heads=heads).to(device)

    # Create sample inputs
    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length * 2, dim).to(device)

    # Test forward pass
    output = model(x_query, x_context)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_self_attention_shape(device, batch_size, seq_length, dim, heads):
    model = SelfAttention(dim=dim, heads=heads).to(device)

    # Create sample inputs
    x = torch.randn(batch_size, seq_length, dim).to(device)

    # Test forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_cross_attention_mask(device, batch_size, seq_length, dim, heads):
    model = CrossAttention(dim=dim, heads=heads).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)

    # Create attention mask (mask out second half of context)
    mask = torch.ones(batch_size, seq_length).to(device)
    mask[:, seq_length // 2 :] = 0

    # Test with mask
    output_masked = model(x_query, x_context, mask)
    assert output_masked.shape == (batch_size, seq_length, dim)


def test_cross_attention_varlen(device, dim, heads):
    model = CrossAttention(dim=dim, heads=heads).to(device)

    # Test with variable sequence lengths
    query_lengths = [3, 5]
    context_lengths = [4, 6]

    total_query_len = sum(query_lengths)
    total_context_len = sum(context_lengths)

    x_query = torch.randn(total_query_len, dim).to(device)
    x_context = torch.randn(total_context_len, dim).to(device)

    # Skip test if not on CUDA with xformers
    if device == "cpu":
        with pytest.raises(NotImplementedError):
            model.forward_varlen(
                x_query,
                x_context,
                torch.tensor(query_lengths),
                torch.tensor(context_lengths),
            )
    else:
        try:
            output = model.forward_varlen(
                x_query,
                x_context,
                torch.tensor(query_lengths),
                torch.tensor(context_lengths),
            )
            assert output.shape == (total_query_len, dim)
        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")


def test_self_attention_varlen(device, dim, heads):
    model = SelfAttention(dim=dim, heads=heads).to(device)

    # Test with variable sequence lengths
    seq_lengths = [3, 5]
    total_len = sum(seq_lengths)

    x = torch.randn(total_len, dim).to(device)

    # Skip test if not on CUDA with xformers
    if device == "cpu":
        with pytest.raises(NotImplementedError):
            model.forward_varlen(x, torch.tensor(seq_lengths))
    else:
        try:
            output = model.forward_varlen(x, torch.tensor(seq_lengths))
            assert output.shape == (total_len, dim)
        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")


def test_invalid_inputs(device, batch_size, seq_length, dim, heads):
    model = CrossAttention(dim=dim, heads=heads).to(device)

    # Test with mismatched batch sizes
    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size + 1, seq_length, dim).to(device)

    with pytest.raises(RuntimeError):
        model(x_query, x_context)


def test_dropout(device, batch_size, seq_length, dim, heads):
    dropout_prob = 0.5
    model = CrossAttention(dim=dim, dropout=dropout_prob, heads=heads).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)

    # Test that outputs are different in training vs eval mode
    model.train()
    out1 = model(x_query, x_context)

    model.eval()
    out2 = model(x_query, x_context)

    # In eval mode, output should be deterministic
    out3 = model(x_query, x_context)

    assert not torch.allclose(out1, out2)
    assert torch.allclose(out2, out3)
