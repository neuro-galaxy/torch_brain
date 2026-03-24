import pytest
import torch
from torch_brain.nn.rotary_attention import RotaryCrossAttention, RotarySelfAttention


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


def test_rotary_cross_attention_shape(device, batch_size, seq_length, dim):
    model = RotaryCrossAttention(dim=dim).to(device)

    # Create sample inputs
    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length * 2, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length * 2, dim).to(device)

    # Test forward pass
    output = model(x_query, x_context, query_pos_emb, context_pos_emb)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_rotary_self_attention_shape(device, batch_size, seq_length, dim):
    model = RotarySelfAttention(dim=dim).to(device)

    # Create sample inputs
    x = torch.randn(batch_size, seq_length, dim).to(device)
    pos_emb = torch.randn(batch_size, seq_length, dim).to(device)

    # Test forward pass
    output = model(x, pos_emb)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_rotary_cross_attention_mask(device, batch_size, seq_length, dim):
    model = RotaryCrossAttention(dim=dim).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)

    # Create attention mask (mask out second half of context)
    mask = torch.ones(batch_size, seq_length).to(device)
    mask[:, seq_length // 2 :] = 0

    # Test with mask
    output_masked = model(x_query, x_context, query_pos_emb, context_pos_emb, mask)
    assert output_masked.shape == (batch_size, seq_length, dim)


def test_rotary_cross_attention_varlen(device, dim):
    model = RotaryCrossAttention(dim=dim).to(device)

    # Test with variable sequence lengths
    query_lengths = [3, 5]
    context_lengths = [4, 6]

    total_query_len = sum(query_lengths)
    total_context_len = sum(context_lengths)

    x_query = torch.randn(total_query_len, dim).to(device)
    x_context = torch.randn(total_context_len, dim).to(device)
    query_pos_emb = torch.randn(total_query_len, dim).to(device)
    context_pos_emb = torch.randn(total_context_len, dim).to(device)

    # Skip test if not on CUDA with xformers
    if device == "cpu":
        with pytest.raises(NotImplementedError):
            model.forward_varlen(
                x_query,
                x_context,
                query_pos_emb,
                context_pos_emb,
                torch.tensor(query_lengths),
                torch.tensor(context_lengths),
            )
    else:
        try:
            output = model.forward_varlen(
                x_query,
                x_context,
                query_pos_emb,
                context_pos_emb,
                torch.tensor(query_lengths),
                torch.tensor(context_lengths),
            )
            assert output.shape == (total_query_len, dim)
        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")


def test_rotary_self_attention_varlen(device, dim):
    model = RotarySelfAttention(dim=dim).to(device)

    # Test with variable sequence lengths
    seq_lengths = [3, 5]
    total_len = sum(seq_lengths)

    x = torch.randn(total_len, dim).to(device)
    pos_emb = torch.randn(total_len, dim).to(device)

    # Skip test if not on CUDA with xformers
    if device == "cpu":
        with pytest.raises(NotImplementedError):
            model.forward_varlen(x, pos_emb, torch.tensor(seq_lengths))
    else:
        try:
            output = model.forward_varlen(x, pos_emb, torch.tensor(seq_lengths))
            assert output.shape == (total_len, dim)
        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")


def test_use_xformers_false(device, batch_size, dim):
    # Use a sequence length that's not a multiple of 8, so xformers would error
    # if it were accidentally called — proving that use_xformers=False routes to
    # the PyTorch SDPA path.
    seq_length = 7

    # Cross attention: forward() should use PyTorch SDPA
    cross = RotaryCrossAttention(dim=dim, use_xformers=False).to(device)
    x_query = torch.randn(batch_size, seq_length, dim, device=device)
    x_context = torch.randn(batch_size, seq_length, dim, device=device)
    q_pos = torch.randn(batch_size, seq_length, dim, device=device)
    c_pos = torch.randn(batch_size, seq_length, dim, device=device)

    out = cross(x_query, x_context, q_pos, c_pos)
    assert out.shape == (batch_size, seq_length, dim)

    # Cross attention: forward_varlen() should raise
    total = seq_length * batch_size
    with pytest.raises((NotImplementedError, RuntimeError)):
        cross.forward_varlen(
            torch.randn(total, dim, device=device),
            torch.randn(total, dim, device=device),
            torch.randn(total, dim, device=device),
            torch.randn(total, dim, device=device),
            torch.tensor([seq_length] * batch_size),
            torch.tensor([seq_length] * batch_size),
        )

    # Self attention: forward() should use PyTorch SDPA
    self_attn = RotarySelfAttention(dim=dim, use_xformers=False).to(device)
    x = torch.randn(batch_size, seq_length, dim, device=device)
    pos = torch.randn(batch_size, seq_length, dim, device=device)

    out = self_attn(x, pos)
    assert out.shape == (batch_size, seq_length, dim)

    # Self attention: forward_varlen() should raise
    with pytest.raises((NotImplementedError, RuntimeError)):
        self_attn.forward_varlen(
            torch.randn(total, dim, device=device),
            torch.randn(total, dim, device=device),
            torch.tensor([seq_length] * batch_size),
        )


def test_invalid_inputs(device, batch_size, seq_length, dim):
    model = RotaryCrossAttention(dim=dim).to(device)

    # Test with mismatched batch sizes
    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size + 1, seq_length, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)

    with pytest.raises(RuntimeError):
        model(x_query, x_context, query_pos_emb, context_pos_emb)


def test_dropout(device, batch_size, seq_length, dim):
    dropout_prob = 0.5
    model = RotaryCrossAttention(dim=dim, dropout=dropout_prob).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim).to(device)

    # Test that outputs are different in training vs eval mode
    model.train()
    out1 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    model.eval()
    out2 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    # In eval mode, output should be deterministic
    out3 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    assert not torch.allclose(out1, out2)
    assert torch.allclose(out2, out3)
