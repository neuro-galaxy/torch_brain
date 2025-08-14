import pytest
import torch
import os
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


@pytest.fixture(autouse=True)
def reset_env_var():
    """Reset the environment variable after each test to avoid interference."""
    yield
    # Reset to default value after each test
    os.environ["TORCH_BRAIN_USE_EFFICIENT_ATTENTION"] = "true"


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("efficient_attention", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_cross_attention_batched_vs_varlen(device, dim, efficient_attention, dropout):
    """Test that batched and varlen cross attention implementations produce the same results."""
    # Set environment variable
    os.environ["TORCH_BRAIN_USE_EFFICIENT_ATTENTION"] = str(efficient_attention).lower()

    model = RotaryCrossAttention(dim=dim, dropout=dropout).to(device)

    # Create test data with variable sequence lengths (multiples of 8)
    query_lengths = [8, 16]
    context_lengths = [16, 24]
    batch_size = len(query_lengths)

    # Create varlen data
    total_query_len = sum(query_lengths)
    total_context_len = sum(context_lengths)

    x_query_varlen = torch.randn(total_query_len, dim, device=device)
    x_context_varlen = torch.randn(total_context_len, dim, device=device)
    query_pos_emb_varlen = torch.randn(total_query_len, dim // 2, device=device)
    context_pos_emb_varlen = torch.randn(total_context_len, dim // 2, device=device)

    # Create batched data from varlen data
    max_query_len = max(query_lengths)
    max_context_len = max(context_lengths)

    x_query_batch = torch.zeros(batch_size, max_query_len, dim, device=device)
    x_context_batch = torch.zeros(batch_size, max_context_len, dim, device=device)
    query_pos_emb_batch = torch.zeros(
        batch_size, max_query_len, dim // 2, device=device
    )
    context_pos_emb_batch = torch.zeros(
        batch_size, max_context_len, dim // 2, device=device
    )
    context_mask = torch.zeros(
        batch_size, max_context_len, device=device, dtype=torch.bool
    )

    # Fill batched tensors
    start_q = 0
    start_c = 0
    for i, (qlen, clen) in enumerate(zip(query_lengths, context_lengths)):
        x_query_batch[i, :qlen] = x_query_varlen[start_q : start_q + qlen]
        x_context_batch[i, :clen] = x_context_varlen[start_c : start_c + clen]
        query_pos_emb_batch[i, :qlen] = query_pos_emb_varlen[start_q : start_q + qlen]
        context_pos_emb_batch[i, :clen] = context_pos_emb_varlen[
            start_c : start_c + clen
        ]
        context_mask[i, :clen] = True
        start_q += qlen
        start_c += clen

    # Run batched forward
    output_batch = model(
        x_query_batch,
        x_context_batch,
        query_pos_emb_batch,
        context_pos_emb_batch,
        context_mask,
    )

    # Run varlen forward
    try:
        output_varlen = model.forward_varlen(
            x_query_varlen,
            x_context_varlen,
            query_pos_emb_varlen,
            context_pos_emb_varlen,
            torch.tensor(query_lengths, device=device),
            torch.tensor(context_lengths, device=device),
        )

        # Compare outputs for non-padded positions
        if dropout == 0.0:
            start_q = 0
            for i, qlen in enumerate(query_lengths):
                assert torch.allclose(
                    output_varlen[start_q : start_q + qlen],
                    output_batch[i, :qlen],
                    atol=1e-4,
                    rtol=1e-4,
                ), f"Varlen and batched outputs differ for batch {i} (efficient={efficient_attention})"
                start_q += qlen

    except (RuntimeError, NotImplementedError) as e:
        if "xformers" in str(e).lower() or "varlen" in str(e).lower():
            pytest.skip("xformers not available for varlen attention")
        else:
            raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("efficient_attention", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_self_attention_batched_vs_varlen(device, dim, efficient_attention, dropout):
    """Test that batched and varlen self attention implementations produce the same results."""
    # Set environment variable
    os.environ["TORCH_BRAIN_USE_EFFICIENT_ATTENTION"] = str(efficient_attention).lower()

    model = RotarySelfAttention(dim=dim, dropout=dropout).to(device)

    # Create test data with variable sequence lengths (multiples of 8)
    seq_lengths = [8, 16]
    batch_size = len(seq_lengths)

    # Create varlen data
    total_len = sum(seq_lengths)
    x_varlen = torch.randn(total_len, dim, device=device)
    pos_emb_varlen = torch.randn(total_len, dim // 2, device=device)

    # Create batched data from varlen data
    max_len = max(seq_lengths)
    x_batch = torch.zeros(batch_size, max_len, dim, device=device)
    pos_emb_batch = torch.zeros(batch_size, max_len, dim // 2, device=device)
    x_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)

    # Fill batched tensors
    start = 0
    for i, slen in enumerate(seq_lengths):
        x_batch[i, :slen] = x_varlen[start : start + slen]
        pos_emb_batch[i, :slen] = pos_emb_varlen[start : start + slen]
        x_mask[i, :slen] = True
        start += slen

    # Run batched forward
    output_batch = model(x_batch, pos_emb_batch, x_mask)

    # Run varlen forward
    try:
        output_varlen = model.forward_varlen(
            x_varlen, pos_emb_varlen, torch.tensor(seq_lengths, device=device)
        )

        # Compare outputs for non-padded positions
        if dropout == 0.0:
            start = 0
            for i, slen in enumerate(seq_lengths):
                assert torch.allclose(
                    output_varlen[start : start + slen],
                    output_batch[i, :slen],
                    atol=1e-4,
                    rtol=1e-4,
                ), f"Varlen and batched outputs differ for batch {i} (efficient={efficient_attention})"
                start += slen

    except (RuntimeError, NotImplementedError) as e:
        if "xformers" in str(e).lower() or "varlen" in str(e).lower():
            pytest.skip("xformers not available for varlen attention")
        else:
            raise
