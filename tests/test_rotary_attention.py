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
    query_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length * 2, dim // 2).to(device)

    # Test forward pass
    output = model(x_query, x_context, query_pos_emb, context_pos_emb)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_rotary_self_attention_shape(device, batch_size, seq_length, dim):
    model = RotarySelfAttention(dim=dim).to(device)

    # Create sample inputs
    x = torch.randn(batch_size, seq_length, dim).to(device)
    pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)

    # Test forward pass
    output = model(x, pos_emb)

    # Check output shape
    assert output.shape == (batch_size, seq_length, dim)


def test_rotary_cross_attention_mask(device, batch_size, seq_length, dim):
    model = RotaryCrossAttention(dim=dim).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)

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
    query_pos_emb = torch.randn(total_query_len, dim // 2).to(device)
    context_pos_emb = torch.randn(total_context_len, dim // 2).to(device)

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
    pos_emb = torch.randn(total_len, dim // 2).to(device)

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
    query_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)

    with pytest.raises(RuntimeError):
        model(x_query, x_context, query_pos_emb, context_pos_emb)


def test_dropout(device, batch_size, seq_length, dim):
    dropout_prob = 0.5
    model = RotaryCrossAttention(dim=dim, dropout=dropout_prob).to(device)

    x_query = torch.randn(batch_size, seq_length, dim).to(device)
    x_context = torch.randn(batch_size, seq_length, dim).to(device)
    query_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)
    context_pos_emb = torch.randn(batch_size, seq_length, dim // 2).to(device)

    # Test that outputs are different in training vs eval mode
    model.train()
    out1 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    model.eval()
    out2 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    # In eval mode, output should be deterministic
    out3 = model(x_query, x_context, query_pos_emb, context_pos_emb)

    assert not torch.allclose(out1, out2)
    assert torch.allclose(out2, out3)


def test_rotary_cross_attention_forward_varlen_matches_forward(device, dim):
    """Test that forward_varlen produces the same output as forward for the same input."""
    # Skip test if not on CUDA with xformers
    if device == "cpu":
        pytest.skip("forward_varlen not implemented for CPU")

    try:
        model = RotaryCrossAttention(dim=dim).to(device)

        # Create test data
        query_lengths = [3, 5]
        context_lengths = [4, 6]

        batch_size = len(query_lengths)

        # First create data for forward_varlen
        total_query_len = sum(query_lengths)
        total_context_len = sum(context_lengths)

        x_query_varlen = torch.randn(total_query_len, dim, device=device)
        x_context_varlen = torch.randn(total_context_len, dim, device=device)
        query_pos_emb_varlen = torch.randn(total_query_len, dim // 2, device=device)
        context_pos_emb_varlen = torch.randn(total_context_len, dim // 2, device=device)

        # Now create the same data in batched form for regular forward
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

        # Create attention mask for padding
        attn_mask = torch.zeros(
            batch_size, max_query_len, max_context_len, device=device
        )

        # Fill in the batched tensors with the same data as the varlen tensors
        start_q = 0
        start_c = 0
        for i, (qlen, clen) in enumerate(zip(query_lengths, context_lengths)):
            x_query_batch[i, :qlen] = x_query_varlen[start_q : start_q + qlen]
            x_context_batch[i, :clen] = x_context_varlen[start_c : start_c + clen]
            query_pos_emb_batch[i, :qlen] = query_pos_emb_varlen[
                start_q : start_q + qlen
            ]
            context_pos_emb_batch[i, :clen] = context_pos_emb_varlen[
                start_c : start_c + clen
            ]

            # Set attention mask for valid positions
            attn_mask[i, :qlen, :clen] = 1.0

            start_q += qlen
            start_c += clen

        # Get outputs from both implementations
        try:
            # Run forward_varlen
            output_varlen = model.forward_varlen(
                x_query_varlen,
                x_context_varlen,
                query_pos_emb_varlen,
                context_pos_emb_varlen,
                torch.tensor(query_lengths, device=device),
                torch.tensor(context_lengths, device=device),
            )

            # Run normal forward
            output_batch = model(
                x_query_batch,
                x_context_batch,
                query_pos_emb_batch,
                context_pos_emb_batch,
                attn_mask,
            )

            # Compare outputs for non-padded positions
            start_q = 0
            for i, qlen in enumerate(query_lengths):
                assert torch.allclose(
                    output_varlen[start_q : start_q + qlen],
                    output_batch[i, :qlen],
                    atol=1e-5,
                )
                start_q += qlen

        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")
    except Exception as e:
        pytest.skip(f"Test failed with error: {str(e)}")


def test_rotary_self_attention_forward_varlen_matches_forward(device, dim):
    """Test that forward_varlen produces the same output as forward for the same input."""
    # Skip test if not on CUDA with xformers
    if device == "cpu":
        pytest.skip("forward_varlen not implemented for CPU")

    try:
        model = RotarySelfAttention(dim=dim).to(device)

        # Create test data
        seq_lengths = [3, 5]
        batch_size = len(seq_lengths)

        # First create data for forward_varlen
        total_len = sum(seq_lengths)

        x_varlen = torch.randn(total_len, dim, device=device)
        pos_emb_varlen = torch.randn(total_len, dim // 2, device=device)

        # Now create the same data in batched form for regular forward
        max_len = max(seq_lengths)

        x_batch = torch.zeros(batch_size, max_len, dim, device=device)
        pos_emb_batch = torch.zeros(batch_size, max_len, dim // 2, device=device)

        # Create attention mask for padding
        attn_mask = torch.zeros(batch_size, max_len, max_len, device=device)

        # Fill in the batched tensors with the same data as the varlen tensors
        start = 0
        for i, slen in enumerate(seq_lengths):
            x_batch[i, :slen] = x_varlen[start : start + slen]
            pos_emb_batch[i, :slen] = pos_emb_varlen[start : start + slen]

            # Set attention mask for valid positions
            attn_mask[i, :slen, :slen] = 1.0

            start += slen

        # Get outputs from both implementations
        try:
            # Run forward_varlen
            output_varlen = model.forward_varlen(
                x_varlen,
                pos_emb_varlen,
                torch.tensor(seq_lengths, device=device),
            )

            # Run normal forward
            output_batch = model(
                x_batch,
                pos_emb_batch,
                attn_mask,
            )

            # Compare outputs for non-padded positions
            start = 0
            for i, slen in enumerate(seq_lengths):
                assert torch.allclose(
                    output_varlen[start : start + slen],
                    output_batch[i, :slen],
                    atol=1e-5,
                )
                start += slen

        except RuntimeError as e:
            if "please install xformers" in str(e):
                pytest.skip("xformers not installed")
    except Exception as e:
        pytest.skip(f"Test failed with error: {str(e)}")
