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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_xformers", [True, False])
def test_rotary_cross_attention_forward_varlen_matches_forward(
    device, dim, use_xformers
):
    """Test that forward_varlen produces the same output as forward for the same input."""
    model = RotaryCrossAttention(dim=dim).to(device)

    # Create test data
    query_lengths = [8, 16]
    context_lengths = [16, 24]

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

    # Create context mask for padding - shape (B, N_c) indicating valid context positions
    context_mask = torch.zeros(
        batch_size, max_context_len, device=device, dtype=torch.bool
    )

    # Fill in the batched tensors with the same data as the varlen tensors
    start_q = 0
    start_c = 0
    for i, (qlen, clen) in enumerate(zip(query_lengths, context_lengths)):
        x_query_batch[i, :qlen] = x_query_varlen[start_q : start_q + qlen]
        x_context_batch[i, :clen] = x_context_varlen[start_c : start_c + clen]
        query_pos_emb_batch[i, :qlen] = query_pos_emb_varlen[start_q : start_q + qlen]
        context_pos_emb_batch[i, :clen] = context_pos_emb_varlen[
            start_c : start_c + clen
        ]

        # Set context mask for valid positions
        context_mask[i, :clen] = True

        start_q += qlen
        start_c += clen

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
        context_mask=context_mask,
        use_xformers=use_xformers,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_xformers", [True, False])
def test_rotary_self_attention_forward_varlen_matches_forward(
    device, dim, use_xformers
):
    """Test that forward_varlen produces the same output as forward for the same input."""
    model = RotarySelfAttention(dim=dim).to(device)

    # Create test data
    seq_lengths = [8, 16]
    batch_size = len(seq_lengths)

    # First create data for forward_varlen
    total_len = sum(seq_lengths)

    x_varlen = torch.randn(total_len, dim, device=device)
    pos_emb_varlen = torch.randn(total_len, dim // 2, device=device)

    # Now create the same data in batched form for regular forward
    max_len = max(seq_lengths)

    x_batch = torch.zeros(batch_size, max_len, dim, device=device)
    pos_emb_batch = torch.zeros(batch_size, max_len, dim // 2, device=device)

    # Create attention mask for padding - shape (B, N) indicating valid positions
    x_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)

    # Fill in the batched tensors with the same data as the varlen tensors
    start = 0
    for i, slen in enumerate(seq_lengths):
        x_batch[i, :slen] = x_varlen[start : start + slen]
        pos_emb_batch[i, :slen] = pos_emb_varlen[start : start + slen]

        # Set attention mask for valid positions
        x_mask[i, :slen] = True

        start += slen

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
        x_mask=x_mask,
        use_xformers=use_xformers,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_attn_mask", [False, True])
@pytest.mark.parametrize("rotate_value", [False, True])
def test_rotary_attention_functions_equivalence(use_attn_mask, rotate_value):
    """Test that all three rotary attention functions produce identical results."""
    try:
        from torch_brain.nn.rotary_attention import (
            rotary_attn_pytorch_func,
            rotary_attn_xformers_func,
            rotary_attn_xformers_varlen_func,
        )
    except ImportError:
        pytest.skip("rotary attention functions not available")

    try:
        import xformers
    except ImportError:
        pytest.skip("xformers not installed")

    device = "cuda"
    batch_size = 2
    num_heads = 4
    dim_head = 16
    inner_dim = num_heads * dim_head

    if use_attn_mask:
        # Use variable sequence lengths to test proper masking (multiples of 8)
        q_seqlens = [8, 16]
        kv_seqlens = [16, 24]
        max_seq_len_q = max(q_seqlens)
        max_seq_len_kv = max(kv_seqlens)
    else:
        # Use fixed sequence lengths when no masking (multiples of 8)
        q_seqlens = [16, 16]
        kv_seqlens = [24, 24]
        max_seq_len_q = 16
        max_seq_len_kv = 24

    # Create batched data with padding
    query = torch.randn(batch_size, max_seq_len_q, inner_dim, device=device)
    key = torch.randn(batch_size, max_seq_len_kv, inner_dim, device=device)
    value = torch.randn(batch_size, max_seq_len_kv, inner_dim, device=device)
    q_pos_emb = torch.randn(batch_size, max_seq_len_q, dim_head, device=device)
    kv_pos_emb = torch.randn(batch_size, max_seq_len_kv, dim_head, device=device)

    # Create attention mask if needed
    attn_mask = None
    if use_attn_mask:
        attn_mask = torch.zeros(
            batch_size, max_seq_len_kv, device=device, dtype=torch.bool
        )
        for i, kv_len in enumerate(kv_seqlens):
            attn_mask[i, :kv_len] = True

    # PyTorch implementation
    out_pytorch = rotary_attn_pytorch_func(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        q_pos_emb=q_pos_emb.clone(),
        kv_pos_emb=kv_pos_emb.clone(),
        attn_mask=attn_mask.clone() if attn_mask is not None else None,
        num_heads=num_heads,
        dropout_p=0.0,
        rotate_value=rotate_value,
    )

    # XFormers implementation
    out_xformers = rotary_attn_xformers_func(
        query=query.clone(),
        key=key.clone(),
        value=value.clone(),
        q_pos_emb=q_pos_emb.clone(),
        kv_pos_emb=kv_pos_emb.clone(),
        attn_mask=attn_mask.clone() if attn_mask is not None else None,
        num_heads=num_heads,
        dropout_p=0.0,
        rotate_value=rotate_value,
    )

    # XFormers varlen implementation (convert batched data to varlen format)
    # Extract only the non-padded data for varlen format
    total_q_len = sum(q_seqlens)
    total_kv_len = sum(kv_seqlens)

    query_varlen = torch.zeros(total_q_len, inner_dim, device=device)
    key_varlen = torch.zeros(total_kv_len, inner_dim, device=device)
    value_varlen = torch.zeros(total_kv_len, inner_dim, device=device)
    q_pos_emb_varlen = torch.zeros(total_q_len, dim_head, device=device)
    kv_pos_emb_varlen = torch.zeros(total_kv_len, dim_head, device=device)

    # Fill varlen tensors with non-padded data
    start_q = 0
    start_kv = 0
    for i, (q_len, kv_len) in enumerate(zip(q_seqlens, kv_seqlens)):
        query_varlen[start_q : start_q + q_len] = query[i, :q_len]
        key_varlen[start_kv : start_kv + kv_len] = key[i, :kv_len]
        value_varlen[start_kv : start_kv + kv_len] = value[i, :kv_len]
        q_pos_emb_varlen[start_q : start_q + q_len] = q_pos_emb[i, :q_len]
        kv_pos_emb_varlen[start_kv : start_kv + kv_len] = kv_pos_emb[i, :kv_len]
        start_q += q_len
        start_kv += kv_len

    out_xformers_varlen = rotary_attn_xformers_varlen_func(
        query=query_varlen.clone(),
        key=key_varlen.clone(),
        value=value_varlen.clone(),
        q_pos_emb=q_pos_emb_varlen.clone(),
        kv_pos_emb=kv_pos_emb_varlen.clone(),
        q_seqlen=q_seqlens,
        kv_seqlen=kv_seqlens,
        num_heads=num_heads,
        dropout_p=0.0,
        rotate_value=rotate_value,
    )

    # Compare outputs for non-padded positions only
    assert torch.allclose(
        out_pytorch, out_xformers, atol=1e-4, rtol=1e-4
    ), "PyTorch and XFormers outputs differ"

    # Compare varlen output with batched outputs for non-padded positions
    start_q = 0
    for i, q_len in enumerate(q_seqlens):
        # Extract non-padded portion from batched outputs
        pytorch_slice = out_pytorch[i, :q_len]
        xformers_slice = out_xformers[i, :q_len]
        varlen_slice = out_xformers_varlen[start_q : start_q + q_len]

        assert torch.allclose(
            pytorch_slice, varlen_slice, atol=1e-4, rtol=1e-4
        ), f"PyTorch and XFormers varlen outputs differ for batch {i}"

        assert torch.allclose(
            xformers_slice, varlen_slice, atol=1e-4, rtol=1e-4
        ), f"XFormers and XFormers varlen outputs differ for batch {i}"

        start_q += q_len
