import pytest
import torch

import torch_brain
from torch_brain.models.shallownet import ShallowNet

@pytest.mark.parametrize("logsoftmax", [True, False])
def test_forward_shape(logsoftmax):
    B, C, T, K = 2, 22, 1000, 4
    m = ShallowNet(in_chans=C, in_times=T, n_classes=K, logsoftmax=logsoftmax).eval()
    x = torch.randn(B, C, T)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (B, K)


def test_final_conv_length_analytic():
    C, T = 22, 1000
    m = ShallowNet(in_chans=C, in_times=T, n_classes=4, filter_time_length=25,
                   pool_time_length=75, pool_time_stride=15, logsoftmax=False).eval()
    # Manual calc:
    L1 = T - 25 + 1           # conv_time
    L3 = (L1 - 75) // 15 + 1  # pool
    assert m.final_conv_length == L3 == 61  # with these defaults


def test_logsoftmax_flag_numerical_equivalence():
    # With identical weights, the log-probabilities must match
    B, C, T, K = 2, 22, 1000, 4
    # build a logits model
    m_lg = ShallowNet(C, T, K, logsoftmax=False).eval()
    # build a log-prob model and copy weights/buffers
    m_lp = ShallowNet(C, T, K, logsoftmax=True).eval()
    m_lp.load_state_dict(m_lg.state_dict(), strict=True)
    x = torch.randn(B, C, T)
    with torch.no_grad():
        y_lg_lp = m_lg(x).log_softmax(dim=1)  # logits → log-probs
        y_lp    = m_lp(x)                     # log-probs
    torch.testing.assert_close(y_lp, y_lg_lp, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("logsoftmax,criterion_cls", [
    (False, torch.nn.CrossEntropyLoss),  # logits → CE
    (True,  torch.nn.NLLLoss),           # log-probs → NLL
])
def test_backward_and_optim_step(logsoftmax, criterion_cls):
    B, C, T, K = 8, 22, 1000, 4
    m = ShallowNet(C, T, K, logsoftmax=logsoftmax)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = criterion_cls()
    x = torch.randn(B, C, T)
    y = torch.randint(0, K, (B,))
    out = m(x)
    # NLL expects log-probs; CE expects logits — paired via parametrize above
    loss = crit(out, y)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
    opt.step()
    # basic sanity
    assert torch.isfinite(loss)
    assert float(grad_norm) > 0.0


def test_outputs_negative_when_logsoftmax_true():
    # LogSoftmax outputs should be ≤ 0
    B, C, T, K = 2, 22, 1000, 4
    m = ShallowNet(C, T, K, logsoftmax=True).eval()
    x = torch.randn(B, C, T)
    with torch.no_grad():
        y = m(x)
    assert (y <= 1e-12).all()  # allow tiny numerical wiggle around 0