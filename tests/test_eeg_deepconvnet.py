import torch
import pytest
from torch_brain.models.eeg_deepconvnet import DeepConvNet
from temporaldata import RegularTimeSeries, Interval
import numpy as np


class DummyData:
    """Minimal Data-like object for DeepConvNet.tokenize."""

    def __init__(self, eeg, trials=None):
        self.eeg = eeg
        if trials is not None:
            self.trials = trials


def make_dummy_data(
    n_chans: int = 40,
    T: int = 500,
    sr: float = 250.0,
    label: int = 1,
    with_trials: bool = True,
) -> DummyData:
    """Create a minimal temporaldata-like EEG sample."""

    # [T, C]
    signal = np.random.randn(T, n_chans).astype(np.float32)

    # Match RegularTimeSeries semantics: *same first dimension* as signal
    zeros = np.zeros_like(signal, dtype=np.float32)  # [T, C]

    domain = Interval(0.0, T / sr)

    eeg = RegularTimeSeries(
        sampling_rate=sr,
        domain=domain,
        signal=signal,  # [T, C]
        qc_rmse=zeros,  # [T, C] same first dim (T)
        quality_metric=zeros,
    )

    if with_trials:
        trials = Interval(
            start=np.array([0.0], dtype=np.float32),
            end=np.array([T / sr], dtype=np.float32),
            label=np.array([label]),
            timekeys=["start", "end"],
        )
        return DummyData(eeg=eeg, trials=trials)

    return DummyData(eeg=eeg)


# ---------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------


def test_tokenizer_outputs_expected_keys():
    model = DeepConvNet(in_chans=40, in_times=500, n_classes=4)
    data = make_dummy_data()

    out = model.tokenize(data)

    expected = {
        "input_values",
        "input_mask",
        "target_values",
        "target_weights",
        "model_hints",
    }

    assert set(out.keys()) == expected
    assert out["input_values"].ndim == 2  # [C, T]
    assert out["input_mask"].dtype == torch.bool
    assert out["target_values"].dtype == torch.long
    assert out["target_weights"].dtype == torch.float32


def test_forward_output_shape_matches_classes():
    model = DeepConvNet(in_chans=40, in_times=500, n_classes=4)
    data = make_dummy_data()

    tokens = model.tokenize(data)
    x = tokens["input_values"].unsqueeze(0)  # [1, C, T]
    yhat = model(x)

    assert yhat.shape == (1, 4)


def test_padding_respects_min_T():
    """DeepConvNet must pad to *its* true min_T, not ShallowNet's."""
    model = DeepConvNet(in_chans=40, in_times=500, n_classes=4)

    # DeepConvNet usually has a min_T ≈ 100–300 depending on kernel settings.
    min_T = model.min_T
    assert isinstance(min_T, int) and min_T > 0

    # Use a deliberately short window
    T_short = max(10, min_T // 4)
    short_data = make_dummy_data(T=T_short, with_trials=False)

    out = model.tokenize(short_data)
    x = out["input_values"]
    mask = out["input_mask"]

    # Must be padded up to exactly min_T
    assert x.shape[1] == min_T
    assert mask.sum().item() == T_short


def test_backward_pass_runs():
    model = DeepConvNet(in_chans=40, in_times=500, n_classes=4)
    data = make_dummy_data()

    tokens = model.tokenize(data)
    x = tokens["input_values"].unsqueeze(0)  # [1, C, T]
    y = torch.tensor([1], dtype=torch.long)

    logits = model(x)
    loss = torch.nn.functional.nll_loss(logits, y)
    loss.backward()  # Ensure gradients flow
