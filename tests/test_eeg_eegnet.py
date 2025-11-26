import torch
import numpy as np
from temporaldata import RegularTimeSeries, Interval

from torch_brain.models.eeg_eegnet import EEGNet


class DummyData:
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
):
    signal = np.random.randn(T, n_chans).astype(np.float32)
    zeros = np.zeros_like(signal)

    eeg = RegularTimeSeries(
        sampling_rate=sr,
        domain=Interval(0.0, T / sr),
        signal=signal,
        qc_rmse=zeros,
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


def test_tokenizer_outputs_expected_keys_and_shapes():
    model = EEGNet(in_chans=40, in_times=500, n_classes=4)
    data = make_dummy_data(n_chans=40, T=500)

    out = model.tokenize(data)

    expected_keys = {
        "input_values",
        "input_mask",
        "target_values",
        "target_weights",
        "model_hints",
    }
    assert set(out.keys()) == expected_keys

    x = out["input_values"]
    mask = out["input_mask"]
    y = out["target_values"]
    w = out["target_weights"]

    # input_values: [C, T’]
    assert x.dim() == 2
    assert x.shape[0] == 40  # channels

    # mask: [T’]
    assert mask.dim() == 1
    assert mask.dtype == torch.bool
    assert mask.shape[0] == x.shape[1]

    # targets
    assert y.shape == torch.Size([])
    assert y.dtype == torch.long
    assert w.shape == torch.Size([])
    assert w.dtype == torch.float32


def test_forward_output_shape_matches_n_classes():
    model = EEGNet(in_chans=40, in_times=500, n_classes=4)
    data = make_dummy_data(n_chans=40, T=500)

    out = model.tokenize(data)
    x = out["input_values"].unsqueeze(0)  # [1, C, T]

    logits = model(x)
    assert logits.shape == (1, 4)


def test_padding_applied_for_short_inputs():
    """Check that tokenizer pads too-short windows to min_T and masks correctly."""
    model = EEGNet(in_chans=40, in_times=500, n_classes=4)

    # Make a short EEG: T_short < min_T
    T_short = 10
    short_data = make_dummy_data(n_chans=40, T=T_short, sr=250.0)

    out = model.tokenize(short_data)
    x = out["input_values"]  # [C, T’]
    mask = out["input_mask"]  # [T’]

    # min_T is defined in the tokenizer as:
    # min_T = pool_time_length * sep_pool_time_length
    min_T = model.pool_time_length * model.sep_pool_time_length

    assert x.shape[1] == min_T  # time dimension padded to min_T
    assert mask.shape[0] == min_T  # mask matches padded length
    assert mask.sum().item() == T_short  # only original timepoints are True


def test_auto_kernel_still_runs_and_shapes_match():
    """Sanity check: auto_kernel produces a working model and forward pass."""
    model = EEGNet(
        in_chans=40,
        in_times=120,
        n_classes=3,
        auto_kernel=True,
        verbose=False,
    )
    data = make_dummy_data(n_chans=40, T=120)

    out = model.tokenize(data)
    x = out["input_values"].unsqueeze(0)  # [1, C, T’]
    logits = model(x)

    assert logits.shape == (1, 3)
