import numpy as np
import torch
import pytest

from temporaldata import Data, RegularTimeSeries, Interval
from torch_brain.models.eeg_deepconvnet import DeepConvNet


def make_temporaldata_eeg_sample(
    in_chans: int,
    in_times: int,
    label: int = 0,
    sampling_rate: float = 250.0,
) -> Data:
    """
    Build a minimal temporaldata.Data object that matches what
    DeepConvNet.tokenize expects:

        data.eeg.signal      -> np.ndarray [T, C]
        data.trials.label -> array-like with at least one element
    """
    # EEG signal: [T, C]
    signal = np.random.randn(in_times, in_chans).astype("float32")

    # Time domain 0..(T / fs)
    duration = in_times / sampling_rate
    domain = Interval(0.0, duration)

    # RegularTimeSeries: first dim is time, so shape (T, C) is correct
    eeg = RegularTimeSeries(
        signal=signal,  # <-- this becomes data.eeg.signal
        sampling_rate=sampling_rate,
        domain=domain,
    )

    # Trials: we only need a 'label' field that DeepConvNet.tokenize can read
    trials = Interval(
        start=np.array([0.0]),
        end=np.array([duration]),
        label=np.array([label]),
        timekeys=["start", "end"],
    )

    # Wrap everything into a Data object
    data = Data(
        eeg=eeg,
        trials=trials,
        domain=domain,
    )
    return data


def test_tokenize_with_real_temporaldata_objects():
    in_chans, in_times, n_classes = 40, 500, 4
    model = DeepConvNet(in_chans=in_chans, in_times=in_times, n_classes=n_classes)

    data = make_temporaldata_eeg_sample(in_chans, in_times, label=2)
    out = model.tokenize(data)

    # Check keys
    expected_keys = {
        "input_values",
        "input_mask",
        "target_values",
        "target_weights",
        "model_hints",
    }
    assert set(out.keys()) == expected_keys

    x = out["input_values"]  # [C, T]
    mask = out["input_mask"]  # [T]
    y = out["target_values"]  # scalar
    w = out["target_weights"]  # scalar
    hints = out["model_hints"]  # dict

    # Shapes & dtypes
    assert isinstance(x, torch.Tensor)
    assert x.ndim == 2
    assert x.shape[0] == in_chans  # channels

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape[0] == x.shape[1]  # time

    assert y.dtype == torch.long
    assert y.shape == torch.Size([])

    assert w.dtype == torch.float32
    assert w.shape == torch.Size([])

    # Labels should be preserved
    assert y.item() == 2
    assert w.item() == 1.0

    # model_hints consistency
    assert hints["in_chans"] == in_chans
    assert hints["in_times"] == x.shape[1]
    assert "min_time_required" in hints
    assert hints["min_time_required"] == model.min_T

    # kernel + pool hints exist and match model
    assert hints["kernel_time_1"] == model.kernel_time_1
    assert hints["kernel_time_2"] == model.kernel_time_2
    assert hints["kernel_time_3"] == model.kernel_time_3
    assert hints["kernel_time_4"] == model.kernel_time_4
    assert hints["pool_time_size"] == model.pool_time_size
    assert hints["pool_time_stride"] == model.pool_time_stride


def test_tokenizer_pads_short_temporaldata_sequences():
    # deliberately small in_times (model will internally bump to min_T)
    in_chans, in_times, n_classes = 40, 80, 4
    model = DeepConvNet(in_chans=in_chans, in_times=in_times, n_classes=n_classes)

    # Very short sequence T_short < min_T
    T_short = 20
    assert T_short < model.min_T  # sanity check

    data = make_temporaldata_eeg_sample(in_chans, T_short, label=1)
    out = model.tokenize(data)

    x = out["input_values"]  # [C, T_pad]
    mask = out["input_mask"]  # [T_pad]
    min_T = model.min_T

    assert x.shape[1] >= min_T
    assert mask.shape[0] == x.shape[1]

    # First T_short entries are True, rest False
    assert mask[:T_short].all()
    assert (~mask[T_short:]).all()


def test_forward_and_backward_with_tokenized_batch():
    in_chans, in_times, n_classes = 40, 500, 4
    model = DeepConvNet(in_chans=in_chans, in_times=in_times, n_classes=n_classes)

    batch_size = 8

    # Build a batch of temporaldata.Data, tokenize, then stack
    datas = [
        make_temporaldata_eeg_sample(in_chans, in_times, label=i % n_classes)
        for i in range(batch_size)
    ]
    tokens = [model.tokenize(d) for d in datas]

    # All tokenized samples should have same (C,T) after padding
    xs = torch.stack([t["input_values"] for t in tokens], dim=0)  # [B, C, T]
    ys = torch.stack([t["target_values"] for t in tokens], dim=0)  # [B]

    outputs = model(xs)  # [B, n_classes]
    assert outputs.shape == (batch_size, n_classes)

    # By default DeepConvNet uses log-softmax → nll_loss is appropriate
    loss = torch.nn.functional.nll_loss(outputs, ys)
    loss.backward()  # should run without error
