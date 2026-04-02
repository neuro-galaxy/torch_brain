import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

import neuroprobe_eval.torch_runner as torch_runner_module
from neuroprobe_eval.models import build_model
from neuroprobe_eval.torch_runner import TorchRunner
from neuroprobe_eval.utils.collate import variable_channel_collate


def _cfg(**model_overrides):
    model = {
        "name": "mlp",
        "device": "cpu",
        "deterministic": False,
    }
    model.update(model_overrides)
    return OmegaConf.create({"model": model, "seed": 0})


def test_torch_runner_auto_device_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runner = TorchRunner(_cfg(device="auto"))
    assert runner.device.type == "cpu"


def test_torch_runner_cuda_requested_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        TorchRunner(_cfg(device="cuda"))


def test_torch_runner_deterministic_mode_configures_torch(monkeypatch):
    calls = {"enabled": None}

    def _fake_use_deterministic_algorithms(enabled):
        calls["enabled"] = bool(enabled)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", _fake_use_deterministic_algorithms
    )
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    runner = TorchRunner(_cfg(device="cpu", deterministic=True))

    assert runner.deterministic is True
    assert calls["enabled"] is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_torch_runner_configures_determinism_before_device_resolution(monkeypatch):
    call_order = []

    def _fake_configure(self):
        _ = self
        call_order.append("configure")
        return True

    def _fake_get_device(self, cfg):
        _ = (self, cfg)
        call_order.append("device")
        return torch.device("cpu")

    monkeypatch.setattr(TorchRunner, "_configure_determinism", _fake_configure)
    monkeypatch.setattr(TorchRunner, "_get_device", _fake_get_device)

    _ = TorchRunner(_cfg(device="cpu", deterministic=True))
    assert call_order == ["configure", "device"]


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _sample(label: int, split: str, sample_idx: int):
    x = np.array(
        [
            [0.1 + label, 0.2 + label, 0.3 + label],
            [0.4 + label, 0.5 + label, 0.6 + label],
        ],
        dtype=np.float32,
    )
    return {
        "x": x,
        "y": int(label),
        "channel_ids": ["c0", "c1"],
        "channel_coords_lip": np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        ),
        "seq_id": np.zeros((2,), dtype=np.int64),
        "recording_id": f"{split}_rec",
        "split": split,
        "sample_idx": int(sample_idx),
        "window_start_sec": 0.0,
        "window_end_sec": 1.0,
    }


def test_torch_runner_loader_mode_runs_fold():
    cfg = _cfg(
        name="mlp",
        device="cpu",
        deterministic=False,
        training_mode="epoch_based",
        max_iter=1,
        patience=1,
        batch_size=2,
    )
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)

    train_ds = _ListDataset(
        [
            _sample(0, "train", 0),
            _sample(1, "train", 1),
            _sample(0, "train", 2),
            _sample(1, "train", 3),
        ]
    )
    val_ds = _ListDataset(
        [
            _sample(0, "val", 0),
            _sample(1, "val", 1),
            _sample(0, "val", 2),
            _sample(1, "val", 3),
        ]
    )
    test_ds = _ListDataset(
        [
            _sample(0, "test", 0),
            _sample(1, "test", 1),
            _sample(0, "test", 2),
            _sample(1, "test", 3),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=variable_channel_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )

    result = runner.run_fold(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert set(result.keys()) == {
        "train_accuracy",
        "train_roc_auc",
        "val_accuracy",
        "val_roc_auc",
        "test_accuracy",
        "test_roc_auc",
    }
    for key in result:
        assert isinstance(result[key], float), f"{key} should be float"


def test_torch_runner_loader_mode_steps_based(monkeypatch):
    """Steps-based training via loaders should complete and return metrics."""
    messages = []
    monkeypatch.setattr(
        torch_runner_module,
        "log",
        lambda message, priority=0, indent=0: messages.append(str(message)),
    )

    cfg = _cfg(
        name="mlp",
        device="cpu",
        deterministic=False,
        training_mode="steps_based",
        total_steps=4,
        validation_interval=2,
        batch_size=2,
    )
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)

    train_ds = _ListDataset(
        [_sample(0, "train", i) for i in range(4)]
        + [_sample(1, "train", i) for i in range(4, 8)]
    )
    val_ds = _ListDataset([_sample(0, "val", 0), _sample(1, "val", 1)])
    test_ds = _ListDataset([_sample(0, "test", 0), _sample(1, "test", 1)])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=variable_channel_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, shuffle=False, collate_fn=variable_channel_collate
    )

    result = runner.run_fold(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert set(result.keys()) == {
        "train_accuracy",
        "train_roc_auc",
        "val_accuracy",
        "val_roc_auc",
        "test_accuracy",
        "test_roc_auc",
    }
    validation_logs = [msg for msg in messages if "train_step=" in msg]
    assert len(validation_logs) == 2
    for msg in validation_logs:
        assert "train_loss=" in msg
        assert "train_acc=" in msg
        assert "train_roc_auc=" in msg
        assert "val_loss=" in msg
        assert "val_acc=" in msg
        assert "val_roc_auc=" in msg


def test_torch_runner_rejects_missing_loaders():
    """run_fold should require all split DataLoaders."""
    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)

    with pytest.raises(
        ValueError, match="run_fold requires train_loader, val_loader, and test_loader"
    ):
        runner.run_fold(model)


def test_torch_runner_rejects_invalid_loader_types():
    """Loader mode should raise if loaders aren't DataLoader instances."""
    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)
    empty_samples = [_sample(0, "val", 0), _sample(1, "val", 1)]
    val_loader = torch.utils.data.DataLoader(
        _ListDataset(empty_samples),
        batch_size=2,
        shuffle=False,
        collate_fn=variable_channel_collate,
    )
    test_loader = torch.utils.data.DataLoader(
        _ListDataset(empty_samples),
        batch_size=2,
        shuffle=False,
        collate_fn=variable_channel_collate,
    )

    with pytest.raises(TypeError, match="train_loader must be"):
        runner.run_fold(
            model,
            train_loader="not_a_loader",
            val_loader=val_loader,
            test_loader=test_loader,
        )


def test_forward_model_passes_pad_mask_when_supported():
    class _PadMaskAwareModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_positions = None
            self.last_pad_mask = None

        def forward(self, x, positions=None, pad_mask=None):
            self.last_positions = positions
            self.last_pad_mask = pad_mask
            return torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)

    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)
    module = _PadMaskAwareModule().to(runner.device)

    x = torch.randn(2, 3, 4, device=runner.device)
    coords = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0]],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [1.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    seq_id = torch.zeros((2, 3), dtype=torch.long)
    pad_mask = torch.tensor([[False, False, True], [False, True, True]])

    _ = runner._forward_model(
        module,
        x,
        coords=coords,
        seq_id=seq_id,
        model_kwargs={"pad_mask": pad_mask},
    )

    assert module.last_positions is not None
    assert module.last_pad_mask is not None
    assert tuple(module.last_pad_mask.shape) == (2, 3)


def test_forward_model_ignores_coords_for_modules_without_coord_support():
    class _NoCoordsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = 0

        def forward(self, x):
            self.called += 1
            return torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)

    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)
    module = _NoCoordsModule().to(runner.device)

    x = torch.randn(2, 3, 4, device=runner.device)
    coords = torch.randn(2, 3, 3)
    seq_id = torch.zeros((2, 3), dtype=torch.long)

    out = runner._forward_model(
        module,
        x,
        coords=coords,
        seq_id=seq_id,
        accepts_coords=False,
    )

    assert tuple(out.shape) == (2, 2)
    assert module.called == 1


def test_forward_model_moves_coords_for_legacy_coord_models(monkeypatch):
    class _LegacyCoordsModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_coords_device = None

        def forward(self, x, coords):
            self.last_coords_device = coords.device
            return torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)

    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)
    module = _LegacyCoordsModule().to(runner.device)

    x = torch.randn(2, 3, 4, device=runner.device)
    coords = torch.randn(2, 3, 3)

    original_is_tensor = torch.is_tensor

    def _fake_is_tensor(_obj):
        return False

    monkeypatch.setattr(torch, "is_tensor", _fake_is_tensor)
    try:
        out = runner._forward_model(
            module,
            x,
            coords=coords,
            accepts_coords=True,
        )
    finally:
        monkeypatch.setattr(torch, "is_tensor", original_is_tensor)

    assert tuple(out.shape) == (2, 2)
    assert module.last_coords_device == runner.device


def test_create_optimizer_and_scheduler_falls_back_to_adam():
    cfg = _cfg(name="mlp", device="cpu", optimizer="NotAnOptimizer")
    runner = TorchRunner(cfg)
    model = build_model(cfg.model)
    model.build_model((2, 3), 2, device=runner.device)

    optimizer, scheduler, learning_rate = runner._create_optimizer_and_scheduler(model)

    assert isinstance(optimizer, torch.optim.Adam)
    assert scheduler is None
    assert learning_rate == pytest.approx(0.001)


def test_infer_classes_from_loader_requires_dict_batch_with_y():
    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)

    tuple_dataset = _ListDataset(
        [
            (np.array([0.0], dtype=np.float32), 0),
            (np.array([1.0], dtype=np.float32), 1),
        ]
    )
    loader = torch.utils.data.DataLoader(tuple_dataset, batch_size=2, shuffle=False)

    with pytest.raises(ValueError, match="Expected dict batches with key 'y'"):
        runner._infer_classes_from_loader(loader)


def test_infer_classes_from_loader_rejects_non_integer_labels():
    cfg = _cfg(name="mlp", device="cpu")
    runner = TorchRunner(cfg)

    samples = [0.5, 1.0]
    loader = torch.utils.data.DataLoader(
        _ListDataset(samples),
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: {"y": np.asarray(batch, dtype=np.float32)},
    )
    with pytest.raises(TypeError, match="Class labels must be integers"):
        runner._infer_classes_from_loader(loader)
