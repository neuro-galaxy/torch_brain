import os

import torch
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.models.mlp_model import MLPModel


def test_mlp_model_init_does_not_override_global_torch_seed(monkeypatch):
    manual_seed_calls = []

    def _record_manual_seed(seed):
        manual_seed_calls.append(seed)
        return torch.Generator()

    monkeypatch.setattr(torch, "manual_seed", _record_manual_seed)

    cfg = OmegaConf.create({"name": "mlp", "device": "cpu", "random_state": 1234})
    model = MLPModel(cfg)

    assert model.random_state == 1234
    assert manual_seed_calls == []
