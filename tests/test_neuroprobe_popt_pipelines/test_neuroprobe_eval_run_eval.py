import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

import neuroprobe_eval.run_eval as run_eval_module


class _FakeWandb:
    def __init__(self):
        self.run = None
        self.init_calls = 0
        self.finish_calls = 0

    def init(self, **kwargs):
        _ = kwargs
        self.init_calls += 1
        self.run = SimpleNamespace(name="fake-run")

    def finish(self):
        self.finish_calls += 1


def _cfg(*, wandb_enabled: bool = True):
    return OmegaConf.create(
        {
            "runtime": {"verbose": False, "seed": 7, "overwrite": True},
            "wandb": {
                "enabled": wandb_enabled,
                "project": "proj",
                "entity": None,
                "name": "run-name",
                "tags": ["a"],
                "notes": None,
                "group": None,
            },
            "model": {
                "name": "logistic",
                "requires_coords": False,
                "requires_aligned_channels": False,
            },
            "preprocessor": {"name": "raw"},
            "dataset": {
                "provider": "neuroprobe2025",
                "regime": "SS-SM",
                "task": "onset",
                "test_subject": 1,
                "test_session": 1,
                "merge_val_into_test": True,
            },
            "submitter": {
                "author": "tester",
                "organization": "org",
                "organization_url": "https://example.com",
            },
        }
    )


def test_main_validates_config_before_wandb_init(monkeypatch):
    cfg = _cfg(wandb_enabled=True)
    fake_wandb = _FakeWandb()
    call_order = []

    monkeypatch.setattr(run_eval_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(run_eval_module, "wandb", fake_wandb)
    monkeypatch.setattr(run_eval_module, "set_verbose", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_module, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_module, "build_preprocessor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_eval_module, "SKLearnRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        run_eval_module, "run_processed_evaluation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        run_eval_module,
        "validate_eval_config",
        lambda *_args, **_kwargs: call_order.append("validate"),
    )

    original_init = fake_wandb.init

    def _init(**kwargs):
        _ = kwargs
        call_order.append("init")
        original_init()

    monkeypatch.setattr(fake_wandb, "init", _init)

    run_eval_module.main.__wrapped__(cfg)

    assert call_order[:2] == ["validate", "init"]
    assert fake_wandb.init_calls == 1
    assert fake_wandb.finish_calls == 1


def test_main_finishes_wandb_when_processed_eval_raises(monkeypatch):
    cfg = _cfg(wandb_enabled=True)
    fake_wandb = _FakeWandb()

    monkeypatch.setattr(run_eval_module, "WANDB_AVAILABLE", True)
    monkeypatch.setattr(run_eval_module, "wandb", fake_wandb)
    monkeypatch.setattr(run_eval_module, "set_verbose", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_module, "log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_eval_module, "build_preprocessor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_eval_module, "SKLearnRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_eval_module, "validate_eval_config", lambda *_args, **_kwargs: None)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_eval_module, "run_processed_evaluation", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        run_eval_module.main.__wrapped__(cfg)

    assert fake_wandb.init_calls == 1
    assert fake_wandb.finish_calls == 1
