"""Fold preparation and execution helpers for processed neuroprobe eval."""

from __future__ import annotations

import gc
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroprobe_eval.models import build_model
from neuroprobe_eval.torch_runner import TorchRunner
from neuroprobe_eval.utils.collate import variable_channel_collate
from neuroprobe_eval.utils.logging_utils import (
    DEFAULT_RESULTS_TIME_BIN,
    log,
    log_fold_metrics,
)
from neuroprobe_eval.utils.data_adapter import build_neuroprobe_torch_fold


def collect_numpy_from_loader(loader, *, model=None, runner_cfg=None):
    """Collect one split DataLoader into numpy arrays."""
    xs = []
    ys = []
    expected_feature_shape = None
    for batch_idx, raw_batch in enumerate(loader):
        batch = raw_batch
        prepare_batch = getattr(model, "prepare_batch", None)
        if callable(prepare_batch):
            prepare_kwargs = {}
            if runner_cfg is not None:
                prepare_kwargs["runner_cfg"] = runner_cfg
            batch = prepare_batch(batch, **prepare_kwargs)
        # Fixed-alignment sklearn-style evaluation expects dense arrays only.
        if not isinstance(batch, dict) or "x" not in batch or "y" not in batch:
            raise ValueError(
                "fixed-channel loader materialization requires dict batches "
                "with 'x' and 'y' keys."
            )
        x = batch["x"]
        y = batch["y"]
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)

        feature_shape = tuple(x.shape[1:])
        if expected_feature_shape is None:
            expected_feature_shape = feature_shape
        elif feature_shape != expected_feature_shape:
            raise ValueError(
                "Inconsistent batch feature shape while materializing fixed-channel "
                "arrays: first batch x.shape[1:]="
                f"{expected_feature_shape}, batch {batch_idx} x.shape[1:]="
                f"{feature_shape}. Ensure preprocessors/model.prepare_batch produce "
                "a consistent feature layout."
            )
        xs.append(x)
        ys.append(y)

    if not xs:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int32)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int32, copy=False)
    return X, y


def merge_eval_splits_for_logistic(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return merged eval split (val + test), preserving original order."""
    if y_val.size == 0:
        return X_test, y_test
    if y_test.size == 0:
        return X_val, y_val
    if X_val.ndim != X_test.ndim or X_val.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Cannot merge validation/test splits for logistic evaluation: "
            "feature shapes differ "
            f"{X_val.shape} vs {X_test.shape}."
        )
    return (
        np.concatenate((X_val, X_test), axis=0),
        np.concatenate((y_val, y_test), axis=0),
    )


def _init_results_population():
    return {
        DEFAULT_RESULTS_TIME_BIN: {
            "time_bin_start": 0.0,
            "time_bin_end": 1.0,
            "folds": [],
        }
    }


def build_torch_split_loaders(
    fold: dict,
    cfg,
    *,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build split-specific torch DataLoaders for a variable-channel fold."""
    runner_cfg = cfg.runner
    batch_size = int(cfg.model.get("batch_size", 64))
    num_workers = int(runner_cfg.get("num_workers", 0))
    pin_memory = bool(runner_cfg.get("pin_memory", True))

    if num_workers > 0:
        warnings.warn(
            "dataset_variable_channel with runner.num_workers>0 can increase memory "
            "usage due to worker copies of materialized split datasets and may be "
            "less stable on some systems. V1 recommendation: runner.num_workers=0.",
            UserWarning,
            stacklevel=2,
        )
    train_generator = None
    if seed is not None:
        # Seed only the train loader's sampling order; val/test stay deterministic
        # via `shuffle=False`.
        train_generator = torch.Generator()
        train_generator.manual_seed(seed)

    train_loader = DataLoader(
        fold["train_split"],
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate,
    )
    val_loader = DataLoader(
        fold["val_split"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate,
    )
    test_loader = DataLoader(
        fold["test_split"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_channel_collate,
    )

    return train_loader, val_loader, test_loader


def run_processed_fold_loop(
    *,
    fold_iter,
    evaluate_fold,
    data_load_time: float,
    prepare_label: str = "prepared payload",
):
    """Execute the per-fold evaluation loop for processed-data runs."""
    results_population = _init_results_population()
    regression_start_time = time.time()

    for fold in fold_iter:
        # Fold iterator yields timing + payload so we can account for data
        # preparation separately from model training/inference time.
        fold_idx = fold["fold_idx"]
        fold_prepare_seconds = fold.get("prepare_seconds", 0.0)
        data_load_time += fold_prepare_seconds
        log(
            f"Fold {fold_idx}: {prepare_label} in {fold_prepare_seconds:.2f}s",
            priority=0,
        )

        fold_result = evaluate_fold(fold_idx, fold)
        fold_result["fold_idx"] = fold_idx
        results_population[DEFAULT_RESULTS_TIME_BIN]["folds"].append(fold_result)
        log_fold_metrics(fold_idx, fold_result)

    regression_run_time = time.time() - regression_start_time
    return results_population, data_load_time, regression_run_time


def iter_variable_channel_folds(
    *,
    n_folds: int,
    dataset_cfg,
    preprocessor,
    seed: int,
    require_coords: bool,
    needs_pool: bool,
):
    """Yield prepared variable-channel fold payloads with timing metadata."""
    for fold_idx in range(n_folds):
        fold_prepare_start = time.time()
        fold = build_neuroprobe_torch_fold(
            dataset_cfg,
            preprocessor,
            fold_idx=fold_idx,
            seed=seed + fold_idx,
            require_coords=require_coords,
            needs_pool=needs_pool,
        )
        yield {
            "fold_idx": fold_idx,
            "prepare_seconds": time.time() - fold_prepare_start,
            "fold": fold,
        }


def evaluate_variable_fold(
    fold_idx,
    fold_payload,
    *,
    cfg,
    runner,
    model_name: str,
    seed: int,
):
    """Evaluate one prepared fold and return fold metrics."""
    fold = fold_payload["fold"]
    train_loader, val_loader, test_loader = build_torch_split_loaders(
        fold,
        cfg,
        seed=seed + fold_idx,
    )

    fold_model = build_model(cfg.model)
    if model_name == "logistic":
        merge_val_into_test = cfg.dataset.merge_val_into_test
        # Logistic path consumes dense numpy arrays derived from split loaders.
        X_train, y_train = collect_numpy_from_loader(
            train_loader,
            model=fold_model,
        )
        X_test, y_test = collect_numpy_from_loader(
            test_loader,
            model=fold_model,
        )
        if merge_val_into_test:
            X_val, y_val = collect_numpy_from_loader(
                val_loader,
                model=fold_model,
            )
            X_test, y_test = merge_eval_splits_for_logistic(
                X_val,
                y_val,
                X_test,
                y_test,
            )
        if X_train.ndim != 2 or X_test.ndim != 2:
            raise ValueError(
                "Logistic fixed-channel evaluation expects 2D features after "
                "model.prepare_batch materialization, got "
                f"X_train.shape={X_train.shape}, "
                f"X_test.shape={X_test.shape}."
            )
        fold_result = runner.run_fold(
            fold_model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    else:
        # Variable-channel and aligned non-logistic models use the same runner path.
        if not isinstance(runner, TorchRunner):
            raise NotImplementedError(
                "Non-logistic models in dataset_variable_channel mode require TorchRunner."
            )
        fold_result = runner.run_fold(
            fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fold_idx=fold_idx,
        )

    del fold_model, fold
    gc.collect()
    return fold_result
