import os

import numpy as np
import pytest

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

import neuroprobe_eval.utils.logging_utils as logging_utils
from neuroprobe_eval.utils.logging_utils import (
    DEFAULT_RESULTS_TIME_BIN,
    build_internal_eval_result,
    build_public_export_result,
    format_results,
    log,
    log_final_wandb_metrics,
    log_fold_metrics,
    resolve_public_result_filename,
    resolve_public_subject_identifier,
)


def test_internal_eval_result_and_btb_export_boundary():
    results_population = {
        "one_second_after_onset": {
            "time_bin_start": 0.0,
            "time_bin_end": 1.0,
            "folds": [
                {
                    "fold_idx": 0,
                    "test_accuracy": 0.6,
                    "test_roc_auc": 0.7,
                }
            ],
        }
    }
    internal = build_internal_eval_result(
        provider="neuroprobe2025",
        task="onset",
        regime="SS-SM",
        subject_id=2,
        trial_id=0,
        model_name="logistic",
        preprocess_type="raw",
        preprocess_parameters={"name": "raw"},
        seed=42,
        results_population=results_population,
        subject_load_time=1.2,
        regression_run_time=3.4,
    )

    exported = format_results(
        internal_result=internal,
        author="tester",
        organization="org",
        organization_url="https://example.com",
    )

    assert internal["provider"] == "neuroprobe2025"
    assert internal["time_bins"][0]["name"] == "one_second_after_onset"
    assert internal["time_bins"][0]["fold_metrics"][0]["fold_idx"] == 0
    assert "model_name" not in internal["config_summary"]

    assert (
        exported["evaluation_results"]["btbank2_0"]["population"] == results_population
    )
    assert exported["config"]["eval_name"] == "onset"
    assert exported["config"]["splits_type"] == "SS-SM"
    assert exported["config"]["model_name"] == "logistic"

    exported_via_public_helper = build_public_export_result(
        internal_result=internal,
        author="tester",
        organization="org",
        organization_url="https://example.com",
    )
    assert exported_via_public_helper == exported
    assert resolve_public_subject_identifier(subject_id=2, trial_id=0) == "btbank2_0"
    assert (
        resolve_public_result_filename(
            eval_name="onset",
            subject_id=2,
            trial_id=0,
        )
        == "population_btbank2_0_onset.json"
    )


def test_log_degrades_gracefully_when_psutil_unavailable(monkeypatch):
    logged_messages = []

    monkeypatch.setattr(logging_utils, "psutil", None)
    monkeypatch.setattr(logging_utils.logger, "info", logged_messages.append)

    log("hello", priority=0)

    assert len(logged_messages) == 1
    assert "hello" in logged_messages[0]
    assert "ram" in logged_messages[0]
    assert "n/a" in logged_messages[0]


def test_log_fold_metrics_handles_skipped_fold_payload(monkeypatch):
    logged_messages = []
    monkeypatch.setattr(logging_utils.logger, "info", logged_messages.append)

    log_fold_metrics(
        2,
        {
            "status": "skipped",
            "skip_reason": "insufficient_class_coverage",
            "insufficient_splits": ["val"],
            "train_class_counts": {"0": 10, "1": 11},
            "val_class_counts": {"0": 6},
            "test_class_counts": {"0": 7, "1": 8},
        },
    )

    assert len(logged_messages) == 1
    assert "Fold 2: skipped" in logged_messages[0]
    assert "insufficient_class_coverage" in logged_messages[0]
    assert "splits=['val']" in logged_messages[0]


def test_log_final_wandb_metrics_handles_missing_default_bin():
    class _FakeWandbRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    run = _FakeWandbRun()
    log_final_wandb_metrics(run, {"different_bin": {"folds": []}})
    assert run.logged == []


def test_log_final_wandb_metrics_aggregates_present_metrics_only():
    class _FakeWandbRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    run = _FakeWandbRun()
    results_population = {
        DEFAULT_RESULTS_TIME_BIN: {
            "folds": [
                {
                    "train_accuracy": 0.8,
                    "train_roc_auc": 0.85,
                    "val_accuracy": 0.75,
                    "val_roc_auc": 0.8,
                    "test_accuracy": 0.78,
                    "test_roc_auc": 0.81,
                },
                {
                    "status": "skipped",
                    "skip_reason": "insufficient_class_coverage",
                },
                {
                    "train_accuracy": 0.9,
                    "test_accuracy": 0.88,
                    "test_roc_auc": 0.9,
                },
            ]
        }
    }

    log_final_wandb_metrics(run, results_population)
    assert len(run.logged) == 1
    payload = run.logged[0]
    assert payload["final/n_folds"] == 3
    assert payload["final/n_completed_folds"] == 2
    assert payload["final/n_skipped_folds"] == 1
    assert payload["final/train_accuracy_mean"] == pytest.approx(0.85)
    assert payload["final/test_roc_auc_mean"] == pytest.approx((0.81 + 0.9) / 2.0)
    assert not np.isnan(payload["final/val_roc_auc_std"])
