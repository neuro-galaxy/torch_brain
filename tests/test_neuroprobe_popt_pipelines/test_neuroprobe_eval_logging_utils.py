import os

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

import neuroprobe_eval.utils.logging_utils as logging_utils
from neuroprobe_eval.utils.logging_utils import (
    build_internal_eval_result,
    build_public_export_result,
    format_results,
    log,
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
