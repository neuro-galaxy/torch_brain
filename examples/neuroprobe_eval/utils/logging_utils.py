"""
Logging and result formatting utilities.
"""

import time
import torch
import json
import os
import logging
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

try:
    import psutil
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    psutil = None


verbose = True  # Global verbose flag

# Get root logger - Hydra automatically configures it
# Using root logger avoids showing module name in logs
logger = logging.getLogger()

DEFAULT_RESULTS_TIME_BIN = "one_second_after_onset"
PUBLIC_SUBJECT_KEY_PREFIX = "btbank"


def set_verbose(value):
    """Set global verbose flag."""
    # Module-level flag keeps logging behavior consistent across imports without
    # threading extra config through every helper.
    global verbose
    verbose = value
    logger.info(f"Verbose set to {value}")


def log(message, priority=0, indent=0):
    """
    Log a message with timestamp and resource usage.
    Uses Python's logging module, which Hydra automatically configures to output
    to both console and log file.

    Args:
        message: Message to log
        priority: Priority level (higher = less important). Priority 0 is always logged.
        indent: Indentation level
    """
    # Priority 0 messages are always logged (critical/important messages)
    # When verbose=False, only priority 0 is logged
    # When verbose=True, priorities 0-4 are logged
    max_log_priority = 0 if not verbose else 4
    if priority > max_log_priority:
        return

    # Use reserved GPU memory (not allocated) to reflect allocator pressure.
    gpu_memory_reserved = (
        torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    )
    ram_usage = _resolve_ram_usage_gb()
    ram_display = f"{ram_usage:05.1f}G" if ram_usage is not None else "  n/a"
    formatted_message = (
        f"[gpu {gpu_memory_reserved:04.1f}G ram {ram_display}] {' '*4*indent}{message}"
    )

    # Use logger - Hydra handles routing to console and file automatically
    logger.info(formatted_message)


def _resolve_ram_usage_gb():
    """Return current process RSS in GiB, or None when memory telemetry is unavailable."""
    if psutil is None:
        return None
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    except Exception:
        return None


def normalize_wandb_tags(raw_tags):
    """Normalize wandb tags from either list or comma-delimited string input."""
    if raw_tags is None or isinstance(raw_tags, list):
        return raw_tags
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",") if t.strip()]
    return [raw_tags] if raw_tags else None


def resolve_public_subject_identifier(*, subject_id: int, trial_id: int) -> str:
    """Return the public/export subject key for one evaluated recording."""
    return f"{PUBLIC_SUBJECT_KEY_PREFIX}{subject_id}_{trial_id}"


def resolve_public_result_filename(
    *,
    eval_name: str,
    subject_id: int,
    trial_id: int,
) -> str:
    """Return the current public/export result filename for one run."""
    subject_identifier = resolve_public_subject_identifier(
        subject_id=subject_id,
        trial_id=trial_id,
    )
    return f"population_{subject_identifier}_{eval_name}.json"


def build_internal_time_bins(results_population: dict) -> list[dict]:
    """Convert the in-memory results structure into internal time-bin payloads."""
    time_bins = []
    for bin_name, bin_payload in dict(results_population).items():
        time_bins.append(
            {
                "name": str(bin_name),
                "time_bin_start": float(bin_payload["time_bin_start"]),
                "time_bin_end": float(bin_payload["time_bin_end"]),
                "fold_metrics": list(bin_payload.get("folds", [])),
            }
        )
    return time_bins


def resolve_result_output_path(
    *,
    eval_name: str,
    subject_id: int,
    trial_id: int,
) -> str:
    """Resolve final JSON output path for one evaluation run.

    Writes into Hydra's runtime output directory so JSON results live next to
    `.hydra/` and `run_eval.log` for the same run.
    """
    if not HydraConfig.initialized():
        raise RuntimeError(
            "Hydra runtime is not initialized; cannot resolve result output path. "
            "Run via the Hydra entrypoint (run_eval.py) or initialize Hydra "
            "before calling evaluation helpers."
        )
    file_save_dir = str(HydraConfig.get().runtime.output_dir)
    os.makedirs(file_save_dir, exist_ok=True)
    return os.path.join(
        file_save_dir,
        resolve_public_result_filename(
            eval_name=eval_name,
            subject_id=subject_id,
            trial_id=trial_id,
        ),
    )


def should_skip_existing_output(cfg, file_save_path: str) -> bool:
    """Whether the current run should skip because the result file exists."""
    if os.path.exists(file_save_path) and not cfg.runtime.overwrite:
        log(f"Skipping {file_save_path} because it already exists", priority=0)
        return True
    return False


def log_fold_metrics(fold_idx: int, fold_result: dict) -> None:
    """Emit the standard fold metric summary line."""
    if "val_accuracy" in fold_result:
        log(
            f"Fold {fold_idx}: Val acc: {fold_result['val_accuracy']:.3f}, "
            f"Val AUC: {fold_result['val_roc_auc']:.3f}, "
            f"Test acc: {fold_result['test_accuracy']:.3f}, "
            f"Test AUC: {fold_result['test_roc_auc']:.3f}",
            priority=0,
        )
    else:
        log(
            f"Fold {fold_idx}: Test acc: {fold_result['test_accuracy']:.3f}, "
            f"Test AUC: {fold_result['test_roc_auc']:.3f}",
            priority=0,
        )


def log_fold_split_sample_counts(
    split_datasets: dict[str, object],
    *,
    fold_idx: int,
    phase: str,
) -> None:
    """Log split sample counts for one fold/materialization phase."""
    counts = {split: len(split_datasets[split]) for split in ("train", "val", "test")}
    log(
        f"Fold {fold_idx} {phase} split sample counts: "
        f"train={counts['train']} val={counts['val']} test={counts['test']}",
        priority=0,
    )


def format_results(
    internal_result,
    author,
    organization,
    organization_url,
):
    """
    Format BTB-style leaderboard JSON from internal evaluation results.

    Args:
        internal_result: Generic internal evaluation result payload.
        author: Author name for submission metadata.
        organization: Organization name for submission metadata.
        organization_url: Organization URL for submission metadata.

    Returns:
        BTB-compatible exported results dictionary.
    """
    return build_public_export_result(
        internal_result=internal_result,
        author=author,
        organization=organization,
        organization_url=organization_url,
    )


def build_public_export_result(
    *,
    internal_result,
    author,
    organization,
    organization_url,
):
    """Format the current public/export JSON artifact from an internal result."""
    model_name = internal_result["model_name"]
    preprocess_type = internal_result["preprocess_type"]
    subject_id = internal_result["test_subject"]
    trial_id = internal_result["test_session"]
    subject_identifier = resolve_public_subject_identifier(
        subject_id=subject_id,
        trial_id=trial_id,
    )
    config_summary = internal_result["config_summary"]

    return {
        "model_name": model_name,
        "author": author,
        "description": f"Simple {model_name} using all electrodes ({preprocess_type if preprocess_type != 'none' else 'voltage'}).",
        "organization": organization,
        "organization_url": organization_url,
        "timestamp": float(internal_result["timestamp"]),
        "evaluation_results": {
            subject_identifier: {"population": internal_result["results_population"]}
        },
        "config": {
            "preprocess": config_summary["preprocess"],
            "seed": config_summary["seed"],
            "subject_id": subject_id,
            "trial_id": trial_id,
            "eval_name": internal_result["task"],
            "splits_type": internal_result["regime"],
            "model_name": model_name,
        },
        "timing": dict(internal_result["timing"]),
    }


def build_internal_eval_result(
    *,
    provider,
    task,
    regime,
    subject_id,
    trial_id,
    model_name,
    preprocess_type,
    preprocess_parameters,
    seed,
    results_population,
    subject_load_time,
    regression_run_time,
):
    """Build generic internal evaluation result payload.

    This structure is provider/task/regime-oriented and does not assume BTB
    submission key naming. Export formatters can map it to release artifacts.
    """
    return {
        "provider": str(provider),
        "task": str(task),
        "regime": str(regime),
        "test_subject": int(subject_id),
        "test_session": int(trial_id),
        "model_name": str(model_name),
        "preprocess_type": str(preprocess_type),
        "time_bins": build_internal_time_bins(results_population),
        "results_population": dict(results_population),
        "timing": {
            "subject_load_time": float(subject_load_time),
            "regression_run_time": float(regression_run_time),
        },
        "config_summary": {
            "preprocess": preprocess_parameters,
            "seed": int(seed),
        },
        # Unix timestamp is easier to aggregate in downstream scripts than a
        # pre-formatted datetime string.
        "timestamp": time.time(),
    }


def log_final_wandb_metrics(wandb_run, results_population) -> None:
    """Log aggregate fold metrics to wandb at the end of a run."""
    if wandb_run is None:
        return
    folds_data = results_population[DEFAULT_RESULTS_TIME_BIN]["folds"]
    if not folds_data:
        return

    train_accs = [f.get("train_accuracy", 0) for f in folds_data]
    train_aucs = [f.get("train_roc_auc", 0) for f in folds_data]
    val_accs = [f.get("val_accuracy", 0) for f in folds_data]
    val_aucs = [f.get("val_roc_auc", 0) for f in folds_data]
    test_accs = [f.get("test_accuracy", 0) for f in folds_data]
    test_aucs = [f.get("test_roc_auc", 0) for f in folds_data]

    wandb_run.log(
        {
            "final/train_accuracy_mean": np.mean(train_accs),
            "final/train_accuracy_std": np.std(train_accs),
            "final/train_roc_auc_mean": np.mean(train_aucs),
            "final/train_roc_auc_std": np.std(train_aucs),
            "final/val_accuracy_mean": np.mean(val_accs),
            "final/val_accuracy_std": np.std(val_accs),
            "final/val_roc_auc_mean": np.mean(val_aucs),
            "final/val_roc_auc_std": np.std(val_aucs),
            "final/test_accuracy_mean": np.mean(test_accs),
            "final/test_accuracy_std": np.std(test_accs),
            "final/test_roc_auc_mean": np.mean(test_aucs),
            "final/test_roc_auc_std": np.std(test_aucs),
            "final/n_folds": len(folds_data),
        }
    )


def format_and_save_results(
    *,
    cfg,
    dataset_provider: str,
    model_name: str,
    preprocess_type: str,
    subject_id: int,
    trial_id: int,
    eval_name: str,
    results_splits_type: str,
    results_population: dict,
    data_load_time: float,
    regression_run_time: float,
    file_save_path: str,
):
    """Build internal/export result payloads and save them to disk."""
    runtime_cfg = cfg.runtime
    submitter_cfg = cfg.submitter
    preprocess_parameters = OmegaConf.to_container(cfg.preprocessor, resolve=True)
    internal_result = build_internal_eval_result(
        provider=dataset_provider,
        task=eval_name,
        regime=results_splits_type,
        subject_id=subject_id,
        trial_id=trial_id,
        model_name=model_name,
        preprocess_type=preprocess_type,
        preprocess_parameters=preprocess_parameters,
        seed=runtime_cfg.seed,
        results_population=results_population,
        subject_load_time=data_load_time,
        regression_run_time=regression_run_time,
    )
    results = format_results(
        internal_result=internal_result,
        author=submitter_cfg.author,
        organization=submitter_cfg.organization,
        organization_url=submitter_cfg.organization_url,
    )
    save_results(results, file_save_path)
    log("Evaluation complete!", priority=0)
    return results


def save_results(results, file_path):
    """
    Save results to JSON file.

    Args:
        results: Results dictionary
        file_path: Path to save file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        # Human-readable JSON helps spot-checking run outputs during sweeps.
        json.dump(results, f, indent=4)
    log(f"Results saved to {file_path}", priority=0)
