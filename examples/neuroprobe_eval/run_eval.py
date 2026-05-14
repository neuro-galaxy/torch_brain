"""
Main evaluation script for neuroprobe using Hydra configuration.
"""

from functools import partial
import logging

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
import time

from neuroprobe_eval.preprocessors import build_preprocessor
from neuroprobe_eval.sklearn_runner import SKLearnRunner
from neuroprobe_eval.torch_runner import TorchRunner
from neuroprobe_eval.utils.pipeline_contracts import (
    validate_eval_config,
    needs_region_intersection_pool,
    resolve_provider_n_folds,
)
from neuroprobe_eval.utils import fold_helpers, logging_utils

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from neuroprobe_eval.utils.logging_utils import (
    log,
    normalize_wandb_tags,
    set_verbose,
)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    root_logger = logging.getLogger()
    wandb_run = None
    try:
        # Hydra automatically configures Python logging to output to both console and log file
        # No manual setup needed - just use the logger from logging_utils

        # Setup
        runtime_cfg = cfg.get("runtime")
        if runtime_cfg is None:
            raise ValueError("cfg.runtime is required and must be a mapping.")
        if not hasattr(runtime_cfg, "get"):
            raise TypeError("cfg.runtime must be a mapping/dict-like object.")
        set_verbose(runtime_cfg.get("verbose", True))
        log("Starting neuroprobe evaluation", priority=0)
        log(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}", priority=1)

        # Validate config before any external service initialization (e.g., wandb).
        validate_eval_config(cfg)

        # Initialize wandb if enabled
        if cfg.get("wandb", {}).get("enabled", False):
            if not WANDB_AVAILABLE:
                log(
                    "WARNING: wandb is enabled in config but not installed. Install with: pip install wandb",
                    priority=0,
                )
            else:
                wandb_cfg = cfg.get("wandb", {})
                # Prepare config for wandb (convert OmegaConf to dict)
                wandb_config = OmegaConf.to_container(cfg, resolve=True)

                # Initialize wandb
                wandb.init(
                    project=wandb_cfg.get("project", "neuroprobe_eval"),
                    entity=wandb_cfg.get("entity") or None,
                    name=wandb_cfg.get("name") or None,
                    tags=normalize_wandb_tags(wandb_cfg.get("tags")),
                    notes=wandb_cfg.get("notes") or None,
                    group=wandb_cfg.get("group") or None,
                    config=wandb_config,
                    reinit=True,  # Allow reinitialization for multiple runs
                )
                wandb_run = wandb.run
                log(
                    f"Wandb initialized: project={wandb_cfg.get('project')}, run={wandb_run.name}",
                    priority=0,
                )

        # Set random seeds
        seed = runtime_cfg.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Build components
        preprocessor = build_preprocessor(cfg.preprocessor)
        runner = (
            SKLearnRunner(cfg) if cfg.model.name == "logistic" else TorchRunner(cfg)
        )  # Model will be built per fold

        # Pass wandb_run to runner if it's a TorchRunner
        if isinstance(runner, TorchRunner) and wandb_run is not None:
            runner.set_wandb_run(wandb_run)

        log(f"Using preprocessor: {cfg.preprocessor.name}", priority=0)
        log(f"Using model: {cfg.model.name}", priority=0)

        if "use_raw_data" in cfg or "raw_data_path" in cfg:
            raise ValueError(
                "Raw-data evaluation has been removed. "
                "Use processed dataset mode via dataset.* config."
            )

        run_processed_evaluation(
            cfg,
            preprocessor,
            runner,
            wandb_run,
        )
    except Exception:
        # Ensure uncaught failures are persisted to Hydra's run_eval.log file.
        root_logger.exception("Unhandled exception during neuroprobe_eval.run_eval")
        raise
    finally:
        if wandb_run is not None:
            wandb.finish()


def run_processed_evaluation(
    cfg,
    preprocessor,
    runner,
    wandb_run=None,
):
    """Run evaluation with processed data."""
    # Canonical processed path: provider-selected variable-channel datasets.
    log("Loading processed data via variable-channel split adapter", priority=0)
    setup_start_time = time.time()
    # Validate config before building providers/models to fail fast on bad inputs.
    validate_eval_config(cfg)
    dataset_cfg = cfg.dataset
    data_load_time = time.time() - setup_start_time
    log(f"Validated eval config in {data_load_time:.2f}s", priority=0)

    subject_id = dataset_cfg.test_subject
    trial_id = dataset_cfg.test_session
    eval_name = dataset_cfg.task
    regime = dataset_cfg.regime
    seed = cfg.runtime.seed

    dataset_provider = dataset_cfg.provider
    requires_aligned = cfg.model.requires_aligned_channels
    needs_pool = needs_region_intersection_pool(
        dataset_provider, regime, requires_aligned
    )
    n_folds = resolve_provider_n_folds(
        dataset_provider=dataset_provider,
        regime=regime,
    )

    log(f"Using dataset.provider='{dataset_provider}'", priority=0)
    log(f"Using dataset.regime='{regime}'", priority=0)
    log(f"Using n_folds={n_folds} from dataset class API", priority=0)

    preprocess_type = cfg.preprocessor.name
    model_name = cfg.model.name
    # Keep result JSON in the Hydra run folder for per-run portability.
    file_save_path = logging_utils.resolve_result_output_path(
        eval_name=eval_name,
        subject_id=subject_id,
        trial_id=trial_id,
    )
    require_coords = cfg.model.requires_coords
    # Short-circuit before fold construction if result already exists.
    if logging_utils.should_skip_existing_output(cfg, file_save_path):
        return

    results_population, data_load_time, regression_run_time = (
        fold_helpers.run_processed_fold_loop(
            fold_iter=fold_helpers.iter_variable_channel_folds(
                n_folds=n_folds,
                dataset_cfg=dataset_cfg,
                preprocessor=preprocessor,
                seed=seed,
                require_coords=require_coords,
                needs_pool=needs_pool,
            ),
            evaluate_fold=partial(
                fold_helpers.evaluate_variable_fold,
                cfg=cfg,
                runner=runner,
                model_name=model_name,
                seed=seed,
            ),
            data_load_time=data_load_time,
            prepare_label="prepared variable-channel payload",
        )
    )

    if hasattr(preprocessor, "unload_model"):
        preprocessor.unload_model()
    logging_utils.format_and_save_results(
        cfg=cfg,
        dataset_provider=dataset_provider,
        model_name=model_name,
        preprocess_type=preprocess_type,
        subject_id=subject_id,
        trial_id=trial_id,
        eval_name=eval_name,
        results_splits_type=regime,
        results_population=results_population,
        data_load_time=data_load_time,
        regression_run_time=regression_run_time,
        file_save_path=file_save_path,
    )
    logging_utils.log_final_wandb_metrics(wandb_run, results_population)


if __name__ == "__main__":
    main()
