"""
Main evaluation script for neuroprobe using Hydra configuration.
"""
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
import gc
import time

# Set ROOT_DIR_BRAINTREEBANK if not already set
if 'ROOT_DIR_BRAINTREEBANK' not in os.environ:
    default_path = '/home/geeling/Projects/tb_buildathon/data/raw/neuroprobe_2025'
    os.environ['ROOT_DIR_BRAINTREEBANK'] = default_path

from models import build_model
from preprocessors import build_preprocessor
from sklearn_runner import SKLearnRunner
from torch_runner import TorchRunner
from utils.data_loader import (
    subset_electrodes,
    create_folds,
    get_time_bins,
    prepare_fold_data,
    prepare_for_model,
    load_processed_data,
    get_processed_folds,
    prepare_processed_fold_data,
)
from utils.logging_utils import (
    log,
    set_verbose,
    format_results,
    save_results,
)


def get_runner(model, cfg):
    """Get appropriate runner based on model type."""
    model_name = cfg.model.name
    if model_name == "logistic":
        return SKLearnRunner(cfg)
    else:
        return TorchRunner(cfg)


def model_name_from_classifier_type(classifier_type):
    """Get model name from classifier type."""
    mapping = {
        'logistic': 'Logistic Regression',
        'mlp': 'MLP',
        'cnn': 'CNN',
        'transformer': 'Transformer'
    }
    return mapping.get(classifier_type, classifier_type)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Setup
    data_source = cfg.get("data_source", "raw")
    set_verbose(cfg.get("verbose", True))
    log("Starting neuroprobe evaluation", priority=0)
    log(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}", priority=1)
    
    # Set random seeds
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))
    
    # Build components
    preprocessor = build_preprocessor(cfg.preprocessor)
    runner = get_runner(None, cfg)  # Model will be built per fold
    
    log(f"Using preprocessor: {cfg.preprocessor.name}", priority=0)
    log(f"Using model: {cfg.model.name}", priority=0)
    
    # Load data and run evaluation
    subject_id = cfg.subject_id
    trial_id = cfg.trial_id
    eval_name = cfg.eval_name
    
    if data_source == "processed":
        processed_data_path = cfg.get("processed_data_path")
        if not processed_data_path:
            raise ValueError("processed_data_path must be set when data_source=processed")
        run_processed_evaluation(cfg, preprocessor, runner, subject_id, trial_id, eval_name, processed_data_path)
    else:
        if hasattr(cfg, 'root_dir_braintreebank') and cfg.root_dir_braintreebank:
            os.environ['ROOT_DIR_BRAINTREEBANK'] = cfg.root_dir_braintreebank
        run_raw_evaluation(cfg, preprocessor, runner, subject_id, trial_id, eval_name)


def run_raw_evaluation(cfg, preprocessor, runner, subject_id, trial_id, eval_name):
    """Run evaluation with raw data."""
    from neuroprobe.braintreebank_subject import BrainTreebankSubject
    
    splits_type = cfg.splits_type
    
    # Load subject
    log(f"Loading subject {subject_id}, trial {trial_id}", priority=0)
    subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.float32)
    subset_electrodes(subject, lite=cfg.get("lite", True), nano=cfg.get("nano", False))
    
    start_time = time.time()
    subject.load_neural_data(trial_id)
    subject_load_time = time.time() - start_time
    log(f"Subject loaded in {subject_load_time:.2f} seconds", priority=0)
    
    # Create folds and time bins
    folds = create_folds(subject, trial_id, eval_name, splits_type, cfg)
    bin_starts, bin_ends = get_time_bins(cfg)
    log(f"Created {len(folds)} folds, {len(bin_starts)} time bins", priority=0)
    
    # Prepare save path
    preprocess_type = cfg.preprocessor.name
    classifier_type = cfg.model.name
    model_name = model_name_from_classifier_type(classifier_type)
    
    preprocess_suffix = f"{preprocess_type}" if preprocess_type != 'raw' else 'voltage'
    if 'stft' in preprocess_type:
        preprocess_suffix += f"_nperseg{cfg.preprocessor.get('nperseg', 512)}"
        preprocess_suffix += f"_poverlap{cfg.preprocessor.get('poverlap', 0.75)}"
        if cfg.preprocessor.get('window', 'hann') != 'hann':
            preprocess_suffix += f"_{cfg.preprocessor.window}"
        preprocess_suffix += f"_maxfreq{cfg.preprocessor.get('max_frequency', 150)}"
        if cfg.preprocessor.get('min_frequency', 0) != 0:
            preprocess_suffix += f"_minfreq{cfg.preprocessor.min_frequency}"
    
    save_dir = cfg.get("save_dir", "eval_results")
    splits_type_dir_map = {
        "WithinSession": "Within-Session",
        "CrossSession": "Cross-Session"
    }
    splits_type_dir = splits_type_dir_map.get(splits_type, splits_type.replace("_", "-"))
    file_save_dir = f"{save_dir}/{splits_type_dir}/{classifier_type}_{preprocess_suffix}"
    os.makedirs(file_save_dir, exist_ok=True)
    
    subject_identifier = f"btbank{subject_id}_{trial_id}"
    file_save_path = f"{file_save_dir}/population_{subject_identifier}_{eval_name}.json"
    if os.path.exists(file_save_path) and not cfg.get("overwrite", False):
        log(f"Skipping {file_save_path} because it already exists", priority=0)
        return
    
    # Evaluate across time bins and folds
    results_population = {"time_bins": []}
    regression_start_time = time.time()
    
    for bin_start, bin_end in zip(bin_starts, bin_ends):
        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }
        
        for fold_idx, fold in enumerate(folds):
            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}", priority=0)
            
            # Prepare data
            X_train, y_train, X_test, y_test = prepare_fold_data(
                fold, preprocessor, bin_start, bin_end, cfg, subject
            )
            X_train, X_test = prepare_for_model(X_train, X_test, classifier_type)
            
            # Build fresh model instance for each fold
            fold_model = build_model(cfg.model)
            
            # Train and evaluate
            fold_result = runner.run_fold(fold_model, X_train, y_train, X_test, y_test)
            bin_results["folds"].append(fold_result)
            
            log(f"Fold {fold_idx+1}: Test acc: {fold_result['test_accuracy']:.3f}, Test AUC: {fold_result['test_roc_auc']:.3f}", priority=0)
            
            del X_train, y_train, X_test, y_test, fold_model
            gc.collect()
        
        # Store results
        only_1second = cfg.get("only_1second", False)
        bins_start_before_word_onset_seconds = cfg.get("bins_start_before_word_onset_seconds", 0.5)
        bins_end_after_word_onset_seconds = cfg.get("bins_end_after_word_onset_seconds", 1.5)
        
        if (bin_start == -bins_start_before_word_onset_seconds and 
            bin_end == bins_end_after_word_onset_seconds and not only_1second):
            results_population["whole_window"] = bin_results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results
        else:
            results_population["time_bins"].append(bin_results)
    
    regression_run_time = time.time() - regression_start_time
    
    # Format and save results
    preprocess_parameters = OmegaConf.to_container(cfg.preprocessor, resolve=True)
    results = format_results(
        model_name=model_name,
        preprocess_type=preprocess_type,
        subject_id=subject_id,
        trial_id=trial_id,
        eval_name=eval_name,
        splits_type=splits_type,
        classifier_type=classifier_type,
        preprocess_parameters=preprocess_parameters,
        only_1second=cfg.get("only_1second", False),
        seed=cfg.get("seed", 42),
        results_population=results_population,
        subject_load_time=subject_load_time,
        regression_run_time=regression_run_time,
        author=cfg.get("author"),
        organization=cfg.get("organization"),
        organization_url=cfg.get("organization_url")
    )
    
    save_results(results, file_save_path)
    log("Evaluation complete!", priority=0)


def run_processed_evaluation(cfg, preprocessor, runner, subject_id, trial_id, eval_name, processed_data_path):
    """Run evaluation with processed data."""
    # Load processed data
    log(f"Loading processed data for subject {subject_id}, trial {trial_id}", priority=0)
    start_time = time.time()
    data, h5_file = load_processed_data(processed_data_path, subject_id, trial_id)
    data_load_time = time.time() - start_time
    log(f"Data loaded in {data_load_time:.2f} seconds", priority=0)
    
    try:
        # Get folds
        folds = get_processed_folds(data, eval_name)
        log(f"Found {len(folds)} pre-computed folds", priority=0)
        
        # Prepare save path
        preprocess_type = cfg.preprocessor.name
        classifier_type = cfg.model.name
        model_name = model_name_from_classifier_type(classifier_type)
        
        preprocess_suffix = f"{preprocess_type}" if preprocess_type != 'raw' else 'voltage'
        if 'stft' in preprocess_type:
            preprocess_suffix += f"_nperseg{cfg.preprocessor.get('nperseg', 512)}"
            preprocess_suffix += f"_poverlap{cfg.preprocessor.get('poverlap', 0.75)}"
            if cfg.preprocessor.get('window', 'hann') != 'hann':
                preprocess_suffix += f"_{cfg.preprocessor.window}"
            preprocess_suffix += f"_maxfreq{cfg.preprocessor.get('max_frequency', 150)}"
            if cfg.preprocessor.get('min_frequency', 0) != 0:
                preprocess_suffix += f"_minfreq{cfg.preprocessor.min_frequency}"
        
        save_dir = cfg.get("save_dir", "eval_results")
        file_save_dir = f"{save_dir}/Within-Session/{classifier_type}_{preprocess_suffix}_processed"
        os.makedirs(file_save_dir, exist_ok=True)
        
        subject_identifier = f"btbank{subject_id}_{trial_id}"
        file_save_path = f"{file_save_dir}/population_{subject_identifier}_{eval_name}.json"
        if os.path.exists(file_save_path) and not cfg.get("overwrite", False):
            log(f"Skipping {file_save_path} because it already exists", priority=0)
            return  # File will be closed in finally block
        
        # Evaluate across folds
        results_population = {"folds": []}
        regression_start_time = time.time()
        
        for fold in folds:
            fold_idx = fold['fold_idx']
            log(f"Fold {fold_idx+1}", priority=0)
            
            # Prepare data
            X_train, y_train, X_test, y_test = prepare_processed_fold_data(
                fold, data, preprocessor, eval_name, fold_idx
            )
            
            # Build fresh model instance for each fold
            fold_model = build_model(cfg.model)
            
            # Train and evaluate
            fold_result = runner.run_fold(fold_model, X_train, y_train, X_test, y_test)
            fold_result["fold_idx"] = fold_idx
            results_population["folds"].append(fold_result)
            
            log(f"Fold {fold_idx+1}: Test acc: {fold_result['test_accuracy']:.3f}, Test AUC: {fold_result['test_roc_auc']:.3f}", priority=0)
            
            del X_train, y_train, X_test, y_test, fold_model
            gc.collect()
        
        regression_run_time = time.time() - regression_start_time
        
        # Format and save results
        preprocess_parameters = OmegaConf.to_container(cfg.preprocessor, resolve=True)
        results = format_results(
            model_name=model_name,
            preprocess_type=preprocess_type,
            subject_id=subject_id,
            trial_id=trial_id,
            eval_name=eval_name,
            splits_type="WithinSession",
            classifier_type=classifier_type,
            preprocess_parameters=preprocess_parameters,
            only_1second=False,
            seed=cfg.get("seed", 42),
            results_population=results_population,
            subject_load_time=data_load_time,
            regression_run_time=regression_run_time,
            author=cfg.get("author"),
            organization=cfg.get("organization"),
            organization_url=cfg.get("organization_url")
        )
        
        save_results(results, file_save_path)
        log("Evaluation complete!", priority=0)
    
    finally:
        # Close h5py file
        h5_file.close()
        log(f"Closed h5py file", priority=2)


if __name__ == "__main__":
    main()
