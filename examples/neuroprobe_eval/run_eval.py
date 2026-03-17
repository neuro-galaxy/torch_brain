"""
Main evaluation script for neuroprobe using Hydra configuration.
"""
import sys
import os
os.environ['ROOT_DIR_BRAINTREEBANK'] = ''

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
import gc
import time

from models import build_model
from preprocessors import build_preprocessor
from sklearn_runner import SKLearnRunner
from torch_runner import TorchRunner
from utils.data_loader import (
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
import neuroprobe.config as neuroprobe_config



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


def reshape_data_for_model(X, preprocessor, n_channels, model_name):
    """
    Reshape flattened data back to spatial structure for CNN/Transformer models.
    
    Args:
        X: Flattened data array of shape (n_samples, flattened_features)
        preprocessor: Preprocessor instance
        n_channels: Number of channels/electrodes
        model_name: Model name ('cnn', 'transformer', 'mlp', 'logistic')
    
    Returns:
        Reshaped data array
    """
    # Only reshape for CNN and Transformer models
    if model_name not in ['cnn', 'transformer']:
        return X
    
    preprocess_type = preprocessor.cfg.name
    
    if 'stft' in preprocess_type:
        # STFT preprocessing: CNN expects (n_samples, n_channels, n_freqs, n_timebins)
        # This matches (channels, freq, time) format expected by Conv2d
        # Calculate dimensions from preprocessor config
        nperseg = preprocessor.cfg.get("nperseg", 512)
        poverlap = preprocessor.cfg.get("poverlap", 0.75)
        min_freq = preprocessor.cfg.get("min_frequency", 0)
        max_freq = preprocessor.cfg.get("max_frequency", 150)
        sampling_rate = preprocessor.cfg.get("sampling_rate", neuroprobe_config.SAMPLING_RATE)
        
        # Calculate n_freqs
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
        n_freqs = int(np.sum((freqs >= min_freq) & (freqs <= max_freq)))
        
        # Calculate n_timebins from flattened size
        # flattened_size = n_samples * n_channels * n_timebins * n_freqs
        # So: n_timebins = flattened_size / (n_samples * n_channels * n_freqs)
        n_samples = X.shape[0]
        flattened_size_per_sample = X.shape[1]
        n_timebins = flattened_size_per_sample // (n_channels * n_freqs)
        
        # Reshape to (n_samples, n_channels, n_timebins, n_freqs) first
        # Then transpose to (n_samples, n_channels, n_freqs, n_timebins) for CNN
        X_reshaped = X.reshape(n_samples, n_channels, n_timebins, n_freqs)
        # Transpose to match CNN expectation: (channels, freq, time)
        X_reshaped = np.transpose(X_reshaped, (0, 1, 3, 2))  # (n_samples, n_channels, n_freqs, n_timebins)
        return X_reshaped
    else:
        # Raw preprocessing: shape should be (n_samples, n_channels, n_time)
        # Calculate n_time from flattened size
        n_samples = X.shape[0]
        flattened_size_per_sample = X.shape[1]
        n_time = flattened_size_per_sample // n_channels
        
        # Reshape to (n_samples, n_channels, n_time)
        X_reshaped = X.reshape(n_samples, n_channels, n_time)
        return X_reshaped


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Setup
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
    
    processed_data_path = cfg.get("processed_data_path")
    if not processed_data_path:
        raise ValueError("processed_data_path must be set in config or via command line")
    
    run_processed_evaluation(cfg, preprocessor, runner, subject_id, trial_id, eval_name, processed_data_path)


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
        
        save_dir = cfg.get("save_dir", "eval_results")
        file_save_dir = f"{save_dir}/Within-Session/{classifier_type}_{preprocess_type}"
        os.makedirs(file_save_dir, exist_ok=True)
        
        subject_identifier = f"btbank{subject_id}_{trial_id}"
        file_save_path = f"{file_save_dir}/population_{subject_identifier}_{eval_name}.json"
        if os.path.exists(file_save_path) and not cfg.get("overwrite", False):
            log(f"Skipping {file_save_path} because it already exists", priority=0)
            return  # File will be closed in finally block
        
        # Evaluate across folds
        results_population = {
            "one_second_after_onset": {
                "time_bin_start": 0.0,
                "time_bin_end": 1.0,
                "folds": []
            }
        }
        regression_start_time = time.time()
        
        for fold in folds:
            fold_idx = fold['fold_idx']
            log(f"Fold {fold_idx+1}", priority=0)
            
            # Prepare data (returns train, val, test splits)
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_processed_fold_data(
                fold, data, preprocessor, eval_name, fold_idx
            )
            
            # Get number of channels for reshaping (needed for CNN/Transformer)
            split_included_channels_train = getattr(
                data.channels, f'included_{eval_name}_fold{fold_idx}_train'
            )
            split_included_channels_test = getattr(
                data.channels, f'included_{eval_name}_fold{fold_idx}_test'
            )
            n_channels_train = int(np.sum(split_included_channels_train))
            n_channels_test = int(np.sum(split_included_channels_test))
            
            # Reshape data for CNN/Transformer models (preserve spatial structure)
            X_train = reshape_data_for_model(X_train, preprocessor, n_channels_train, classifier_type)
            X_val = reshape_data_for_model(X_val, preprocessor, n_channels_test, classifier_type)
            X_test = reshape_data_for_model(X_test, preprocessor, n_channels_test, classifier_type)
            
            # Build fresh model instance for each fold
            fold_model = build_model(cfg.model)
            
            # Train and evaluate
            # Pass all 3 splits to torch models, only train/test to sklearn models
            if cfg.model.name == "logistic":
                fold_result = runner.run_fold(fold_model, X_train, y_train, X_test, y_test)
            else:
                fold_result = runner.run_fold(fold_model, X_train, y_train, X_val, y_val, X_test, y_test)
            
            fold_result["fold_idx"] = fold_idx
            results_population["one_second_after_onset"]["folds"].append(fold_result)
            
            log(f"Fold {fold_idx+1}: Test acc: {fold_result['test_accuracy']:.3f}, Test AUC: {fold_result['test_roc_auc']:.3f}", priority=0)
            
            del X_train, y_train, X_val, y_val, X_test, y_test, fold_model
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
