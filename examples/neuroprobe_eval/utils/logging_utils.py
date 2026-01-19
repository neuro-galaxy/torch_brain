"""
Logging and result formatting utilities.
"""
import time
import psutil
import torch
import json
import os


verbose = True  # Global verbose flag


def set_verbose(value):
    """Set global verbose flag."""
    global verbose
    verbose = value


def log(message, priority=0, indent=0):
    """
    Log a message with timestamp and resource usage.
    
    Args:
        message: Message to log
        priority: Priority level (higher = less important)
        indent: Indentation level
    """
    global verbose
    max_log_priority = -1 if not verbose else 4
    if priority > max_log_priority:
        return
    
    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


def format_results(model_name, preprocess_type, subject_id, trial_id, eval_name, splits_type,
                   classifier_type, preprocess_parameters, only_1second, seed, results_population,
                   subject_load_time, regression_run_time, author=None, organization=None, organization_url=None):
    """
    Format evaluation results into a dictionary compatible with Neuroprobe leaderboard format.
    
    Args:
        model_name: Name of the model
        preprocess_type: Type of preprocessing
        subject_id: Subject ID
        trial_id: Trial ID
        eval_name: Evaluation name
        splits_type: Type of splits
        classifier_type: Type of classifier
        preprocess_parameters: Preprocessing parameters dict
        only_1second: Whether only 1 second was evaluated
        seed: Random seed
        results_population: Population results dict
        subject_load_time: Time to load subject
        regression_run_time: Time for regression run
        author: Author name (optional, defaults to "Andrii Zahorodnii")
        organization: Organization name (optional, defaults to "MIT")
        organization_url: Organization URL (optional, defaults to "https://azaho.org/")
    
    Returns:
        Formatted results dictionary
    """
    # Use provided metadata or defaults
    if author is None:
        author = "Andrii Zahorodnii"
    if organization is None:
        organization = "MIT"
    if organization_url is None:
        organization_url = "https://azaho.org/"
    
    # Format subject identifier as btbank{subject_id}_{trial_id} per Neuroprobe format
    subject_identifier = f"btbank{subject_id}_{trial_id}"
    
    results = {
        "model_name": model_name,
        "author": author,
        "description": f"Simple {model_name} using all electrodes ({preprocess_type if preprocess_type != 'none' else 'voltage'}).",
        "organization": organization,
        "organization_url": organization_url,
        "timestamp": time.time(),
        
        "evaluation_results": {
            subject_identifier: {
                "population": results_population
            }
        },
        
        "config": {
            "preprocess": preprocess_parameters,
            "only_1second": only_1second,
            "seed": seed,
            "subject_id": subject_id,
            "trial_id": trial_id,
            "splits_type": splits_type,
            "classifier_type": classifier_type,
        },
        
        "timing": {
            "subject_load_time": subject_load_time,
            "regression_run_time": regression_run_time,
        }
    }
    
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
        json.dump(results, f, indent=4)
    log(f"Results saved to {file_path}", priority=0)

