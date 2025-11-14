#!/usr/bin/env python3
"""
Aggregate per-session evaluation results into per-task files for Neuroprobe leaderboard submission.

Also supports exporting to DataFrame format matching the notebook.
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_results_file(file_path):
    """Load a results JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_results_from_json(file_path):
    """Load fold-level results from a JSON file for DataFrame export."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    eval_results = data.get("evaluation_results", {})
    
    for subject_key, subject_data in eval_results.items():
        if subject_key.startswith("btbank"):
            parts = subject_key.replace("btbank", "").split("_")
            if len(parts) == 2:
                subject_id = int(parts[0])
                trial_id = int(parts[1])
            else:
                continue
        else:
            continue
        
        population_results = subject_data.get("population", {})
        
        # Check if we have folds directly or nested in time_bins/one_second_after_onset
        if "folds" in population_results:
            folds = population_results["folds"]
            for fold_idx, fold_result in enumerate(folds):
                results.append({
                    "model_name": data.get("model_name", "Unknown"),
                    "eval_name": data.get("config", {}).get("eval_name", "unknown"),
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "fold_idx": fold_result.get("fold_idx", fold_idx),
                    "train_acc": fold_result.get("train_accuracy", np.nan),
                    "test_acc": fold_result.get("test_accuracy", np.nan),
                    "test_auc": fold_result.get("test_roc_auc", np.nan),
                })
        elif "one_second_after_onset" in population_results:
            one_second = population_results["one_second_after_onset"]
            folds = one_second.get("folds", [])
            for fold_idx, fold_result in enumerate(folds):
                results.append({
                    "model_name": data.get("model_name", "Unknown"),
                    "eval_name": data.get("config", {}).get("eval_name", "unknown"),
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "fold_idx": fold_idx,
                    "train_acc": fold_result.get("train_accuracy", np.nan),
                    "test_acc": fold_result.get("test_accuracy", np.nan),
                    "test_auc": fold_result.get("test_roc_auc", np.nan),
                })
    
    return results


def aggregate_results(results_dir, splits_type, task_name, output_dir=None):
    """
    Aggregate per-session results into per-task files.
    
    Args:
        results_dir: Base directory containing results
        splits_type: Split type (Within-Session, Cross-Session, or "all")
        task_name: Task name (e.g., "onset", "gpt2_surprisal") or "all"
        output_dir: Output directory (defaults to results_dir)
    
    Returns:
        Dictionary mapping task names to aggregated results
    """
    if output_dir is None:
        output_dir = results_dir
    
    if splits_type == "all":
        split_dirs = ["Within-Session", "Cross-Session"]
    else:
        split_dirs = [splits_type]
    
    aggregated = {}
    
    for split_dir in split_dirs:
        split_path = Path(results_dir) / split_dir
        if not split_path.exists():
            print(f"Warning: Split directory not found: {split_path}")
            continue
        
        for model_dir in split_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            for result_file in model_dir.glob("population_btbank*_*.json"):
                parts = result_file.stem.replace("population_", "").split("_")
                if len(parts) < 3:
                    continue
                
                task = parts[-1]
                
                if task_name != "all" and task != task_name:
                    continue
                
                try:
                    results = load_results_file(result_file)
                    
                    if task not in aggregated:
                        aggregated[task] = {
                            "model_name": results.get("model_name", "Unknown"),
                            "author": results.get("author", "Unknown"),
                            "description": results.get("description", ""),
                            "organization": results.get("organization", "Unknown"),
                            "organization_url": results.get("organization_url", ""),
                            "timestamp": results.get("timestamp", 0),
                            "evaluation_results": {}
                        }
                    
                    if "evaluation_results" in results:
                        aggregated[task]["evaluation_results"].update(results["evaluation_results"])
                    
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
                    continue
    
    # Save aggregated results
    for task, aggregated_result in aggregated.items():
        output_path = Path(output_dir) / split_dirs[0] / f"population_{task}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(aggregated_result, f, indent=4)
        
        print(f"Saved aggregated results: {output_path}")
        print(f"  Sessions included: {len(aggregated_result['evaluation_results'])}")
    
    return aggregated


def aggregate_to_dataframe(results_dir, model_name=None, preprocessor_name=None):
    """
    Aggregate all JSON results into a DataFrame (notebook format).
    
    Args:
        results_dir: Directory containing JSON results files
        model_name: Optional filter by model name
        preprocessor_name: Optional filter by preprocessor name
    
    Returns:
        pandas.DataFrame with columns: model_name, eval_name, subject_id, trial_id, fold_idx, train_acc, test_acc, test_auc
    """
    results_dir = Path(results_dir)
    all_results = []
    
    json_files = list(results_dir.rglob("population_*.json"))
    
    for json_file in json_files:
        try:
            if preprocessor_name:
                path_str = str(json_file)
                if preprocessor_name not in path_str:
                    continue
            
            fold_results = load_results_from_json(json_file)
            all_results.extend(fold_results)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    if model_name:
        df = df[df["model_name"] == model_name]
    
    return df


def print_model_results(results_df):
    """Print summary statistics matching the notebook format."""
    if results_df.empty:
        print("No results to display")
        return
    
    print('-' * 100)
    grouped = results_df.groupby('eval_name', as_index=False)
    task_means = grouped[['train_acc', 'test_acc', 'test_auc']].mean(numeric_only=True)
    auc_sem = grouped['test_auc'].sem().rename(columns={'test_auc': 'test_auc_sem'})
    task_means['test_auc_sem'] = auc_sem['test_auc_sem']
    print(task_means)
    print('-' * 100)
    print(f"Mean train accuracy: {task_means['train_acc'].mean():.4f}")
    print(f"Mean test accuracy: {task_means['test_acc'].mean():.4f}")
    
    mean_auc = task_means['test_auc'].mean()
    sem_auc = (task_means['test_auc_sem']**2).sum()**0.5 / len(task_means)
    print(f"Mean test AUC: {mean_auc:.4f} ± {sem_auc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-session evaluation results into per-task files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval_results",
        help="Base directory containing results (default: eval_results)"
    )
    parser.add_argument(
        "--split-type",
        type=str,
        choices=["Within-Session", "Cross-Session", "all"],
        default="all",
        help="Split type to aggregate (default: all)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Task name to aggregate (e.g., 'onset', 'gpt2_surprisal') or 'all' (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (defaults to results-dir)"
    )
    parser.add_argument(
        "--to-dataframe",
        action="store_true",
        help="Also export results to DataFrame CSV format (notebook format)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        help="CSV file path for DataFrame export (default: results_dir/results.csv)"
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary statistics for DataFrame export"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Aggregating Neuroprobe Evaluation Results")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Split type: {args.split_type}")
    print(f"Task: {args.task}")
    print("="*80)
    
    # Aggregate to leaderboard format
    aggregated = aggregate_results(
        args.results_dir,
        args.split_type,
        args.task,
        args.output_dir
    )
    
    if aggregated:
        print(f"\nSuccessfully aggregated {len(aggregated)} task(s)")
    else:
        print("\nNo results found to aggregate")
    
    # Export to DataFrame if requested
    if args.to_dataframe:
        print("\n" + "="*80)
        print("Exporting to DataFrame format")
        print("="*80)
        
        df = aggregate_to_dataframe(args.results_dir)
        
        if not df.empty:
            csv_path = args.output_csv or os.path.join(args.results_dir, "results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved DataFrame to {csv_path}")
            print(f"Found {len(df)} fold results")
            
            if args.print_summary:
                print_model_results(df)
        else:
            print("No results found for DataFrame export")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
