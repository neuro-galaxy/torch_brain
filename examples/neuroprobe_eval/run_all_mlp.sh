#!/bin/bash
# Run all MLP experiments
# Usage: ./run_all_mlp.sh [processed_data_path]

set -e

# Get processed data path from argument or use default
PROCESSED_DATA_PATH=${1:-/home/geeling/Projects/tb_buildathon/data/processed/neuroprobe_2025}

# Preprocessors to run
PREPROCESSORS="laplacian_stft" # raw stft 

# All tasks from neuroprobe config
TASKS="onset speech volume delta_volume" # onset speech volume delta_volume pitch word_index word_gap gpt2_surprisal word_head_pos word_part_speech word_length global_flow local_flow frame_brightness face_num

# All subject/trial combinations (Neuroprobe Lite)
SUBJECT_TRIALS=(
    "1 1"
    
)
# "1 2"
#     "2 0"
    # "2 4"
    # "3 0"
    # "3 1"
    # "4 0"
    # "4 1"
    # "7 0"
    # "7 1"
    # "10 0"
    # "10 1"

echo "=========================================="
echo "Running all Neuroprobe MLP experiments"
echo "=========================================="
echo "Processed data path: $PROCESSED_DATA_PATH"
echo "Preprocessors: $PREPROCESSORS"
echo "Tasks: $TASKS"
echo "Subject/Trials: ${#SUBJECT_TRIALS[@]} combinations"
echo "=========================================="
echo ""

# Run experiments
for prep in $PREPROCESSORS; do
    echo "=========================================="
    echo "Running experiments for preprocessor: $prep"
    echo "=========================================="
    
    for task in $TASKS; do
        for subject_trial in "${SUBJECT_TRIALS[@]}"; do
            read -r subject trial <<< "$subject_trial"
            
            echo "Running: $task (subject $subject, trial $trial) - $prep"

            echo "Running: MLP"
            python run_eval.py \
                preprocessor=$prep \
                eval_name=$task \
                subject_id=$subject \
                trial_id=$trial \
                model=mlp \
                data_source=processed \
                processed_data_path=$PROCESSED_DATA_PATH \
                verbose=false \
                overwrite=false || echo "Warning: Failed for $task (sub $subject, trial $trial)"
        done
    done
done

# # Aggregate results only if directory exists
# if [ -d "eval_results/Within-Session" ] || [ -d "eval_results/Cross-Session" ]; then
#     echo ""
#     echo "=========================================="
#     echo "Aggregating results"
#     echo "=========================================="
    
#     python aggregate_results.py \
#         --results-dir eval_results \
#         --split-type all \
#         --task all \
#         --to-dataframe \
#         --output-csv eval_results/results.csv \
#         --print-summary
# else
#     echo ""
#     echo "No results found to aggregate (directory does not exist yet)"
# fi

# echo ""
# echo "=========================================="
# echo "All experiments complete!"
# echo "=========================================="

