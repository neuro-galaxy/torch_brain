#!/bin/bash
#SBATCH -GA100:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00

cd ..

file_path="/home/hice1/aandre8/scratch/alex/results"
files=(
    block_results.txt
    choice_results.txt
    wheel_results.txt
    whisker_results.txt
)

for file in "${files[@]}"; do
    f="$file_path/$file"
    if [ -f "$f" ]; then
        echo "Removing existing file: $f"
        rm "$f"
    fi
    echo "Creating $f"
    touch "$f"
done

INPUT_FILE="eval_cfg.csv"
exec 3< "$INPUT_FILE"

while IFS=',' read -r session task save_file postfix <&3; do
    echo "Processing session: $session, task: $task"
    srun --ntasks=1 --exclusive python eval_ndt2_single_task.py \
    --eid "$session" --behavior "$task" --save_file "${file_path}/${save_file}" \
    --ckpt_postfix=$postfix

done < "$INPUT_FILE"

wait