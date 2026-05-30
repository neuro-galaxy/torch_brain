#!/bin/bash
#SBATCH --job-name=long_mila
#SBATCH --output=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_output_%j.txt
#SBATCH --error=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=32:00:00
#SBATCH --partition=long

dataset=allen_brain_observatory_calcium
export WANDB_PROJECT=allen_bo_calcium

module load cuda/11.2/nccl/2.8
module load cuda/11.2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA

module load miniconda
conda activate poyo_conda

# wandb credentials
set -a
source .env
set +a

cd /home/mila/x/xuejing.pan/POYO/project-kirby

# Uncompress the data to SLURM_TMPDIR single node
snakemake --rerun-triggers=mtime -c1 allen_brain_observatory_calcium_unfreeze
nvidia-smi
tuning_sess_ids=(
        "\"644026238\"" 
        "\"662348804\"" "\"613968705\"" 
        #"\"578674360\"" "\"667004159\"" "\"565216523\"" 
)
#cd /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo
cd /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/poyo_hparam_sweep
#wandb agent neuro-galaxy/allen_bo_calcium/tws82q04
# Loop through each session ID
#for sess_id in "${tuning_sess_ids[@]}"
#do
    # Create the sweep and capture the sweep ID
    #sweep_output=wandb sweep --name "565216523" wandb_sweep.yaml

    # Extract the sweep ID from the output
    #sweep_id=$(echo "$sweep_output" | grep -oP '(?<="id": ")[^"]+')
    #echo $sweep_id

    # Run the wandb agent with the extracted sweep ID
    #wandb agent --entity neuro-galaxy --project allen_bo_calcium --sweep_id $sweep_id
#done
wandb agent neuro-galaxy/allen_bo_calcium/m2ossjrh
