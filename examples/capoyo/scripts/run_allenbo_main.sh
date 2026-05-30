#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=main
#
#set -e
#dataset=allen_brain_observatory_calcium
dataset=allen_brain_observatory_calcium
export WANDB_PROJECT=allen_bo_calcium

module load anaconda/3
module load cuda/11.2/nccl/2.8
module load cuda/11.2
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA

#conda activate poyo
source $HOME/poyo_env/bin/activate
# wandb credentials
set -a
source .env
set +a

cd /home/mila/x/xuejing.pan/POYO/project-kirby

# Uncompress the data to SLURM_TMPDIR single node
snakemake --rerun-triggers=mtime -c1 allen_brain_observatory_calcium_unfreeze
nvidia-smi

#srun --export=ALL,WANDB_PROJECT=allen_bo_calcium python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train.py \
#        --config-name train_allen_bo.yaml data_root=$SLURM_TMPDIR/uncompressed name=allen_bo_single

#### AUTO RUN MULTI_SESSIONS
vis_p_sess_ids=(
        "\"627823695\"" "\"652094901\"" "\"545446482\"" "\"662361096\"" "\"645086975\"" 
        "\"657078119\"" "\"581150104\"" "\"644026238\"" "\"652091264\"" "\"613968705\"" 
        "\"647155122\"" "\"657082055\"" "\"652842495\"" "\"692345003\"" "\"645413759\"" 
        "\"710504563\"" "\"598635821\"" "\"657016267\"" "\"612543999\"" "\"540684467\"" 
        "\"502205092\"" "\"680156911\"" "\"623347352\"" "\"539487468\"" "\"604145810\"" 
        "\"661437140\"" "\"612044635\"" "\"710778377\"" "\"688678766\"" "\"510517131\"" 
        "\"583279803\"" "\"580163817\"" "\"501021421\"" "\"653125130\"" "\"652842572\"" 
        "\"689388034\"" "\"539497234\"" "\"582918858\"" "\"606353987\"" "\"511534603\"" 
        "\"672211004\"" "\"539290504\"" "\"702934964\"" "\"530645663\"" "\"526504941\"" 
        "\"561312435\"" "\"664404274\"" "\"577379202\"" "\"580043440\"" "\"573720508\"" 
        "\"674679019\"" "\"508753256\"" "\"547388708\"" "\"575939366\"" "\"593373156\"" 
        "\"502962794\"" "\"657390171\"" "\"512270518\"" "\"587344053\"" "\"671618887\"" 
        "\"712178483\"" "\"643645390\"" "\"663485329\"" "\"659491419\"" "\"502115959\"" 
        "\"601423209\"" "\"528402271\"" "\"649401936\"" "\"546716391\"" "\"510214538\"" 
        "\"662033243\"" "\"502608215\"" "\"704298735\"" "\"663479824\"" "\"527048992\"" 
        "\"653932505\"" "\"592407200\"" "\"643592303\"" "\"503109347\"" "\"609894681\"" 
        "\"670728674\"" "\"596584192\"" "\"531134090\"" "\"669861524\"" "\"675477919\"" 
        "\"590168385\"" "\"571137446\"" "\"637671554\"" "\"595263154\"" "\"657389972\"" 
        "\"584196534\"" "\"660513003\"" "\"603576132\"" "\"671164733\"" "\"658518486\"" 
        "\"661328410\"" "\"596779487\"" "\"676503588\"" "\"672206735\"" "\"657650110\"" 
        "\"501729039\"" "\"590047029\"" "\"570278597\"" "\"650079244\"" "\"585900296\"" 
        "\"637998955\"" "\"653122667\"" "\"598564173\"" "\"581026088\"" "\"670395999\"" 
        "\"637669270\"" "\"679702884\"" "\"665722301\"" "\"680150733\"" "\"541290571\"" 
        "\"541010698\"" "\"603224878\"" "\"683257169\"" "\"595806300\"" "\"617395455\"" 
        "\"508356957\"" "\"576095926\"" "\"510514474\"" "\"712178511\"" "\"501574836\"" 
        "\"657080632\"" "\"649938038\"" "\"673914981\"" "\"559192380\"" "\"524691284\"" 
        "\"617381605\"" "\"673171528\"" "\"662974315\"" "\"657775947\"" "\"637115675\""
)


for sess_id in "${vis_p_sess_ids[@]}"
do
        completion_flag="/home/mila/x/xuejing.pan/scratch/completion_flags/completed_${sess_id}.txt"

        # Check if the session has already been completed
        if [ -f "$completion_flag" ]; then
                echo "Session $sess_id has already been completed."
                continue
        fi
        echo "Running session $sess_id"

        sortset=$sess_id
        echo $sess_id
        srun --export=ALL,WANDB_PROJECT=allen_bo_calcium python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train.py \
        --config-name train_allen_bo.yaml data_root=$SLURM_TMPDIR/uncompressed name=$sortset ++dataset.0.selection.0.sortset=$sortset

        touch "$completion_flag"
done
