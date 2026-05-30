#!/bin/bash
#SBATCH --job-name=single_gpu_mila
#SBATCH --output=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_output_%j.txt
#SBATCH --error=/home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/slurm_outputs/slurm_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=main
#
#set -e
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
snakemake --rerun-triggers=mtime --cores 4 allen_brain_observatory_calcium_unfreeze
nvidia-smi

#srun --export=ALL,WANDB_PROJECT=allen_bo_calcium python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train.py \
#        --config-name train_allen_bo.yaml data_root=$SLURM_TMPDIR/uncompressed name=allen_bo_single

#### AUTO RUN MULTI_SESSIONS

#vis_l_sess_ids=(
        #"\"662982346\"" "\"645689073\"" "\"613091721\"" "\"502793808\"" "\"501940850\"" 
        #"\"546641574\"" "\"614556106\"" "\"511194579\"" "\"698762886\"" "\"607063420\"" 
        #"\"688580172\"" "\"651770186\"" "\"652737678\"" "\"603452291\"" "\"715923832\"" 
        #"\"627823636\"" "\"662358771\"" "\"662348804\"" "\"606802468\"" "\"670395725\"" 
        #"\"601887677\"" "\"552427971\"" "\"657391625\"" "\"506823562\"" 
        #"\"611658482\"" 
        #"\"674276329\"" "\"557848210\"" "\"612549085\"" "\"601910964\"" "\"603187982\"" 
        #"\"645256361\"" "\"673475020\"" "\"657224241\"" "\"573850303\"" "\"657009581\"" 
        #"\"576273468\"" "\"647143225\"" "\"602866800\"" "\"601805379\"" "\"707923645\"" 
        #"\"529688779\"" "\"686442556\"" "\"651769499\"" "\"558476282\"" "\"601841437\"" 
        #"\"639932847\"" "\"665726618\"" "\"669237515\"" "\"596509886\"" "\"507129766\"" 
        #"\"550455111\"" "\"686449092\"" "\"585035184\"" "\"560578599\"" "\"614571626\"" 
        #"\"597028938\"" "\"605883133\"" "\"581153070\"" "\"657915168\"" "\"560926639\"" 
        #"\"653551965\"" "\"662219852\"" "\"511595995\"" "\"654532828\"" "\"664914611\"" 
        #"\"644051974\"" "\"652737867\"" "\"652092676\"" "\"552410386\"" "\"623339221\"" 
        #"\"506809539\"" "\"556321897\"" "\"646016204\"" "\"595808594\"" "\"667004159\"" 
        #"\"647595665\"" "\"562122508\"" "\"572606382\"" "\"699155265\"" "\"623587006\"" 
        #"\"582867147\"" "\"584983136\"" "\"682051855\"" "\"580095655\"" "\"509958730\"" 
        #"\"511573879\"" "\"603978471\"" "\"584544569\"" "\"672207947\"" "\"576001843\"" 
        #"\"507990552\"" "\"501929610\"" "\"573261515\"" "\"682049099\"" "\"583136567\"" 
        #"\"567878987\"" "\"676024666\"" "\"564425777\"" 
        #"\"653123929\""
        #)

vis_am_sess_ids=(
        #"\"556353209\"" "\"595273803\"" "\"556344224\"" "\"570305847\"" "\"566307038\"" 
        #"\"638262558\"" "\"552760671\"" "\"565698388\"" "\"616779893\"" "\"637669284\"" 
        #"\"562711440\"" "\"569792817\"" "\"5f69457162\"" "\"707006626\"" "\"647603932\"" 
        #"\"569718097\"" "\"556665481\"" "\"571177441\"" "\"712919665\"" "\"560027980\"" 
        #"\"652094917\"" "\"566458505\"" "\"550851591\"" "\"613599811\"" "\"557304694\"" 
        #"\"578674360\"" "\"642884591\"" "\"576411246\"" "\"611638995\"" "\"551834174\"" 
        #"\"569739027\"" "\"601904502\"" "\"605606109\"" "\"575302108\"" "\"565216523\""
)

tuning_sess_ids=(
        #"\"644026238\"" 
        #"\"662348804\"" "\"613968705\"" 
        "\"578674360\""
        #"\"667004159\"" "\"565216523\"" 
)

patch_sizes=(
        2
        # 5 10 30
)
cd /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo

#for p_sz in "${patch_sizes[@]}"
#do
for sess_id in "${tuning_sess_ids[@]}"
do
        sortset=$sess_id
#                
        echo $sess_id
        srun --export=ALL,WANDB_PROJECT=allen_bo_calcium python /home/mila/x/xuejing.pan/POYO/project-kirby/examples/capoyo/train.py \
        --config-name train_allen_bo.yaml name=$sortset ++dataset.0.selection.0.sortset=$sortset log_dir=/home/mila/x/xuejing.pan/scratch/poyo_logs
done
#done
#data_root=$SLURM_TMPDIR/uncompressed