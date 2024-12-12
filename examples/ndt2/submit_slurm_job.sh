source config.env
source ${VENV_PATH}

cd slurm
job=SlurmQualiRes # change for your job located in the slurm folder
sbatch \
    -J IBL-NDT2 \
    -o ${LOG_DIR}/${job}.out \
    -e ${LOG_DIR}/${job}.err \
    ${job}.sbatch