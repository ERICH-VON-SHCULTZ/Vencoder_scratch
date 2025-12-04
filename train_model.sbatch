#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=32:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=ibot
#SBATCH --output=./ibot.out


module purge
cd /scratch/yw8866/Vencoder_scratch
OVERLAY_FILE=/scratch/yw8866/sftenv/overlay-15GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv \
        --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; bash ibot_train.sh"