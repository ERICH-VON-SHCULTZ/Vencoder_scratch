#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=120GB
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --job-name=pt_small
#SBATCH --output=./pretrain_small.out
#SBATCH --error=./pretrain_small.err


module purge

singularity exec --nv --overlay /scratch/qz2086/sing-deeplearning/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif /bin/bash -c "

source /ext3/env.sh

conda activate dl_310

export WANDB_API_KEY=1b1a517bc2b443fb94e4051a52c52bad6412faee
export WANDB_INSECURE_DISABLE_SSL=true

python train.py \
    --data_dir /scratch/qz2086/fall2025_finalproject/data \
    --batch_size 384 \
    --epochs 20 \
    --out_dim 2048 \
    --lr 0.00005 \
    --min_lr 1e-6 \
    --local_crops_number 2 \
    --warmup_epochs 1 \
    --warmup_teacher_temp_epochs 2 \
    --teacher_temp 0.1 \
    --warmup_teacher_temp 0.1 \
    --momentum_teacher 0.996 \
    --weight_decay 0.04 \
    --weight_decay_end 0.1 \
    --clip_grad 3.0 \
    --num_workers 8 \
    --patch_size 8 \
    --wandb_project "vit-small-finally-working" \
    --val_freq 2 \
    --save_freq 3 \
    --knn_search \
    --evaluate


"
