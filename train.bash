export WANDB_API_KEY=1b1a517bc2b443fb94e4051a52c52bad6412faee
export WANDB_INSECURE_DISABLE_SSL=true

python train.py \
    --data_dir ./data \
    --batch_size 128 \
    --epochs 100 \
    --out_dim 2048 \
    --lr 0.0005 \
    --min_lr 1e-7 \
    --local_crops_number 2 \
    --warmup_epochs 10 \
    --warmup_teacher_temp_epochs 30 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --momentum_teacher 0.996 \
    --weight_decay 0.04 \
    --weight_decay_end 0.2 \
    --clip_grad 3.0 \
    --num_workers 8 \
    --patch_size 8 \
    --wandb_project "vit-small-dino-official" \
    --val_freq 2 \
    --save_freq 4 \
    --knn_search \
    --evaluate


"
