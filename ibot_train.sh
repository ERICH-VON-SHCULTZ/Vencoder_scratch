export WANDB_API_KEY=8c90714e5d51c86fbf892d838bcc75e4d682a50f
export WANDB_INSECURE_DISABLE_SSL=true

python3 train_ibot.py \
  --wandb_project "ViT25start" \
  --data_dir "./data" \
  --output_dir "./checkpoints/ibot_vits_p8_100ep" \
  --batch_size 64 \
  --epochs 100 \
  --patch_size 8 \
  --lr 0.0005 \
  --min_lr 1e-6 \
  --weight_decay 0.04 \
  --weight_decay_end 0.4 \
  --warmup_epochs 10 \
  --momentum_teacher 0.999 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 10 \
  --clip_grad 3.0 \
  --global_crops_number 2 \
  --local_crops_number 6 \
  --drop_path_rate 0.1 \
  --out_dim 8192 \
  --norm_last_layer True \
  --save_freq 5 \
  --val_freq 5 \
  --num_workers 8 \
  --seed 42