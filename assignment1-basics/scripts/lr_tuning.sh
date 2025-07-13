#!/bin/bash

# 超参数列表
learning_rates=(1e-1 5e-2 1e-2 5e-3 1e-3)

for lr in "${learning_rates[@]}"; do
  echo "Training with learning_rate = $lr"

  uv run cs336_basics/training_together.py \
    --d_model 512 \
    --num_heads 16 \
    --d_ff 1344 \
    --vocab_size 10000 \
    --num_layers 4 \
    --context_length 256 \
    --theta 10000 \
    --batch_size 64 \
    --max_steps 20000 \
    --lr_warmup_steps 500 \
    --lr_decay_steps 20000 \
    --learning_rate $lr \
    --optimizer adamw \
    --train_data "data/TinyStoriesV2-GPT4-train.npy" \
    --val_data "data/TinyStoriesV2-GPT4-valid.npy" \
    --checkpoint_dir "checkpoints/lr_${lr}" \
    --log_interval 50 \
    --eval_interval 50 \
    --save_interval 1000 \
    --device auto \
    --dtype float32 \
    --use_wandb \
    --wandb_project "CS336_assignment1" \
    --wandb_run "lr:${lr} bs:64 steps:20k"
done