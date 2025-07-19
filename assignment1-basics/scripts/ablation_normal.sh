#!/bin/bash

uv run cs336_basics/ablation_experiments/train_ablation.py \
  --d_model 512 \
  --num_heads 16 \
  --d_ff 1344 \
  --vocab_size 10000 \
  --num_layers 4 \
  --context_length 256 \
  --theta 10000 \
  --device auto \
  --dtype float32 \
  --batch_size 64 \
  --learning_rate 5e-3 \
  --optimizer adamw \
  --lr_warmup_steps_ratio 0.1 \
  --lr_decay_steps_ratio 1 \
  --max_tokens 327680000 \
  --log_num 2000 \
  --eval_num 2000 \
  --save_num 5 \
  --checkpoint_dir "checkpoints/TinyStories_LM" \
  --train_data_path "data/TinyStoriesV2-GPT4-train.npy" \
  --val_data_path "data/TinyStoriesV2-GPT4-valid.npy" \
  --wandb_project "CS336_assignment1_ablation" \