#!/bin/bash

uv run cs336_basics/training_together.py \
  --d_model 512 \
  --num_heads 16 \
  --d_ff 1344 \
  --vocab_size 10000 \
  --num_layers 4 \
  --context_length 256 \
  --theta 10000 \
  --batch_size 32 \
  --max_steps 40000 \
  --lr_warmup_steps 2000 \
  --lr_decay_steps 40000 \
  --learning_rate 3e-4 \
  --optimizer adamw \
  --train_data "data/TinyStoriesV2-GPT4-train.npy" \
  --val_data "data/TinyStoriesV2-GPT4-valid.npy" \
  --checkpoint_dir "checkpoints/TinyStories_LM" \
  --log_interval 50 \
  --eval_interval 50 \
  --save_interval 2000 \
  --device auto \
  --dtype float32 \
  --use_wandb \
  "$@"