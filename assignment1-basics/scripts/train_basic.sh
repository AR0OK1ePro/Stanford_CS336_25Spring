#!/bin/bash

uv run cs336_basics/train.py \
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
  --lr_schedule cosine \
  --lr_warmup_steps 2000 \
  --lr_decay_steps 20000 \
  --max_steps 20000 \
  --eval_interval 100 \
  --save_interval 5000 \
  --log_interval 100 \
  --checkpoint_dir "checkpoints/basic_training" \
  --train_data_path "data/TinyStoriesV2-GPT4-train.npy" \
  --val_data_path "data/TinyStoriesV2-GPT4-valid.npy" \
  --experiment_name "basic_transformer" \
  --wandb_project "CS336_assginment1_basic_training" \
  --seed 42