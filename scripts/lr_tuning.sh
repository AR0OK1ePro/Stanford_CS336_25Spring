#!/bin/bash
uv run cs336_basics/training_together.py \
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
  --optimizer adamw \
  --lr_warmup_steps 1600 \
  --lr_decay_steps 20000 \
  --max_steps 20000 \
  --log_interval 10 \
  --eval_interval 10 \
  --save_interval 1000 \
  --checkpoint_dir "checkpoints/TinyStories_LM" \
  --train_data "data/TinyStoriesV2-GPT4-train.npy" \
  --val_data "data/TinyStoriesV2-GPT4-valid.npy" \
  --use_wandb \
  --wandb_project "CS336_assignment1" \
  --sweep \
  --sweep_method grid \
  --sweep_runs 5 \
    --sweep_param "learning_rate,log_uniform_values,1e-5,1e-2" \ \
  "$@"