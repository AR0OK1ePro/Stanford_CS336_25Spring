#!/bin/bash

uv run cs336_basics/train.py \
  --d_model 128 \
  --num_heads 4 \
  --d_ff 320 \
  --vocab_size 10000 \
  --num_layers 2 \
  --context_length 128 \
  --theta 10000 \
  --batch_size 16 \
  --max_steps 2000 \
  --lr_warmup_steps 10 \
  --lr_decay_steps 2000 \
  --learning_rate 1e-0 \
  --optimizer adamw \
  --train_data "data/TinyStoriesV2-GPT4-train.npy" \
  --val_data "data/TinyStoriesV2-GPT4-valid.npy" \
  --checkpoint_dir "checkpoints/single_minibatch" \
  --log_interval 10 \
  --eval_interval 20000 \
  --save_interval 20000 \
  --device auto \
  --dtype float32 \
  --single_minibatch \
  "$@"