#!/usr/bin/env python3
"""
Main training script for transformer language model.
Supports configurable hyperparameters, efficient data loading, checkpointing, and logging.
"""

import os
import sys
import argparse
import time
import math
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
import logging
import wandb

# Import our custom modules
from cs336_basics.transformer import transformer_lm
from cs336_basics.train_transformer import (
    cross_entropy, SGD, AdamW, learning_rate_schedule, 
    gradient_clipping, save_checkpoint, load_checkpoint, data_loading
)

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048
    vocab_size: int = 50304
    num_layers: int = 6
    context_length: int = 1024
    theta: float = 10000.0
    device: str = "auto"
    dtype: str = "float32"

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    lr_min_ratio: float = 0.1
    max_steps: int = 100000
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 100
    grad_accumulation_steps: int = 1
    checkpoint_dir: str = "checkpoints"

@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_data_path: str = "train.bin"
    val_data_path: str = "val.bin"

class MemoryMappedDataset:
    """Memory-efficient dataset using numpy memmap."""
    def __init__(self, data_path: str, context_length: int, device: str, single_minibatch: bool = False):
        self.context_length = context_length
        self.device = device
        self.single_minibatch = single_minibatch
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.data_size = len(self.data)
        if self.data_size < context_length + 1:
            raise ValueError(f"Dataset too small: {self.data_size} < {context_length + 1}")
        print(f"Loaded dataset from {data_path}: {self.data_size:,} tokens")
    
    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return data_loading(self.data, batch_size, self.context_length, self.single_minibatch, device=self.device)

class TrainingLogger:
    """Handles logging to console and optionally to an active Weights & Biases run."""
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('training.log'), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: dict[str, float], step: int):
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
        if wandb.run:
            wandb.log(metrics, step=step)
    
    def log_info(self, message: str):
        self.logger.info(message)

def get_lr(step: int, config: TrainingConfig) -> float:
    if config.lr_schedule == "constant":
        return config.learning_rate
    elif config.lr_schedule == "cosine":
        return learning_rate_schedule(
            step, config.learning_rate, config.learning_rate * config.lr_min_ratio,
            config.lr_warmup_steps, config.lr_decay_steps
        )
    elif config.lr_schedule == "linear":
        if step < config.lr_warmup_steps:
            return config.learning_rate * (step / config.lr_warmup_steps)
        else:
            progress = (step - config.lr_warmup_steps) / (config.lr_decay_steps - config.lr_warmup_steps)
            progress = min(1.0, progress)
            return config.learning_rate * (1 - progress) * (1 - config.lr_min_ratio) + config.learning_rate * config.lr_min_ratio
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

def create_model(config: ModelConfig) -> transformer_lm:
    dtype = getattr(torch, config.dtype)
    model = transformer_lm(
        d_model=config.d_model, num_heads=config.num_heads, d_ff=config.d_ff,
        vocab_size=config.vocab_size, num_layers=config.num_layers,
        context_length=config.context_length, theta=config.theta,
        device=config.device, dtype=dtype
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with {param_count:,} parameters")
    return model

def create_optimizer(model: torch.nn.Module, config: TrainingConfig):
    if config.optimizer.lower() == "adamw":
        return AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2), eps=config.eps
        )
    elif config.optimizer.lower() == "sgd":
        return SGD(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

def evaluate_model(model: transformer_lm, dataset: MemoryMappedDataset, config: TrainingConfig, num_batches: int = 10) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = dataset.get_batch(config.batch_size)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item() * inputs.numel()
            total_tokens += inputs.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {"val_loss": avg_loss, "val_perplexity": perplexity}

def train_step(model: transformer_lm, optimizer, dataset: MemoryMappedDataset, config: TrainingConfig, step: int) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    for _ in range(config.grad_accumulation_steps):
        inputs, targets = dataset.get_batch(config.batch_size)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss = loss / config.grad_accumulation_steps
        loss.backward()
        total_loss += loss.item()
    current_lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    gradient_clipping(model.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return {"train_loss": total_loss, "learning_rate": current_lr, "step": step}

def train():
    """Main training function, compatible with wandb sweeps."""
    
    with wandb.init(project="CS336_assignment1") as run:
        config = run.config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        device = config.device
        if device == "auto":
            if torch.cuda.is_available(): device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        
        model_config = ModelConfig(
            d_model=config.d_model, num_heads=config.num_heads, d_ff=config.d_ff,
            vocab_size=config.vocab_size, num_layers=config.num_layers,
            context_length=config.context_length, theta=config.theta, device=device, dtype=config.dtype
        )
        
        max_steps = config.max_tokens // config.batch_size // config.context_length
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * max_steps)
        lr_decay_steps = int(config.lr_decay_steps_ratio * max_steps)
        log_interval = max_steps // config.log_num
        eval_interval = max_steps // config.eval_num
        save_interval = max_steps // config.save_num
        print(f"Using device: {device}, max_steps: {max_steps}")
        
        training_config = TrainingConfig(
            batch_size=config.batch_size, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            optimizer=config.optimizer, beta1=config.beta1, beta2=config.beta2, eps=config.eps,
            lr_schedule=config.lr_schedule, lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps, max_steps=max_steps,
            eval_interval=eval_interval, save_interval=save_interval,
            log_interval=log_interval, grad_accumulation_steps=config.grad_accumulation_steps,
            checkpoint_dir=config.checkpoint_dir
        )
        
        data_config = DataConfig(train_data_path=config.train_data_path, val_data_path=config.val_data_path)
        
        # Create a unique checkpoint directory for each sweep run
        if config.resume_from:
            checkpoint_dir = config.resume_from
        else:
            checkpoint_dir = Path(training_config.checkpoint_dir) / run.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger = TrainingLogger()
        
        model = create_model(model_config)
        optimizer = create_optimizer(model, training_config)

        if model_config.device == 'cuda':
            model = torch.compile(model)
            torch.set_float32_matmul_precision('high')
        elif model_config.device == 'mps':
            model = torch.compile(model, backend="aot_eager")
        
        train_dataset = MemoryMappedDataset(data_config.train_data_path, model_config.context_length, model_config.device, config.single_minibatch)
        val_dataset = None
        if data_config.val_data_path and os.path.exists(data_config.val_data_path):
            val_dataset = MemoryMappedDataset(data_config.val_data_path, model_config.context_length, model_config.device)
        
        start_step = 0
        if config.resume_from:
            start_step = load_checkpoint(config.resume_from, model, optimizer)
            logger.log_info(f"Resumed training from step {start_step}")
        
        logger.log_info("Starting training...")
        start_time = time.time()
        
        for step in range(start_step, training_config.max_steps):
            train_metrics = train_step(model, optimizer, train_dataset, training_config, step)
            
            if step % training_config.log_interval == 0:
                elapsed = time.time() - start_time
                train_metrics.update({
                    "elapsed_time": elapsed,
                    "tokens_per_second": (step * training_config.batch_size * model_config.context_length) / elapsed if elapsed > 0 else 0
                })
                logger.log_metrics(train_metrics, step)
            
            if val_dataset and step > 0 and step % training_config.eval_interval == 0:
                val_metrics = evaluate_model(model, val_dataset, training_config)
                logger.log_metrics(val_metrics, step)
            
            if step > 0 and step % training_config.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step}.pt"
                save_checkpoint(model, optimizer, step, checkpoint_path)
                logger.log_info(f"Saved checkpoint: {checkpoint_path}")
        
        elapsed = time.time() - start_time
        train_metrics.update({
            "elapsed_time": elapsed,
            "tokens_per_second": (step * training_config.batch_size * model_config.context_length) / elapsed if elapsed > 0 else 0
        })
        logger.log_metrics(train_metrics, step)
        
        if val_dataset:
            val_metrics = evaluate_model(model, val_dataset, training_config, num_batches=50)
            logger.log_metrics(val_metrics, training_config.max_steps)

        final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
        save_checkpoint(model, optimizer, training_config.max_steps, final_checkpoint_path)
        logger.log_info(f"Training completed. Final checkpoint saved: {final_checkpoint_path}")
        
        if config.single_minibatch and os.path.exists("data/single_minibatch.npy"):
            os.remove("data/single_minibatch.npy")

def main():

    parser = argparse.ArgumentParser(description="Train transformer language model", fromfile_prefix_chars='@')
    # ModelConfig
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32")
    # TrainingConfig
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps_ratio", type=float, default=0.1)
    parser.add_argument("--lr_decay_steps_ratio", type=float, default=1)
    parser.add_argument("--lr_min_ratio", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=327680000)
    parser.add_argument("--log_num", type=int, default=2000)
    parser.add_argument("--eval_num", type=int, default=2000)
    parser.add_argument("--save_num", type=int, default=5)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # DataConfig
    parser.add_argument("--train_data_path", type=str, default="train.bin")
    parser.add_argument("--val_data_path", type=str, default="val.bin")
    # General arguments
    parser.add_argument("--wandb_project", type=str, default="CS336_assignment1_hyperparameter_tuning", help="Wandb project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single_minibatch", action="store_true", help="Overfit to a single minibatch")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")

    args = parser.parse_args()

    def args_to_sweep_parameters(args, sweep_overrides):
        sweep_parameters = {k: {"value": v} for k, v in vars(args).items()}
        sweep_parameters.update(sweep_overrides)
        return sweep_parameters

    sweep_config = {
        'method': "grid",
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': args_to_sweep_parameters(args, {"learning_rate": {"values": [1e-3, 3e-3, 5e-3, 1e-2]}})
    }

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=train, count=10)

if __name__ == "__main__":
    main()