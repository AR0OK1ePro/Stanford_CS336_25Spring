#!/usr/bin/env python3
"""
Main training script for transformer language model.
Supports configurable hyperparameters, efficient data loading, checkpointing, and logging.
"""

import os
import sys
import argparse
import json
import time
import math
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any
import logging
import contextlib

# Import our custom modules
from cs336_basics.transformer import transformer_lm
from cs336_basics.train_transformer import (
    cross_entropy, SGD, AdamW, learning_rate_schedule, 
    gradient_clipping, save_checkpoint, load_checkpoint, data_loading
)

# Optional: Weights & Biases logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Logging to console only.")

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

@contextlib.contextmanager
def dummy_context_manager():
    """A context manager that does nothing, for when wandb is disabled."""
    yield None

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
    def __init__(self, use_wandb: bool):
        self.use_wandb = use_wandb and HAS_WANDB
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('training.log'), logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: dict[str, float], step: int):
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
        if self.use_wandb and wandb.run:
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

def train(args):
    """Main training function, compatible with wandb sweeps."""
    
    context_manager = dummy_context_manager()
    if HAS_WANDB and args.use_wandb:
        context_manager = wandb.init(project=args.wandb_project, config=args)

    with context_manager as run:
        config = run.config if (HAS_WANDB and args.use_wandb) else args

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        device = config.device
        if device == "auto":
            if torch.cuda.is_available(): device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        print(f"Using device: {device}")
        
        model_config = ModelConfig(
            d_model=config.d_model, num_heads=config.num_heads, d_ff=config.d_ff,
            vocab_size=config.vocab_size, num_layers=config.num_layers,
            context_length=config.context_length, theta=config.theta, device=device, dtype=config.dtype
        )
        
        training_config = TrainingConfig(
            batch_size=config.batch_size, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, max_grad_norm=config.max_grad_norm,
            optimizer=config.optimizer, beta1=config.beta1, beta2=config.beta2, eps=config.eps,
            lr_schedule=config.lr_schedule, lr_warmup_steps=config.lr_warmup_steps,
            lr_decay_steps=config.lr_decay_steps, max_steps=config.max_steps,
            eval_interval=config.eval_interval, save_interval=config.save_interval,
            log_interval=config.log_interval, grad_accumulation_steps=config.grad_accumulation_steps,
            checkpoint_dir=config.checkpoint_dir
        )
        
        data_config = DataConfig(train_data_path=config.train_data, val_data_path=config.val_data)
        
        # Create a unique checkpoint directory for each sweep run
        if config.resume_from:
            checkpoint_dir = config.resume_from
        else:
            checkpoint_dir = Path(training_config.checkpoint_dir) / run.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger = TrainingLogger(use_wandb=config.use_wandb)
        
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
        
        final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
        save_checkpoint(model, optimizer, training_config.max_steps, final_checkpoint_path)
        logger.log_info(f"Training completed. Final checkpoint saved: {final_checkpoint_path}")
        
        if val_dataset:
            val_metrics = evaluate_model(model, val_dataset, training_config, num_batches=50)
            logger.log_metrics(val_metrics, training_config.max_steps)
        
        if config.single_minibatch and os.path.exists("data/single_minibatch.npy"):
            os.remove("data/single_minibatch.npy")

def main():
    parser = argparse.ArgumentParser(description="Train transformer language model", fromfile_prefix_chars='@')
    
    # General arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="CS336_assignment1", help="Wandb project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--single_minibatch", action="store_true", help="Overfit to a single minibatch")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from checkpoint")

    # Add arguments for all configs
    for config_class in [ModelConfig, TrainingConfig, DataConfig]:
        for field_name, field_type in config_class.__annotations__.items():
            default_value = getattr(config_class(), field_name)
            parser.add_argument(f"--{field_name}", type=field_type, default=default_value)

    # Sweep arguments
    parser.add_argument("--sweep", action="store_true", help="Run a wandb sweep")
    parser.add_argument("--sweep_method", type=str, default="random", choices=["random", "grid", "bayes"], help="Wandb sweep method")
    parser.add_argument("--sweep_runs", type=int, default=10, help="Number of wandb sweep runs")
    parser.add_argument("--sweep_param", action="append", help="Define a hyperparameter to sweep over. Format: 'name,type,val1,val2,...' e.g., 'learning_rate,log_uniform_values,1e-5,1e-3' or 'num_layers,values,4,6,8'")

    args = parser.parse_args()

    if args.sweep:
        sweep_parameters = {}
        if args.sweep_param:
            for param_str in args.sweep_param:
                parts = param_str.split(',')
                name = parts[0]
                dist_type = parts[1]
                values = [float(v) if '.' in v or 'e' in v else int(v) for v in parts[2:]]
                
                param_dict = {"distribution": dist_type}
                if dist_type in ["uniform", "log_uniform_values"]:
                    param_dict['min'] = values[0]
                    param_dict['max'] = values[1]
                elif dist_type == "values":
                    param_dict['values'] = values
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")
                sweep_parameters[name] = param_dict

        sweep_config = {
            'method': args.sweep_method,
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': sweep_parameters
        }
        
        if not sweep_parameters:
            print("Error: --sweep was specified but no --sweep_param was provided.")
            sys.exit(1)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, lambda: train(args), count=args.sweep_runs)
    else:
        train(args)

if __name__ == "__main__":
    main()