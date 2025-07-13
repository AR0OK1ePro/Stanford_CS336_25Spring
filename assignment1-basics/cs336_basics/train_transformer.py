import torch
import math
import numpy as np
import os
from jaxtyping import Float, Int
from einops import reduce
from collections.abc import Callable, Iterable
from typing import Optional

def cross_entropy(
        logits: Float[torch.Tensor, "batch_size seq_len vocab_size"], 
        targets: Int[torch.Tensor, "batch_size seq_len"]) -> float:

    logits_max = reduce(logits, "... vocab_size -> ...", "max")
    logits_stable = logits - logits_max.unsqueeze(-1)
    logits_exp_sum = reduce(torch.exp(logits_stable), "... vocab_size -> ...", "sum")
    logits_targets = logits_stable.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = (torch.log(logits_exp_sum) - logits_targets).mean()
    return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
            
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas: tuple = (0.9, 0.999), eps=1e-8):
        if lr < 0 or weight_decay < 0 or weight_decay > 1 or betas[0] < 0 or betas[0] > 1 or betas[1] < 0 or betas[1] > 1:
            raise ValueError
        defaults = {"lr": lr, "beta_1": betas[0], "beta_2": betas[1], "weight_decay": weight_decay, "epsilon": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            epsilon = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                lr_t = lr * (math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t))
                p.data -= lr_t * (m / (torch.sqrt(v) + epsilon))
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    if t < 0:
        raise ValueError("Invalid t < 0")
    if t < t_w:
        return float(t / t_w) * lr_max
    elif t <= t_c:
        return lr_min + (1 + math.cos(float(t - t_w) / float(t_c - t_w) * math.pi)) * (lr_max - lr_min) / 2 
    else:
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], l2_norm_max: float, eps=1e-6):
    """
    Clips the gradients of the given parameters so that their L2 norm does not exceed l2_norm_max.

    Args:
        params: Iterable of torch.nn.Parameter.
        l2_norm_max: Maximum allowed L2 norm for gradients.
        eps: Small epsilon for numerical stability.

    Returns:
        The input params, with gradients clipped in-place.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return params
    l2_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
    if l2_norm > l2_norm_max:
        scale = l2_norm_max / (l2_norm + eps)
        for g in grads:
            g.mul_(scale)
    return params

def data_loading(x: np.array, batch_size, context_length, single_minibatch=False, device="mps"):
    if single_minibatch:
        minibatch_path = "data/single_minibatch.npy"
        if os.path.exists(minibatch_path):
            indices = np.load(minibatch_path)
        else:
            indices = np.random.randint(0, len(x) - context_length, size=batch_size)
            np.save(minibatch_path, indices)
    else:
        indices = np.random.randint(0, len(x) - context_length, size=batch_size)

    # Efficiently create batches using broadcasting
    offsets = np.arange(context_length)
    input_indices = indices[:, np.newaxis] + offsets

    input_sequence = x[input_indices]
    output_sequence = x[input_indices + 1]

    # Use torch.from_numpy to avoid a copy
    input_tensor = torch.from_numpy(input_sequence).to(device=device, dtype=torch.long)
    output_tensor = torch.from_numpy(output_sequence).to(device=device, dtype=torch.long)
    return input_tensor, output_tensor

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration, out):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(obj, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
