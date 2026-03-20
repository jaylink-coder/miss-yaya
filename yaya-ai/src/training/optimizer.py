"""Optimizer and learning rate scheduler for Yaya training.

Implements AdamW with cosine learning rate schedule and warmup,
following modern LLM training best practices.
"""

import math
from typing import Optional, List

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> AdamW:
    """Create AdamW optimizer with weight decay applied selectively.

    Weight decay is NOT applied to:
    - Bias terms
    - LayerNorm / RMSNorm weights
    - Embedding weights

    This follows standard practice from GPT/LLaMA training.

    Args:
        model: The model to optimize.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        epsilon: AdamW epsilon.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases, norms, and embeddings
        if any(nd in name for nd in ["bias", "layernorm", "norm", "embedding"]):
            no_decay_params.append(param)
        elif param.dim() < 2:
            # Scalars and 1D tensors (biases, norm weights)
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Log parameter counts
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer: {num_decay:,} params with decay, {num_no_decay:,} without")

    return AdamW(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 2000,
    max_steps: int = 100000,
    min_lr_ratio: float = 0.1,
    schedule_type: str = "cosine",
) -> LambdaLR:
    """Create learning rate scheduler with warmup.

    Supports:
    - cosine: Linear warmup then cosine decay to min_lr
    - linear: Linear warmup then linear decay to min_lr
    - constant: Linear warmup then constant LR

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Final LR as a fraction of peak LR.
        schedule_type: Type of schedule ('cosine', 'linear', 'constant').

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Post-warmup phase
        if schedule_type == "constant":
            return 1.0

        # Calculate progress after warmup
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        progress = min(1.0, progress)

        if schedule_type == "cosine":
            # Cosine decay from 1.0 to min_lr_ratio
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        elif schedule_type == "linear":
            # Linear decay from 1.0 to min_lr_ratio
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        elif schedule_type == "wsd":
            # Warmup-Stable-Decay: hold peak LR for 80% of post-warmup steps,
            # then cosine decay over last 20%. Much more robust than cosine when
            # token count is limited — avoids locking into noise during decay.
            decay_start = 0.8
            if progress < decay_start:
                return 1.0  # stable phase: full LR
            decay_progress = (progress - decay_start) / (1.0 - decay_start)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * decay_progress)
            )
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)
