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
    use_8bit: bool = True,
    layer_lr_decay: float = 1.0,
) -> AdamW:
    """Create AdamW optimizer with weight decay applied selectively.

    Weight decay is NOT applied to:
    - Bias terms
    - LayerNorm / RMSNorm weights
    - Embedding weights

    Layer-wise LR decay (LLRD):
        When layer_lr_decay < 1.0, deeper layers get a smaller LR.
        Layer i of N total gets LR * decay^(N - i).
        This stabilises early layers (learned general features)
        while letting upper layers adapt faster — crucial for fine-tuning.
        Set layer_lr_decay=1.0 (default) for pretraining (all layers equal).

    Args:
        model: The model to optimize.
        learning_rate: Peak learning rate for the top layer.
        weight_decay: Weight decay coefficient.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        epsilon: AdamW epsilon.
        use_8bit: Use 8-bit Adam if bitsandbytes is available.
        layer_lr_decay: Per-layer LR multiplier (1.0 = disabled, 0.9 = typical).

    Returns:
        Configured AdamW optimizer.
    """
    no_decay_names = {"bias", "layernorm", "norm", "embedding"}

    if layer_lr_decay < 1.0:
        # ── Layer-wise LR decay ──────────────────────────────────────────────
        # Assign each parameter to a depth bucket, then scale its LR.
        param_groups = []
        num_layers = getattr(model, 'config', None)
        num_layers = (
            num_layers.num_hidden_layers if num_layers else
            sum(1 for n, _ in model.named_parameters() if ".layers." in n or ".h." in n)
        )
        if num_layers == 0:
            num_layers = 12  # safe fallback

        depth_cache: dict = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Parse layer index from parameter name
            import re as _re
            m = _re.search(r'\.(?:layers|h|blocks)\.(\d+)\.', name)
            depth = int(m.group(1)) if m else (num_layers if "lm_head" in name else 0)
            depth_cache[name] = depth

        # Build one param group per (depth, no_decay) combination
        groups: dict = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            depth = depth_cache[name]
            is_no_decay = any(nd in name for nd in no_decay_names) or param.dim() < 2
            key = (depth, is_no_decay)
            if key not in groups:
                scale = layer_lr_decay ** (num_layers - depth)
                groups[key] = {
                    "params": [],
                    "lr": learning_rate * scale,
                    "weight_decay": 0.0 if is_no_decay else weight_decay,
                }
            groups[key]["params"].append(param)

        param_groups = list(groups.values())
        num_decay   = sum(p.numel() for g in param_groups for p in g["params"] if g["weight_decay"] > 0)
        num_nodecay = sum(p.numel() for g in param_groups for p in g["params"] if g["weight_decay"] == 0)
        print(f"Optimizer (LLRD decay={layer_lr_decay}): "
              f"{num_decay:,} params with decay, {num_nodecay:,} without, "
              f"{len(param_groups)} LR groups")
    else:
        # ── Standard flat LR ────────────────────────────────────────────────
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay_names) or param.dim() < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in no_decay_params)
        print(f"Optimizer: {num_decay:,} params with decay, {num_nodecay:,} without")

    # Use 8-bit Adam if available — cuts optimizer memory from 8GB to 2GB,
    # essential for 1B+ models on 16GB GPUs like T4
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW (bitsandbytes) — saves ~6GB optimizer memory")
            return bnb.optim.AdamW8bit(
                param_groups,
                lr=learning_rate,
                betas=(beta1, beta2),
                eps=epsilon,
            )
        except ImportError:
            print("bitsandbytes not found — falling back to standard AdamW")

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
