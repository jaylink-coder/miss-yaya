"""Exponential Moving Average (EMA) of model weights.

EMA maintains a shadow copy of model parameters as a running average:
    ema_param = decay * ema_param + (1 - decay) * model_param

Benefits:
- EMA weights typically generalize better than the last checkpoint
- Acts as implicit ensembling
- Helps smooth out noisy gradient updates
- Standard in: DALL-E, Stable Diffusion, many SOTA LLMs

Usage:
    ema = EMA(model, decay=0.9999)
    # In training loop, after optimizer.step():
    ema.update()
    # For evaluation, swap to EMA weights:
    with ema.average_parameters():
        eval_loss = evaluate(model)
    # Weights restore automatically after the with block
"""

import copy
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Args:
        model:       The model whose parameters to track
        decay:       EMA decay factor (0.9999 typical for LLMs;
                     lower values like 0.999 respond faster to changes)
        update_every: Only update EMA every N optimizer steps
                      (saves compute for large models)
        warmup_steps: During warmup, use smaller effective decay
                      to let EMA catch up to model quickly
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 1,
        warmup_steps: int = 100,
    ):
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Shadow parameters (on same device as model)
        self.shadow: dict = {}
        self.backup: dict = {}
        self._register()

    def _register(self):
        """Initialise shadow weights as copies of current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    def _effective_decay(self) -> float:
        """Ramp up decay during warmup so EMA is not stuck at initial weights."""
        if self.step_count < self.warmup_steps:
            return min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        return self.decay

    @torch.no_grad()
    def update(self):
        """Update shadow weights. Call once per optimizer step."""
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return

        decay = self._effective_decay()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    param.data.float(), alpha=1.0 - decay
                )

    def apply_shadow(self):
        """Replace model weights with EMA weights (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self):
        """Restore original model weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    @contextmanager
    def average_parameters(self):
        """Context manager: temporarily use EMA weights.

        Usage:
            with ema.average_parameters():
                loss = evaluate(model)
            # model weights restored here
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> dict:
        return {
            "shadow":      self.shadow,
            "step_count":  self.step_count,
            "decay":       self.decay,
        }

    def load_state_dict(self, state: dict):
        self.shadow     = state["shadow"]
        self.step_count = state.get("step_count", 0)
        self.decay      = state.get("decay", self.decay)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"EMA saved to {path}")

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)
        print(f"EMA loaded from {path} (step {self.step_count})")
