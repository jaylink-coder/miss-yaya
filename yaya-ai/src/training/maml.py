"""Model-Agnostic Meta-Learning (MAML) for fast task adaptation.

MAML trains a model to have initial weights θ such that a small number of
gradient steps on a new task's support set leads to strong performance on
that task's query set.  This gives the model the ability to adapt to new
domains in 1-5 gradient steps at inference time — equivalent to biological
"learning to learn".

Algorithm (Finn et al. 2017):
    For each meta-batch of tasks:
        For each task τ_i:
            1. Sample support set S_i and query set Q_i
            2. Compute task loss on S_i: L_S(θ)
            3. Inner update: θ'_i = θ − α * ∇_θ L_S(θ)       (inner loop)
            4. Compute query loss with adapted weights: L_Q(θ'_i)
        Meta-update: θ ← θ − β * ∇_θ Σ_i L_Q(θ'_i)           (outer loop)

Yaya integration:
    - Works with LoRA adapters — only adapts lora_A/lora_B in inner loop
      for parameter efficiency (reduces inner-loop compute by ~200x)
    - Compatible with EWC penalty on outer loop to prevent forgetting
      across meta-updates
    - MAML state can be saved/restored alongside regular checkpoints

Usage:
    maml = MAML(model, MAMLConfig(inner_lr=0.01, inner_steps=5))

    # One meta-update step (call inside training loop):
    meta_loss = maml.outer_step(task_batches, optimizer)

    # Fast adaptation at inference time (no meta-update):
    adapted_params = maml.adapt(support_set, steps=5)
    # Use adapted_params with functional forward pass

Reference: Finn, Abbeel, Levine (2017) "Model-Agnostic Meta-Learning for
Fast Adaptation of Deep Networks" — https://arxiv.org/abs/1703.03400
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MAMLConfig:
    # Inner-loop learning rate (task-specific adaptation step size)
    inner_lr: float = 0.01
    # Number of gradient steps in the inner loop per task
    inner_steps: int = 5
    # Number of tasks sampled per meta-update (meta-batch size)
    meta_batch_size: int = 4
    # Only adapt LoRA adapter params in the inner loop (much cheaper)
    lora_only: bool = True
    # Use first-order MAML approximation (no second-order gradients)
    # First-order is ~10x cheaper and nearly as good in practice (Reptile)
    first_order: bool = True


# ---------------------------------------------------------------------------
# MAML
# ---------------------------------------------------------------------------

class MAML:
    """Model-Agnostic Meta-Learning wrapper for YayaForCausalLM.

    Supports:
    - First-order MAML (default, efficient) and second-order MAML
    - LoRA-only inner loop for parameter efficiency
    - Compatible with EWC outer-loop penalty
    """

    def __init__(self, model: nn.Module, config: Optional[MAMLConfig] = None):
        self.model = model
        self.config = config or MAMLConfig()

    # ------------------------------------------------------------------
    # Core MAML operations
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        support_batch: Dict[str, torch.Tensor],
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run inner-loop adaptation on a support set.

        Performs ``inner_steps`` gradient steps using ``inner_lr``.
        Uses functional forward pass so gradients can flow back through
        the adaptation process for the outer loop.

        Args:
            support_batch: dict with 'input_ids' and 'labels'
            params:        starting parameters (default: model's current params)

        Returns:
            adapted_params: dict of parameter tensors after inner-loop steps
        """
        if params is None:
            params = self._get_params()

        # Clone to leaf tensors with grad so autograd.grad can differentiate
        adapted = {k: v.clone().detach().requires_grad_(True) for k, v in params.items()}

        for _ in range(self.config.inner_steps):
            loss = self._functional_loss(support_batch, adapted)
            grads = torch.autograd.grad(
                loss,
                list(adapted.values()),
                create_graph=not self.config.first_order,
                allow_unused=True,
            )
            adapted = {
                name: (p - self.config.inner_lr * (g if g is not None else torch.zeros_like(p)))
                for (name, p), g in zip(adapted.items(), grads)
            }

        return adapted

    def query_loss(
        self,
        query_batch: Dict[str, torch.Tensor],
        adapted_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss on the query set using adapted parameters."""
        return self._functional_loss(query_batch, adapted_params)

    def outer_step(
        self,
        task_batches: List[Tuple[Dict, Dict]],
        meta_optimizer: torch.optim.Optimizer,
        ewc: Optional[object] = None,
    ) -> float:
        """Perform one meta-update over a batch of tasks.

        Args:
            task_batches:   list of (support_batch, query_batch) dicts
            meta_optimizer: outer-loop optimizer (e.g. AdamW on model params)
            ewc:            optional EWC instance for continual learning penalty

        Returns:
            meta_loss (float) — average query loss across tasks
        """
        meta_optimizer.zero_grad(set_to_none=True)
        total_meta_loss = torch.tensor(0.0, device=self._device())

        for support_batch, query_batch in task_batches:
            adapted = self.inner_loop(support_batch)
            q_loss = self.query_loss(query_batch, adapted)
            total_meta_loss = total_meta_loss + q_loss

        meta_loss = total_meta_loss / max(len(task_batches), 1)

        # EWC outer-loop penalty to prevent meta-update from forgetting prior tasks
        if ewc is not None:
            meta_loss = meta_loss + ewc.penalty()

        meta_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        meta_optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_batch: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fast adaptation at inference time (no outer gradient).

        Given a small support set, returns adapted parameters that can be
        used for a few-shot forward pass.

        Args:
            support_batch: support set with 'input_ids' and 'labels'
            steps:         override inner_steps (default: config.inner_steps)

        Returns:
            adapted_params dict — apply with apply_params() before inference
        """
        orig_steps = self.config.inner_steps
        if steps is not None:
            self.config.inner_steps = steps
        try:
            with torch.enable_grad():
                return self.inner_loop(support_batch)
        finally:
            self.config.inner_steps = orig_steps

    def apply_params(self, params: Dict[str, torch.Tensor]) -> None:
        """Temporarily apply adapted parameters to the model in-place."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data.copy_(params[name])

    def restore_params(self, original_params: Dict[str, torch.Tensor]) -> None:
        """Restore original parameters after inference with adapted params."""
        self.apply_params(original_params)

    def snapshot_params(self) -> Dict[str, torch.Tensor]:
        """Return a copy of current model parameters for later restoration."""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_params(self) -> Dict[str, torch.Tensor]:
        """Get trainable parameters (LoRA only if lora_only=True).

        Returns tensors that are leaf variables with requires_grad=True so
        that autograd.grad can differentiate through the inner-loop steps.
        """
        params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.config.lora_only and not ("lora_A" in name or "lora_B" in name):
                continue
            params[name] = param
        return params

    def _functional_loss(
        self,
        batch: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass with provided param tensors via functional_call.

        Uses ``torch.func.functional_call`` to substitute only the keys
        present in ``params`` — the rest of the model uses its live weights.
        This lets gradients flow back through ``params`` for MAML meta-updates.
        """
        from torch.func import functional_call

        outputs = functional_call(
            self.model,
            params,
            args=(),
            kwargs={
                "input_ids": batch["input_ids"],
                "labels": batch["labels"],
                "attention_mask": batch.get("attention_mask"),
            },
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device
