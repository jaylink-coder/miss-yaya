"""Elastic Weight Consolidation (EWC) — prevents catastrophic forgetting.

After training on a reference task, compute the diagonal Fisher Information
matrix for each parameter.  During subsequent fine-tuning, a quadratic
penalty anchors parameters to their reference values, weighted by Fisher:

    L_total = L_task + (lambda / 2) * Σ_i  F_i * (θ_i - θ*_i)^2

The more important a parameter was for the previous task (high Fisher), the
more it is penalised for moving.

Reference: Kirkpatrick et al. 2017 — https://arxiv.org/abs/1612.00796
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EWC:
    """Elastic Weight Consolidation.

    Usage pattern (mirrors EMA in ema.py):
        # After initial SFT training is complete:
        ewc = EWC(model, lambda_ewc=5000.0)
        ewc.compute_fisher(reference_dataloader, num_samples=200, device=device)

        # During subsequent fine-tuning, in the loss computation:
        loss = cross_entropy_loss + ewc.penalty()

        # Save alongside checkpoint:
        ewc.save(os.path.join(ckpt_dir, "ewc.pt"))

        # Resume:
        ewc.load(os.path.join(ckpt_dir, "ewc.pt"))
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 5000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Diagonal Fisher estimate per parameter
        self.fisher: dict[str, torch.Tensor] = {}
        # Reference weights θ* (snapshot after reference-task training)
        self.optimal_params: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = 200,
        device: Optional[torch.device] = None,
    ) -> None:
        """Estimate diagonal Fisher by accumulating squared gradients.

        Run num_samples batches from the reference dataloader, performing
        forward + backward each time.  The squared gradient of each parameter
        is an unbiased estimate of the Fisher diagonal under the empirical
        distribution of the data.

        Args:
            dataloader: Reference-task data (same format as training data).
            num_samples: Number of batches to use.  More = better estimate.
            device: Device to run on.  Defaults to first model parameter device.
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()

        # Initialise Fisher accumulators at zero
        fisher_accum: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_accum[name] = torch.zeros_like(param.data, dtype=torch.float32)

        batches_seen = 0
        for batch in dataloader:
            if batches_seen >= num_samples:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            self.model.zero_grad()
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch.get("attention_mask"),
            )
            loss = outputs["loss"]
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal estimate)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accum[name] += param.grad.data.float().pow(2)

            batches_seen += 1

        if batches_seen == 0:
            print("WARNING: EWC Fisher computation saw 0 batches — penalty will be zero.")
            return

        # Normalise by number of batches
        for name in fisher_accum:
            fisher_accum[name] /= batches_seen

        self.fisher = {name: f.to(device) for name, f in fisher_accum.items()}

        # Snapshot current weights as the reference θ*
        self.optimal_params = {
            name: param.data.clone().to(device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        trainable = len(self.fisher)
        print(
            f"EWC Fisher computed over {batches_seen} batches, "
            f"{trainable} parameter tensors stored."
        )
        self.model.train()

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def penalty(self) -> torch.Tensor:
        """Compute the EWC quadratic regularisation penalty.

        Returns a scalar tensor.  Returns 0.0 if Fisher has not been computed yet.
        Fisher and reference weights are moved to each parameter's device on the fly,
        so this works correctly with DDP and mixed-device setups.
        """
        if not self.fisher:
            # Fisher not yet computed — no penalty
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name not in self.fisher:
                continue
            fisher = self.fisher[name].to(param.device)
            ref = self.optimal_params[name].to(param.device)
            loss = loss + (fisher * (param - ref).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss

    # ------------------------------------------------------------------
    # State dict / save / load  (mirrors EMA interface)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "fisher": self.fisher,
            "optimal_params": self.optimal_params,
            "lambda_ewc": self.lambda_ewc,
        }

    def load_state_dict(self, state: dict) -> None:
        self.fisher = state["fisher"]
        self.optimal_params = state["optimal_params"]
        self.lambda_ewc = state.get("lambda_ewc", self.lambda_ewc)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"EWC state saved to {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)
        print(f"EWC state loaded from {path} (lambda={self.lambda_ewc})")
