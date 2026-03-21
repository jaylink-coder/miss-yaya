"""Learning Rate Range Test (LR Finder).

Implements Leslie Smith's LR range test (2015):
  - Ramp LR exponentially from min_lr to max_lr over N steps
  - Record loss at each step
  - Best LR is just before the loss stops falling (steepest slope)

Usage:
    finder = LRFinder(model, optimizer, loss_fn, device)
    best_lr = finder.find(train_dataloader)
    finder.plot()  # if matplotlib available
"""

import math
import copy
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LRFinder:
    """Find the optimal learning rate for a model.

    Algorithm:
        1. Save model and optimizer state
        2. Ramp LR exponentially from min_lr → max_lr
        3. Record smoothed loss at each step
        4. Return LR at steepest negative slope (best descent)
        5. Restore original model and optimizer state
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        smoothing: float = 0.05,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.smoothing = smoothing

        self.history_lr: List[float] = []
        self.history_loss: List[float] = []
        self._best_lr: Optional[float] = None

    def find(
        self,
        dataloader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
        diverge_threshold: float = 4.0,
    ) -> float:
        """Run the LR range test.

        Args:
            dataloader:         Training dataloader (will be iterated cyclically)
            min_lr:             Starting LR
            max_lr:             Ending LR (test stops if loss diverges before)
            num_steps:          Number of steps to ramp
            diverge_threshold:  Stop if loss exceeds best_loss * threshold

        Returns:
            Suggested learning rate (steepest descent point / 10)
        """
        # Save state
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())

        # Set initial LR
        for g in self.optimizer.param_groups:
            g["lr"] = min_lr

        # Exponential LR multiplier per step
        lr_mult = (max_lr / min_lr) ** (1.0 / num_steps)

        self.history_lr = []
        self.history_loss = []

        best_loss = float("inf")
        smoothed_loss = 0.0
        data_iter = iter(dataloader)

        self.model.train()
        print(f"LR Finder: scanning {min_lr:.2e} → {max_lr:.2e} over {num_steps} steps")

        for step in range(num_steps):
            # Get next batch (cycling)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Forward
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda" if "cuda" in str(self.device) else "cpu",
                dtype=torch.float16,
                enabled=torch.cuda.is_available(),
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"]

            loss.backward()
            self.optimizer.step()

            # Smooth loss (exponential moving average)
            loss_val = loss.item()
            smoothed_loss = (
                loss_val if step == 0
                else self.smoothing * loss_val + (1 - self.smoothing) * smoothed_loss
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history_lr.append(current_lr)
            self.history_loss.append(smoothed_loss)

            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Divergence check
            if step > 5 and smoothed_loss > diverge_threshold * best_loss:
                print(f"  Loss diverged at step {step}, lr={current_lr:.2e}. Stopping.")
                break

            if step % 10 == 0:
                print(f"  step {step:3d} | lr={current_lr:.2e} | loss={smoothed_loss:.4f}")

            # Update LR for next step
            for g in self.optimizer.param_groups:
                g["lr"] *= lr_mult

        # Restore state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)

        # Find best LR: point of steepest negative slope
        self._best_lr = self._steepest_slope_lr()
        print(f"\nSuggested LR: {self._best_lr:.2e}  (use this as peak LR)")
        return self._best_lr

    def _steepest_slope_lr(self) -> float:
        """Find LR at the point of maximum loss decrease."""
        if len(self.history_loss) < 3:
            return self.history_lr[len(self.history_lr) // 2]

        losses = self.history_loss
        lrs = self.history_lr

        # Compute gradient of loss w.r.t. log(lr)
        min_grad_idx = 0
        min_grad = 0.0
        for i in range(1, len(losses) - 1):
            grad = losses[i + 1] - losses[i - 1]
            if grad < min_grad:
                min_grad = grad
                min_grad_idx = i

        # Return LR at steepest descent (divide by 10 for safety margin)
        best = lrs[min_grad_idx]
        return best / 10.0

    @property
    def best_lr(self) -> Optional[float]:
        return self._best_lr

    def plot(self, skip_start: int = 5, skip_end: int = 5):
        """Plot loss vs LR curve (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — pip install matplotlib")
            return

        lrs   = self.history_lr[skip_start:-skip_end or None]
        losses = self.history_loss[skip_start:-skip_end or None]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Smoothed Loss")
        ax.set_title("LR Finder — Loss vs Learning Rate")
        if self._best_lr:
            ax.axvline(x=self._best_lr, color="red", linestyle="--",
                       label=f"Suggested LR: {self._best_lr:.2e}")
            ax.legend()
        plt.tight_layout()
        plt.savefig("lr_finder.png", dpi=150)
        print("Saved lr_finder.png")
        plt.show()
