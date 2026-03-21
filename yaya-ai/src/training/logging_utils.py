"""Logging utilities for training monitoring.

Supports console logging and Weights & Biases integration
for tracking loss, learning rate, throughput, and other metrics.
"""

import os
import time
import sys
from typing import Optional, Dict, Any
from collections import defaultdict


class TrainingLogger:
    """Unified logger for training metrics.

    Logs to console and optionally to Weights & Biases.
    Tracks running averages and computes throughput metrics.
    """

    def __init__(
        self,
        project: str = "yaya-ai",
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        log_steps: int = 10,
        rank: int = 0,
    ):
        """Initialize logger.

        Args:
            project: W&B project name.
            run_name: W&B run name.
            use_wandb: Whether to use W&B logging.
            log_steps: Log every N steps.
            rank: Process rank (only rank 0 logs).
        """
        self.log_steps = log_steps
        self.rank = rank
        self.is_main = rank == 0

        # Running metrics
        self._metrics: Dict[str, list] = defaultdict(list)
        self._step_start_time = time.time()
        self._total_tokens = 0

        # Initialize W&B on main process only
        self.wandb_run = None
        wandb_disabled = (
            not use_wandb
            or not project
            or os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1", "yes")
            or os.environ.get("WANDB_MODE", "").lower() == "disabled"
        )
        if not wandb_disabled and self.is_main:
            try:
                import wandb
                os.environ.setdefault("WANDB_SILENT", "true")
                self.wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    resume="allow",
                )
                print(f"W&B initialized: {project}/{run_name}")
            except BaseException as e:
                print(f"W&B initialization failed: {e}. Continuing without W&B.")

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        tokens_per_step: int = 0,
        extra_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log metrics for a single training step.

        Args:
            step: Current global step.
            loss: Training loss.
            learning_rate: Current learning rate.
            grad_norm: Gradient norm (optional).
            tokens_per_step: Tokens processed this step.
            extra_metrics: Additional metrics to log.
        """
        self._metrics["loss"].append(loss)
        self._metrics["lr"].append(learning_rate)
        if grad_norm is not None:
            self._metrics["grad_norm"].append(grad_norm)
        self._total_tokens += tokens_per_step

        # Log at specified intervals
        if step % self.log_steps == 0 and self.is_main:
            elapsed = time.time() - self._step_start_time
            n_steps = len(self._metrics["loss"])
            tokens_per_sec = tokens_per_step * n_steps / max(elapsed, 1e-6)

            # Compute averages
            avg_loss = sum(self._metrics["loss"]) / len(self._metrics["loss"])

            # Console output
            msg = (
                f"Step {step:>8d} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {learning_rate:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f}"
            )
            if grad_norm is not None:
                avg_gn = sum(self._metrics["grad_norm"]) / len(self._metrics["grad_norm"])
                msg += f" | GradNorm: {avg_gn:.3f}"
            msg += f" | Total: {self._total_tokens / 1e9:.3f}B tokens"

            print(msg, flush=True)

            # W&B logging
            if self.wandb_run is not None:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/learning_rate": learning_rate,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/total_tokens": self._total_tokens,
                }
                if grad_norm is not None:
                    avg_gn = sum(self._metrics["grad_norm"]) / len(self._metrics["grad_norm"])
                    log_dict["train/grad_norm"] = avg_gn
                if extra_metrics:
                    for k, v in extra_metrics.items():
                        log_dict[f"train/{k}"] = v

                import wandb
                wandb.log(log_dict, step=step)

            # Reset running metrics
            self._metrics.clear()
            self._step_start_time = time.time()

    def log_eval(self, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics.

        Args:
            step: Current global step.
            metrics: Dict of metric name to value.
        """
        if not self.is_main:
            return

        # Console output
        parts = [f"EVAL Step {step:>8d}"]
        for k, v in metrics.items():
            parts.append(f"{k}: {v:.4f}")
        print(" | ".join(parts), flush=True)

        # W&B logging
        if self.wandb_run is not None:
            import wandb
            log_dict = {f"eval/{k}": v for k, v in metrics.items()}
            wandb.log(log_dict, step=step)

    def log_model_info(self, info: str):
        """Log model architecture info."""
        if self.is_main:
            print(info, flush=True)

    def finish(self):
        """Finalize logging (close W&B run)."""
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
