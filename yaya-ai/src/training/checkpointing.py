"""Checkpoint saving and loading for training resilience.

Handles saving/loading of model weights, optimizer state, scheduler state,
and training metadata. Supports sharded saving for large models.
"""

import os
import json
import glob
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    """Manages model checkpoints during training.

    Features:
    - Save model, optimizer, scheduler, and training state
    - Keep only the last N checkpoints
    - Atomic saves (write to temp, then rename)
    - Resume from latest checkpoint
    """

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int = 5,
        save_optimizer: bool = True,
    ):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            model: The model to save.
            optimizer: Optimizer state (optional).
            scheduler: LR scheduler state (optional).
            step: Current training step.
            epoch: Current epoch.
            loss: Current loss value.
            extra_state: Any additional state to save.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_name = f"checkpoint-{step:08d}"
        ckpt_dir = os.path.join(self.save_dir, ckpt_name)
        temp_dir = ckpt_dir + "_temp"

        os.makedirs(temp_dir, exist_ok=True)

        # Save model weights
        model_state = model.state_dict()
        torch.save(model_state, os.path.join(temp_dir, "model.pt"))

        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(temp_dir, "optimizer.pt"))

        # Save scheduler state
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(temp_dir, "scheduler.pt"))

        # Save training metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
        }
        if extra_state:
            metadata.update(extra_state)

        with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Atomic rename
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        os.rename(temp_dir, ckpt_dir)

        # Update "latest" symlink/pointer
        latest_path = os.path.join(self.save_dir, "latest")
        with open(latest_path, "w") as f:
            f.write(ckpt_name)

        print(f"Checkpoint saved: {ckpt_dir} (step={step}, loss={loss:.4f})")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return ckpt_dir

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to load state into (optional).
            scheduler: Scheduler to load state into (optional).
            checkpoint_path: Specific checkpoint to load. If None, loads latest.
            map_location: Device to map tensors to.

        Returns:
            Training metadata dict.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            print("No checkpoint found. Starting from scratch.")
            return {"step": 0, "epoch": 0, "loss": float("inf")}

        # Verify checkpoint directory actually exists (catches unmounted Drive, bad paths, etc.)
        if not os.path.isdir(checkpoint_path):
            print(f"WARNING: Checkpoint path does not exist: {checkpoint_path}")
            print("Starting from scratch.")
            return {"step": 0, "epoch": 0, "loss": float("inf")}

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load model weights
        model_path = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)
            print("  Model weights loaded OK.")
        else:
            print(f"  WARNING: model.pt not found at {model_path} — model stays at current init!")

        # Load optimizer state
        opt_path = os.path.join(checkpoint_path, "optimizer.pt")
        if optimizer is not None and os.path.exists(opt_path):
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=map_location, weights_only=True)
            )
            print("  Optimizer state loaded OK.")
        elif optimizer is not None:
            print("  WARNING: optimizer.pt not found — optimizer state reset.")

        # Load scheduler state
        sched_path = os.path.join(checkpoint_path, "scheduler.pt")
        if scheduler is not None and os.path.exists(sched_path):
            scheduler.load_state_dict(
                torch.load(sched_path, map_location=map_location, weights_only=True)
            )
            print("  Scheduler state loaded OK.")
        elif scheduler is not None:
            print("  WARNING: scheduler.pt not found — scheduler state reset.")

        # Load metadata
        meta_path = os.path.join(checkpoint_path, "metadata.json")
        metadata = {"step": 0, "epoch": 0, "loss": float("inf")}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            print(f"  Metadata loaded: step={metadata.get('step', 0)}, "
                  f"epoch={metadata.get('epoch', 0)}, loss={metadata.get('loss', 'n/a')}")
        else:
            print(f"  WARNING: metadata.json not found at {meta_path}")

        # Fallback: infer step from checkpoint directory name (e.g. checkpoint-00015008 → 15008)
        if metadata.get("step", 0) == 0:
            ckpt_name = os.path.basename(checkpoint_path.rstrip("/\\"))
            if ckpt_name.startswith("checkpoint-"):
                try:
                    inferred_step = int(ckpt_name.split("-")[-1])
                    if inferred_step > 0:
                        print(f"  Inferred step {inferred_step} from checkpoint directory name.")
                        metadata["step"] = inferred_step
                except ValueError:
                    pass

        print(f"  Resuming from step {metadata['step']}")
        return metadata

    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint directory."""
        latest_path = os.path.join(self.save_dir, "latest")
        if os.path.exists(latest_path):
            with open(latest_path, "r") as f:
                ckpt_name = f.read().strip()
            ckpt_dir = os.path.join(self.save_dir, ckpt_name)
            if os.path.exists(ckpt_dir):
                return ckpt_dir

        # Fallback: find highest-numbered checkpoint
        ckpt_dirs = sorted(glob.glob(os.path.join(self.save_dir, "checkpoint-*")))
        if ckpt_dirs:
            return ckpt_dirs[-1]

        return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return

        ckpt_dirs = sorted(glob.glob(os.path.join(self.save_dir, "checkpoint-*")))
        while len(ckpt_dirs) > self.keep_last_n:
            old_dir = ckpt_dirs.pop(0)
            print(f"  Removing old checkpoint: {old_dir}")
            shutil.rmtree(old_dir)

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return sorted(glob.glob(os.path.join(self.save_dir, "checkpoint-*")))
