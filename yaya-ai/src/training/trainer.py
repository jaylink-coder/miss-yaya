"""Main Trainer — orchestrates the full training loop.

Handles:
- Model initialization and distributed wrapping
- Training loop with gradient accumulation
- Mixed precision (BF16)
- Gradient clipping and norm logging
- Periodic evaluation and checkpointing
- Resume from checkpoint
"""

import os
import signal
import time
import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import ModelConfig, TrainingConfig
from src.model.yaya_model import YayaForCausalLM
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.ema import EMA
from src.training.ewc import EWC
from src.training.checkpointing import CheckpointManager
from src.training.logging_utils import TrainingLogger
from src.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    is_main_process,
    barrier,
    all_reduce_mean,
)


class Trainer:
    """Main training orchestrator for Yaya model.

    Usage:
        config = load_model_config("configs/model/yaya_1_5b.yaml")
        train_config = load_training_config("configs/training/train_1_5b.yaml")
        model = YayaForCausalLM(config)
        trainer = Trainer(model, train_config, train_dataloader, eval_dataloader)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """Initialize trainer.

        Args:
            model: The model to train.
            config: Training configuration.
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
        """
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup distributed training
        dist_info = setup_distributed()
        self.rank = dist_info["rank"]
        self.local_rank = dist_info["local_rank"]
        self.world_size = dist_info["world_size"]
        self.device = dist_info["device"]

        # Move model to device
        self.model = model.to(self.device)

        # Enable gradient checkpointing if configured
        if config.gradient_checkpointing and hasattr(self.model, "model"):
            self.model.model.enable_gradient_checkpointing()

        # Wrap model for distributed training
        if self.world_size > 1:
            self.model = wrap_model_ddp(self.model, self.local_rank)

        # Get unwrapped model reference (for saving)
        self.unwrapped_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Create optimizer
        self.optimizer = create_optimizer(
            self.unwrapped_model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            epsilon=config.adam_epsilon,
            layer_lr_decay=getattr(config, "layer_lr_decay", 1.0),
        )

        # Create LR scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
            schedule_type=config.lr_scheduler,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.save_dir,
            keep_last_n=config.keep_last_n,
        )

        # Logger
        self.logger = TrainingLogger(
            project=config.wandb_project,
            run_name=config.wandb_run_name,
            use_wandb=is_main_process(),
            log_steps=config.log_steps,
            rank=self.rank,
        )

        # Mixed precision
        self.use_amp = config.dtype in ("bfloat16", "float16")
        self.amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        # GradScaler is only needed for float16 — bfloat16 has enough range and doesn't need it
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and self.amp_dtype == torch.float16)
        )

        # EMA (Exponential Moving Average) weights — optional, improves eval generalization
        self.ema: Optional[EMA] = None
        if getattr(config, "ema_decay", 0.0) > 0:
            self.ema = EMA(self.unwrapped_model, decay=config.ema_decay)
            print(f"EMA enabled (decay={config.ema_decay})")

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self.last_train_loss = 0.0
        self._interrupted = False

    def train(self, resume_from: Optional[str] = None):
        """Run the full training loop.

        Args:
            resume_from: Path to checkpoint to resume from.
                         If None, tries to auto-resume from latest.
        """
        # Save checkpoint on SIGINT/SIGTERM so training can resume after interruption
        def _handle_signal(signum, frame):
            if is_main_process():
                print(f"\nSignal {signum} received — saving emergency checkpoint at step {self.global_step}...", flush=True)
            self._interrupted = True

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Try to resume from checkpoint
        # Determine which checkpoint will be loaded (for EMA path resolution)
        _ckpt_path = resume_from or self.checkpoint_manager.get_latest_checkpoint()
        metadata = self.checkpoint_manager.load(
            self.unwrapped_model,
            self.optimizer,
            self.scheduler,
            checkpoint_path=resume_from,
        )
        self.global_step = metadata.get("step", 0)
        self.epoch = metadata.get("epoch", 0)

        # Restore EMA shadow weights if a checkpoint exists
        if self.ema is not None and _ckpt_path is not None:
            ema_path = os.path.join(_ckpt_path, "ema.pt")
            if os.path.exists(ema_path):
                self.ema.load(ema_path)

        # Log model info
        if is_main_process():
            summary = self.unwrapped_model.generate_summary()
            self.logger.log_model_info(summary)
            print(f"\nStarting training from step {self.global_step}")
            print(f"  Max steps: {self.config.max_steps}")
            print(f"  Batch size per device: {self.config.per_device_batch_size}")
            print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
            print(f"  World size: {self.world_size}")
            effective_batch = (
                self.config.per_device_batch_size
                * self.config.gradient_accumulation_steps
                * self.world_size
            )
            print(f"  Effective batch size: {effective_batch}")
            print(f"  Sequence length: {self.config.max_seq_length}")
            tokens_per_step = effective_batch * self.config.max_seq_length
            print(f"  Tokens per step: {tokens_per_step:,}")
            print(f"  Device: {self.device}")
            print(f"  Precision: {self.config.dtype}")
            print()

        # Training loop
        self.model.train()

        while self.global_step < self.config.max_steps:
            self.epoch += 1
            self._train_epoch()

            if self._interrupted or self.global_step >= self.config.max_steps:
                break

        # Final save (also covers emergency save on interrupt)
        if is_main_process():
            final_ckpt = self.checkpoint_manager.save(
                self.unwrapped_model,
                self.optimizer,
                self.scheduler,
                step=self.global_step,
                epoch=self.epoch,
                loss=self.last_train_loss,
            )
            if self.ema is not None:
                self.ema.save(os.path.join(final_ckpt, "ema.pt"))
            print(f"\nTraining complete at step {self.global_step}")

        self.logger.finish()
        cleanup_distributed()

    def _train_epoch(self):
        """Run a single training epoch."""
        self.model.train()
        accumulation_loss = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            if self.global_step >= self.config.max_steps or self._interrupted:
                break

            # Heartbeat: confirm data loading and GPU transfer are working
            if batch_idx == 0 and self.epoch == 1 and is_main_process():
                print(f"[Epoch 1, batch 0] Data loaded OK. Moving to {self.device}...",
                      flush=True)

            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if batch_idx == 0 and self.epoch == 1 and is_main_process():
                print(f"[Epoch 1, batch 0] On device. Running forward pass...",
                      flush=True)

            # Forward pass with mixed precision
            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass — scaler is a no-op when disabled (bfloat16 or fp32)
            self.scaler.scale(loss).backward()
            accumulation_loss += loss.item()

            # Optimizer step (after accumulation)
            is_accumulation_step = (batch_idx + 1) % self.config.gradient_accumulation_steps == 0

            if is_accumulation_step:
                # Gradient clipping — must unscale first so clip sees true gradient norms
                grad_norm = None
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)

                    # Optional gradient noise injection (Neelakantan et al. 2015)
                    # Adds Gaussian noise scaled by η/(1+t)^0.55, helping escape
                    # sharp local minima. Only active when grad_noise_eta > 0.
                    if getattr(self.config, "grad_noise_eta", 0.0) > 0:
                        eta = self.config.grad_noise_eta
                        std = (eta / (1 + self.global_step) ** 0.55) ** 0.5
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.add_(torch.randn_like(p.grad) * std)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    ).item()

                # Optimizer step + scaler update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Update EMA shadow weights
                if self.ema is not None:
                    self.ema.update()

                self.global_step += 1

                # Compute tokens processed
                tokens_per_step = (
                    batch["input_ids"].numel()
                    * self.config.gradient_accumulation_steps
                    * self.world_size
                )

                # Log metrics
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.log_step(
                    step=self.global_step,
                    loss=accumulation_loss,
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    tokens_per_step=tokens_per_step,
                )

                self.last_train_loss = accumulation_loss
                accumulation_loss = 0.0

                # Evaluate
                if (
                    self.eval_dataloader is not None
                    and self.config.eval_steps > 0
                    and self.global_step % self.config.eval_steps == 0
                ):
                    self._evaluate()

                # Save checkpoint
                if (
                    self.config.save_steps > 0
                    and self.global_step % self.config.save_steps == 0
                    and is_main_process()
                ):
                    ckpt_dir = self.checkpoint_manager.save(
                        self.unwrapped_model,
                        self.optimizer,
                        self.scheduler,
                        step=self.global_step,
                        epoch=self.epoch,
                        loss=self.last_train_loss,
                    )
                    if self.ema is not None:
                        self.ema.save(os.path.join(ckpt_dir, "ema.pt"))
                    barrier()

    @torch.no_grad()
    def _evaluate(self):
        """Run evaluation on the eval dataset.
        Uses EMA weights if available — EMA model typically has lower eval loss.
        """
        if self.ema is not None:
            self.ema.apply_shadow()
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            if num_batches >= self.config.eval_samples:
                break

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch.get("attention_mask"),
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # All-reduce loss across processes
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        loss_tensor = all_reduce_mean(loss_tensor)
        avg_loss = loss_tensor.item()

        # Compute perplexity
        perplexity = math.exp(min(avg_loss, 20))  # Clamp to avoid overflow

        metrics = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

        self.logger.log_eval(self.global_step, metrics)

        # Track best model
        if avg_loss < self.best_eval_loss and is_main_process():
            self.best_eval_loss = avg_loss
            best_ckpt = self.checkpoint_manager.save(
                self.unwrapped_model,
                self.optimizer,
                self.scheduler,
                step=self.global_step,
                epoch=self.epoch,
                loss=avg_loss,
                extra_state={"best_eval_loss": avg_loss},
            )
            if self.ema is not None:
                self.ema.save(os.path.join(best_ckpt, "ema.pt"))

        # Restore original weights after EMA eval
        if self.ema is not None:
            self.ema.restore()

        self.model.train()
