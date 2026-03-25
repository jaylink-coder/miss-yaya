"""Online Learner — learn from inference-time feedback without full retraining.

Collects (prompt, response, score) examples into a rolling buffer and
periodically runs micro-finetune steps directly on the live model.

This is Yaya's equivalent of synaptic plasticity in response to new
experience — continuous, lightweight, targeted weight updates triggered
by feedback at inference time.

Usage:
    learner = OnlineLearner(model, tokenizer, OnlineLearnerConfig(), device)

    # After each inference round, if feedback is available:
    learner.add_example(prompt, response, score=1.0)   # positive
    learner.add_example(prompt, bad_response, score=-1.0)  # negative (stored but not SFT'd)

    # Micro-finetune fires automatically when buffer has enough examples.
    # Can also be triggered manually:
    learner.step()

    # Persist state (e.g. on server shutdown):
    learner.save_state("checkpoints/online_learner.pt")
    learner.load_state("checkpoints/online_learner.pt")
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.training.ewc import EWC
    from src.training.synthetic_replay import SyntheticReplay


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OnlineLearnerConfig:
    buffer_capacity: int = 1000          # max examples kept in rolling buffer
    min_examples_to_finetune: int = 32   # don't micro-finetune until this many collected
    finetune_every_n_examples: int = 50  # trigger a step every N new additions
    micro_finetune_steps: int = 10       # gradient steps per trigger
    micro_lr: float = 5e-5
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    buffer_path: str = "checkpoints/online_buffer.jsonl"


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class OnlineLearner:
    """Collects feedback examples and triggers periodic micro-finetuning.

    Thread-safe: add_example() / step() are protected by a Lock, making
    this safe to call from concurrent HTTP request handlers.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: OnlineLearnerConfig,
        device: torch.device,
        ewc: Optional["EWC"] = None,
    ):
        if tokenizer is None:
            raise ValueError(
                "OnlineLearner requires a tokenizer for encoding prompt/response pairs. "
                "Pass the model's tokenizer when constructing OnlineLearner."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.ewc = ewc  # Optional EWC regulariser — prevents forgetting during micro-finetune

        self.buffer: deque[dict] = deque(maxlen=config.buffer_capacity)
        self.examples_since_last_finetune = 0
        self._lock = threading.Lock()

        # Lightweight optimizer — only trains whatever has requires_grad=True.
        # When LoRA is active, this naturally covers only the adapter params.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable_params, lr=config.micro_lr)

        # Restore persisted buffer from disk if it exists
        self._load_buffer_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_example(self, prompt: str, response: str, score: float = 1.0) -> None:
        """Add a feedback example to the rolling buffer.

        Args:
            prompt:   The input prompt that was given to the model.
            response: The model response (or the desired response).
            score:    Feedback signal.  > 0 = positive (used for SFT loss),
                      <= 0 = negative (stored for future DPO use, not trained on now).
        """
        example = {"prompt": prompt, "response": response, "score": score}

        with self._lock:
            self.buffer.append(example)
            self.examples_since_last_finetune += 1
            self._append_to_disk(example)

            should_step = (
                len(self.buffer) >= self.config.min_examples_to_finetune
                and self.examples_since_last_finetune >= self.config.finetune_every_n_examples
            )

        if should_step:
            self.step()

    def step(self) -> Optional[float]:
        """Run micro_finetune_steps gradient updates on positive examples from buffer.

        Returns the average loss over the micro-steps, or None if there were
        not enough positive examples.
        """
        with self._lock:
            positives = [e for e in self.buffer if e["score"] > 0]
            if len(positives) < max(1, self.config.min_examples_to_finetune // 4):
                return None
            # Sample up to 32 positive examples per step
            import random
            sample = random.sample(positives, min(32, len(positives)))
            self.examples_since_last_finetune = 0

        total_loss = 0.0
        self.model.train()

        for _ in range(self.config.micro_finetune_steps):
            self._optimizer.zero_grad(set_to_none=True)
            batch_loss = self._compute_batch_loss(sample)
            if batch_loss is None:
                continue
            # EWC penalty — prevents online updates from erasing prior knowledge
            if self.ewc is not None:
                batch_loss = batch_loss + self.ewc.penalty()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self._optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / max(self.config.micro_finetune_steps, 1)
        print(
            f"[OnlineLearner] micro-finetune complete — "
            f"steps={self.config.micro_finetune_steps}, avg_loss={avg_loss:.4f}, "
            f"buffer_size={len(self.buffer)}"
        )
        return avg_loss

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save buffer and optimizer state to disk."""
        state = {
            "buffer": list(self.buffer),
            "examples_since_last_finetune": self.examples_since_last_finetune,
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"[OnlineLearner] state saved to {path}")

    def load_state(self, path: str) -> None:
        """Restore buffer and optimizer state from disk."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.buffer = deque(state.get("buffer", []), maxlen=self.config.buffer_capacity)
        self.examples_since_last_finetune = state.get("examples_since_last_finetune", 0)
        self._optimizer.load_state_dict(state["optimizer"])
        print(f"[OnlineLearner] state loaded from {path}, buffer_size={len(self.buffer)}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_batch_loss(self, examples: list[dict]) -> Optional[torch.Tensor]:
        """Tokenise examples and compute cross-entropy loss (prompt masked)."""
        losses = []

        for ex in examples:
            text = ex["prompt"] + ex["response"]
            try:
                ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            except Exception:
                continue

            ids = ids[: self.config.max_seq_length]
            if len(ids) < 2:
                continue

            input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
            labels = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)

            # Mask prompt tokens so we only train on the response
            try:
                prompt_ids = self.tokenizer.encode(ex["prompt"], add_bos=True)
            except Exception:
                prompt_ids = []
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[:, :prompt_len] = -100

            outputs = self.model(input_ids=input_ids, labels=labels)
            if "loss" in outputs and outputs["loss"] is not None:
                losses.append(outputs["loss"])

        if not losses:
            return None
        return torch.stack(losses).mean()

    def _append_to_disk(self, example: dict) -> None:
        """Append one example to the JSONL buffer file."""
        try:
            os.makedirs(os.path.dirname(self.config.buffer_path) or ".", exist_ok=True)
            with open(self.config.buffer_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(example) + "\n")
        except Exception as e:
            print(f"[OnlineLearner] WARNING: could not write to buffer file: {e}")

    def _load_buffer_from_disk(self) -> None:
        """Restore buffer from JSONL file if it exists."""
        if not os.path.exists(self.config.buffer_path):
            return
        try:
            with open(self.config.buffer_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.buffer.append(json.loads(line))
            print(
                f"[OnlineLearner] restored {len(self.buffer)} examples "
                f"from {self.config.buffer_path}"
            )
        except Exception as e:
            print(f"[OnlineLearner] WARNING: could not read buffer file: {e}")
