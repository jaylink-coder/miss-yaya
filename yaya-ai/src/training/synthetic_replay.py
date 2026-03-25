"""Generative synthetic replay for continual learning.

Instead of storing raw user data, the model periodically generates synthetic
completions from stored task anchors (short representative prompts).  These
pseudo-examples are replayed during micro-finetuning to rehearse prior
knowledge — privacy-friendly and storage-efficient.

Key idea (Shin et al. 2017 "Continual Learning with Deep Generative Models"):
    Store anchors (prompt strings) instead of raw examples.
    Generate completions from the *current* model with no-grad,
    then train on those (prompt, completion) pairs *with* grad.
    This reinforces the model's own outputs on past-task inputs.

Usage:
    replay = SyntheticReplay(model, tokenizer, ReplayConfig())

    # Register past-task anchors (e.g. after finishing a training phase):
    replay.add_anchor("Explain photosynthesis:")
    replay.add_anchor("Translate to Swahili:")

    # In your training / micro-finetune loop:
    loss = task_loss + replay.replay_loss()

    # Persist between sessions:
    replay.save("checkpoints/replay.pt")
    replay.load("checkpoints/replay.pt")
"""

from __future__ import annotations

import os
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    # How many past-task prompts to keep (ring buffer)
    num_anchors: int = 20
    # Synthetic completions generated per anchor per replay_loss() call
    samples_per_anchor: int = 2
    # Max tokens for each synthetic completion
    max_new_tokens: int = 64
    # Sampling temperature for synthetic generation (higher = more diverse)
    temperature: float = 0.8
    # Weight of replay regularization relative to the main task loss
    replay_weight: float = 0.3
    # Skip a new anchor if it is too similar to an existing one
    # (measured as approximate char-level edit distance on first 50 chars)
    min_anchor_distance: int = 5


# ---------------------------------------------------------------------------
# SyntheticReplay
# ---------------------------------------------------------------------------

class SyntheticReplay:
    """Privacy-friendly generative replay buffer.

    Stores lightweight prompt anchors instead of raw user data.  At replay
    time it generates synthetic completions from those anchors using the live
    model, then returns a language-modelling loss over those pairs so the
    model keeps fitting its own prior outputs on past-task inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[ReplayConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ReplayConfig()
        self._anchors: deque[str] = deque(maxlen=self.config.num_anchors)
        self._lock = threading.Lock()

        self.stats: Dict[str, Any] = {
            "anchors_stored": 0,
            "replay_steps": 0,
            "total_replay_loss": 0.0,
        }

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    def add_anchor(self, prompt: str) -> bool:
        """Register a past-task prompt as a memory anchor.

        Returns True if accepted, False if too similar to an existing anchor.
        """
        with self._lock:
            if self._is_duplicate(prompt):
                return False
            self._anchors.append(prompt)
            self.stats["anchors_stored"] = len(self._anchors)
            return True

    def add_anchors(self, prompts: List[str]) -> int:
        """Register multiple prompts; return count accepted."""
        return sum(self.add_anchor(p) for p in prompts)

    def num_anchors(self) -> int:
        return len(self._anchors)

    # ------------------------------------------------------------------
    # Replay loss
    # ------------------------------------------------------------------

    def replay_loss(self) -> Optional[torch.Tensor]:
        """Compute replay regularization loss.

        1. Sample a subset of anchors.
        2. For each anchor, generate synthetic completions with no-grad
           (the model as its own "teacher").
        3. Compute supervised language-modelling loss *with grad* over
           those (anchor, synthetic-completion) pairs.
        4. Return ``replay_weight * loss`` — caller adds it to task loss.

        Returns None if no anchors are stored or generation fails.
        """
        if not self._anchors:
            return None

        with self._lock:
            n = max(1, len(self._anchors) // 2)
            anchors = random.sample(list(self._anchors), min(n, len(self._anchors)))

        device = next(self.model.parameters()).device
        batch = self._build_replay_batch(anchors, device)
        if batch is None:
            return None

        self.model.train()
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if loss is not None:
            self.stats["replay_steps"] += 1
            self.stats["total_replay_loss"] += loss.item()
            return self.config.replay_weight * loss
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        state = {"anchors": list(self._anchors), "stats": self.stats}
        torch.save(state, path)
        print(f"[SyntheticReplay] {len(self._anchors)} anchors saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        with self._lock:
            self._anchors = deque(
                state.get("anchors", []), maxlen=self.config.num_anchors
            )
            self.stats.update(state.get("stats", {}))
            self.stats["anchors_stored"] = len(self._anchors)
        print(f"[SyntheticReplay] loaded {len(self._anchors)} anchors from {path}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_duplicate(self, prompt: str) -> bool:
        for anchor in self._anchors:
            if _approx_distance(prompt, anchor) < self.config.min_anchor_distance:
                return True
        return False

    def _build_replay_batch(
        self, anchors: List[str], device: torch.device
    ) -> Optional[Dict[str, torch.Tensor]]:
        pairs: List[tuple] = []
        for anchor in anchors:
            for _ in range(self.config.samples_per_anchor):
                completion = self._generate_completion(anchor, device)
                if completion:
                    pairs.append((anchor, completion))
        if not pairs:
            return None
        return self._tokenize_pairs(pairs, device)

    def _generate_completion(self, prompt: str, device: torch.device) -> Optional[str]:
        """Sample a synthetic completion from the current model (no-grad)."""
        try:
            input_ids = self.tokenizer.encode(prompt, add_bos=True)
            if not input_ids:
                return None
            prompt_len = len(input_ids)
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=device
            )
            self.model.eval()
            generated = list(input_ids)
            past_kv = None

            with torch.no_grad():
                for _ in range(self.config.max_new_tokens):
                    inp = input_tensor[:, -1:] if past_kv is not None else input_tensor
                    out = self.model(
                        input_ids=inp, past_key_values=past_kv, use_cache=True
                    )
                    logits = out["logits"][:, -1, :]
                    if self.config.temperature != 1.0:
                        logits = logits / self.config.temperature
                    probs = torch.softmax(logits.float(), dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                    generated.append(next_id)
                    if next_id == self.tokenizer.eos_id:
                        break
                    input_tensor = torch.tensor(
                        [[next_id]], dtype=torch.long, device=device
                    )
                    past_kv = out.get("past_key_values")

            response_ids = generated[prompt_len:]
            return self.tokenizer.decode(response_ids) if response_ids else None
        except Exception:
            return None

    def _tokenize_pairs(
        self, pairs: List[tuple], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Build padded input_ids + labels (prompt tokens masked with -100)."""
        encoded: List[tuple] = []
        max_len = 0
        pad_id = getattr(self.tokenizer, "pad_id", 0)

        for prompt, response in pairs:
            p_ids = self.tokenizer.encode(prompt, add_bos=True)
            r_ids = self.tokenizer.encode(response, add_eos=True)
            ids = p_ids + r_ids
            # Only compute loss on response tokens
            labels = [-100] * len(p_ids) + r_ids
            encoded.append((ids, labels))
            max_len = max(max_len, len(ids))

        input_ids_list, labels_list = [], []
        for ids, labels in encoded:
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [pad_id] * pad_len)
            labels_list.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long, device=device),
            "labels": torch.tensor(labels_list, dtype=torch.long, device=device),
        }


def _approx_distance(a: str, b: str) -> int:
    """Fast approximate edit distance on first 50 chars (char overlap)."""
    a, b = a[:50].lower(), b[:50].lower()
    common = sum(x == y for x, y in zip(a, b))
    return len(a) + len(b) - 2 * common
