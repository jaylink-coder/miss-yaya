"""Reward Model for Yaya — lightweight scorer for RLHF / online feedback quality.

Architecture:
    YayaForCausalLM backbone (frozen or LoRA-adapted) + a single linear
    "reward head" on top of the final hidden state at the EOS token position.
    Outputs a scalar reward score per (prompt, response) pair.

Two operating modes:
  1. Training  — supervised on preference pairs: (prompt, chosen, rejected)
                 using the Bradley-Terry pairwise loss.
  2. Inference — scores a (prompt, response) pair and returns a scalar.

Integration with OnlineLearner / ElasticGuard:
    The reward model can replace (or augment) human feedback scores:
        rm_score = reward_model.score(prompt, response)
        guard.add_example(prompt, response, score=rm_score)

Usage:
    # Train from preference data
    rm = RewardModel(base_model, config=RewardModelConfig())
    trainer = RewardModelTrainer(rm, optimizer, device)
    trainer.train_step(batch)          # batch: list of (prompt, chosen, rejected)

    # Inference
    score = rm.score(prompt, response, tokenizer, device)
    # Returns float in roughly [-5, 5]; higher = better response

Files:
    src/training/reward_model.py — this file
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardModelConfig:
    """Configuration for the reward model head and training."""
    # Head architecture
    hidden_size: int = 2048            # must match base model hidden_size
    dropout: float = 0.1
    # Training
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    # Scoring
    score_clamp: float = 10.0          # clamp output to [-clamp, clamp]
    # Label smoothing for Bradley-Terry loss (0 = off)
    label_smoothing: float = 0.0


# ---------------------------------------------------------------------------
# Reward head
# ---------------------------------------------------------------------------

class RewardHead(nn.Module):
    """Single linear layer that maps hidden states to a scalar reward."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Args:
            hidden: (batch, hidden_size) — representation of the last token.
        Returns:
            (batch,) scalar reward scores.
        """
        return self.linear(self.drop(hidden)).squeeze(-1)


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """Reward model wrapping a YayaForCausalLM backbone + reward head.

    The backbone's parameters can be fully frozen or LoRA-adapted.
    Only the reward head is trained by default.

    Args:
        backbone:  A YayaForCausalLM instance (or any model that accepts
                   input_ids and returns a dict with 'hidden_states' or
                   that we can pull the last hidden state from).
        config:    RewardModelConfig.
        freeze_backbone: If True (default), only the head is trainable.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: Optional[RewardModelConfig] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.config = config or RewardModelConfig()
        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.head = RewardHead(
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout,
        )

    def _get_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run backbone and extract the hidden state at the last non-pad token.

        Returns:
            (batch, hidden_size) tensor.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Try to get hidden states — handle both our model format and HF format
        if isinstance(outputs, dict):
            if "last_hidden_state" in outputs:
                hidden_states = outputs["last_hidden_state"]
            elif "hidden_states" in outputs and outputs["hidden_states"] is not None:
                hidden_states = outputs["hidden_states"][-1]
            else:
                # Fall back: re-run without loss to get logits, then back-project
                # This path should rarely be hit for a properly configured model.
                raise ValueError(
                    "RewardModel: backbone output does not contain hidden states. "
                    "Pass output_hidden_states=True or use a compatible backbone."
                )
        else:
            # Assume standard HF BaseModelOutput
            hidden_states = outputs.last_hidden_state

        # Index of the last non-padding token per sequence
        if attention_mask is not None:
            # last 1 in each row
            seq_lengths = attention_mask.sum(dim=1) - 1   # (batch,)
        else:
            seq_lengths = torch.full(
                (input_ids.shape[0],), input_ids.shape[1] - 1,
                dtype=torch.long, device=input_ids.device
            )

        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_hidden = hidden_states[batch_idx, seq_lengths]  # (batch, hidden_size)
        return last_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns scalar rewards of shape (batch,)."""
        last_hidden = self._get_last_hidden(input_ids, attention_mask)
        rewards = self.head(last_hidden)
        if self.config.score_clamp > 0:
            rewards = rewards.clamp(-self.config.score_clamp, self.config.score_clamp)
        return rewards

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        response: str,
        tokenizer: Any,
        device: torch.device,
    ) -> float:
        """Score a (prompt, response) pair.  Returns float."""
        self.eval()
        text = prompt + response
        try:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        except Exception:
            return 0.0
        ids = ids[: self.config.max_seq_length]
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        reward = self.forward(input_ids)
        return float(reward.squeeze().item())

    @torch.no_grad()
    def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: Any,
        device: torch.device,
    ) -> List[float]:
        """Score a list of (prompt, response) pairs."""
        return [self.score(p, r, tokenizer, device) for p, r in pairs]


# ---------------------------------------------------------------------------
# Bradley-Terry pairwise loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Pairwise preference loss (Bradley-Terry model).

    Maximises the log-probability that the chosen response is preferred
    over the rejected one:
        L = -log σ(r_chosen − r_rejected)

    With label smoothing:
        L = -(1-ε) * log σ(r_c - r_r) - ε * log σ(r_r - r_c)

    Args:
        reward_chosen:   (batch,) rewards for preferred responses.
        reward_rejected: (batch,) rewards for rejected responses.
        label_smoothing: Smoothing factor ε in [0, 0.5).

    Returns:
        Scalar mean loss.
    """
    logits = reward_chosen - reward_rejected   # (batch,)
    if label_smoothing > 0.0:
        eps = label_smoothing
        loss = -(
            (1.0 - eps) * F.logsigmoid(logits)
            + eps * F.logsigmoid(-logits)
        )
    else:
        loss = -F.logsigmoid(logits)
    return loss.mean()


# ---------------------------------------------------------------------------
# RewardModelTrainer
# ---------------------------------------------------------------------------

class RewardModelTrainer:
    """Lightweight trainer for the reward model on preference pairs.

    Preference batch format:
        Each item is a dict with keys:
            'prompt':   str
            'chosen':   str  (preferred response)
            'rejected': str  (dispreferred response)

    The trainer tokenises these, runs the reward model on both, and
    computes the Bradley-Terry loss.
    """

    def __init__(
        self,
        model: RewardModel,
        tokenizer: Any,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        cfg = model.config
        self.optimizer = optimizer or torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )
        self.cfg = cfg

    def _encode(self, prompt: str, response: str) -> torch.Tensor:
        """Tokenise and return input_ids on device."""
        text = prompt + response
        try:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        except Exception:
            ids = [0]
        ids = ids[: self.cfg.max_seq_length]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def train_step(self, batch: List[dict]) -> float:
        """One gradient step on a preference batch.

        Args:
            batch: List of dicts with 'prompt', 'chosen', 'rejected'.

        Returns:
            Loss value (float).
        """
        self.model.train()
        chosen_rewards = []
        rejected_rewards = []

        for item in batch:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            ids_c = self._encode(prompt, chosen)
            ids_r = self._encode(prompt, rejected)

            r_c = self.model(ids_c)
            r_r = self.model(ids_r)
            chosen_rewards.append(r_c)
            rejected_rewards.append(r_r)

        if not chosen_rewards:
            return 0.0

        chosen_t = torch.cat(chosen_rewards)      # (batch,)
        rejected_t = torch.cat(rejected_rewards)  # (batch,)

        loss = bradley_terry_loss(
            chosen_t, rejected_t,
            label_smoothing=self.cfg.label_smoothing,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg.max_grad_norm,
        )
        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def eval_accuracy(self, batch: List[dict]) -> float:
        """Fraction of pairs where reward_chosen > reward_rejected."""
        self.model.eval()
        correct = 0
        for item in batch:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            r_c = self.model(self._encode(prompt, chosen))
            r_r = self.model(self._encode(prompt, rejected))
            if r_c.item() > r_r.item():
                correct += 1
        return correct / max(len(batch), 1)

    def save(self, path: str) -> None:
        """Save reward head weights (backbone is frozen, no need to save)."""
        torch.save({"head": self.model.head.state_dict()}, path)
        print(f"[RewardModel] saved to {path}")

    def load(self, path: str) -> None:
        """Load reward head weights."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.head.load_state_dict(state["head"])
        print(f"[RewardModel] loaded from {path}")
