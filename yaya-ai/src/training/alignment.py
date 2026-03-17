"""RLHF and DPO alignment training for Yaya model.

Implements Direct Preference Optimization (DPO) for aligning the model
with human preferences without requiring a separate reward model.

DPO Reference: https://arxiv.org/abs/2305.18290

Also includes reward model training utilities for traditional RLHF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    beta: float = 0.1               # KL penalty coefficient
    label_smoothing: float = 0.0    # Label smoothing for DPO loss
    loss_type: str = "sigmoid"      # 'sigmoid' (standard DPO) or 'hinge' or 'ipo'
    reference_free: bool = False    # If True, skip reference model log probs


class DPOTrainer:
    """Direct Preference Optimization trainer.

    DPO directly optimizes the policy model using preference pairs
    (chosen vs rejected responses) without a separate reward model.

    The DPO loss is:
        L = -log sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x)
                                - log_ref(y_w|x) + log_ref(y_l|x)))

    where y_w = chosen, y_l = rejected, pi = policy, ref = reference.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        config: Optional[DPOConfig] = None,
    ):
        """Initialize DPO trainer.

        Args:
            policy_model: The model being optimized.
            reference_model: Frozen reference model (usually the SFT checkpoint).
                           If None and reference_free=False, creates a copy.
            config: DPO configuration.
        """
        self.policy_model = policy_model
        self.config = config or DPOConfig()

        if reference_model is not None:
            self.reference_model = reference_model
        elif not self.config.reference_free:
            # Deep copy the policy as reference and freeze it
            import copy
            self.reference_model = copy.deepcopy(policy_model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = None

    @torch.no_grad()
    def _get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token log probabilities under a model.

        Args:
            model: The model to compute log probs for.
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Sum of log probs for each sequence [batch]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        if attention_mask is not None:
            mask = attention_mask[:, 1:]  # Shift to match
            per_token_log_probs = per_token_log_probs * mask

        # Sum log probs per sequence
        return per_token_log_probs.sum(dim=-1)

    def _get_log_probs_with_grad(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Same as _get_log_probs but with gradients enabled."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        if attention_mask is not None:
            mask = attention_mask[:, 1:]
            per_token_log_probs = per_token_log_probs * mask

        return per_token_log_probs.sum(dim=-1)

    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the DPO loss for a batch of preference pairs.

        Args:
            chosen_input_ids: Chosen response tokens [batch, seq_len]
            chosen_labels: Chosen response labels [batch, seq_len]
            rejected_input_ids: Rejected response tokens [batch, seq_len]
            rejected_labels: Rejected response labels [batch, seq_len]
            chosen_attention_mask: Mask for chosen [batch, seq_len]
            rejected_attention_mask: Mask for rejected [batch, seq_len]

        Returns:
            Dict with 'loss', 'chosen_reward', 'rejected_reward', 'accuracy'.
        """
        # Policy model log probs (with gradients)
        policy_chosen_logps = self._get_log_probs_with_grad(
            self.policy_model, chosen_input_ids, chosen_labels, chosen_attention_mask
        )
        policy_rejected_logps = self._get_log_probs_with_grad(
            self.policy_model, rejected_input_ids, rejected_labels, rejected_attention_mask
        )

        # Reference model log probs (no gradients)
        if self.reference_model is not None:
            ref_chosen_logps = self._get_log_probs(
                self.reference_model, chosen_input_ids, chosen_labels, chosen_attention_mask
            )
            ref_rejected_logps = self._get_log_probs(
                self.reference_model, rejected_input_ids, rejected_labels, rejected_attention_mask
            )
        else:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        # Log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps
        logits = chosen_logratios - rejected_logratios

        # Compute loss based on type
        if self.config.loss_type == "sigmoid":
            # Standard DPO loss
            loss = -F.logsigmoid(self.config.beta * logits).mean()
        elif self.config.loss_type == "hinge":
            # Hinge loss variant
            loss = torch.relu(1.0 - self.config.beta * logits).mean()
        elif self.config.loss_type == "ipo":
            # IPO (Identity Preference Optimization) loss
            loss = (logits - 1.0 / (2.0 * self.config.beta)).pow(2).mean()
        else:
            loss = -F.logsigmoid(self.config.beta * logits).mean()

        # Label smoothing
        if self.config.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-self.config.beta * logits).mean()
            loss = (1.0 - self.config.label_smoothing) * loss + self.config.label_smoothing * smooth_loss

        # Implicit rewards for logging
        chosen_rewards = self.config.beta * chosen_logratios.detach()
        rejected_rewards = self.config.beta * rejected_logratios.detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": loss,
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean(),
            "accuracy": reward_accuracy,
        }


class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for DPO training with preference pairs.

    Loads JSONL files where each line contains:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
    ):
        import json
        from pathlib import Path

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        if Path(data_path).is_file():
            files = [data_path]
        else:
            files = sorted(Path(data_path).glob("*.jsonl"))

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        print(f"PreferenceDataset loaded: {len(self.samples):,} pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair.

        Returns dict with chosen_input_ids, chosen_labels, chosen_attention_mask,
        rejected_input_ids, rejected_labels, rejected_attention_mask.
        """
        sample = self.samples[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # Tokenize chosen
        chosen_text = prompt + chosen
        chosen_ids = self.tokenizer.encode(chosen_text, add_bos=True, add_eos=True)
        chosen_ids = chosen_ids[:self.max_seq_length]

        # Tokenize rejected
        rejected_text = prompt + rejected
        rejected_ids = self.tokenizer.encode(rejected_text, add_bos=True, add_eos=True)
        rejected_ids = rejected_ids[:self.max_seq_length]

        # Create labels (mask prompt tokens with -100)
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        prompt_len = len(prompt_ids)

        chosen_labels = list(chosen_ids)
        chosen_labels[:prompt_len] = [-100] * min(prompt_len, len(chosen_labels))

        rejected_labels = list(rejected_ids)
        rejected_labels[:prompt_len] = [-100] * min(prompt_len, len(rejected_labels))

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "chosen_attention_mask": torch.ones(len(chosen_ids), dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            "rejected_attention_mask": torch.ones(len(rejected_ids), dtype=torch.long),
        }


class RewardModel(nn.Module):
    """Reward model for RLHF (scalar reward prediction).

    Takes a language model backbone and adds a scalar head
    that predicts a reward score for a given input sequence.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int):
        """Initialize reward model.

        Args:
            base_model: Pre-trained language model backbone.
            hidden_size: Hidden dimension of the backbone.
        """
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward score for input sequences.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Reward scores [batch]
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["logits"]  # Actually we need hidden states

        # Use last non-padding token's hidden state
        if attention_mask is not None:
            # Find the last non-padding position
            seq_lengths = attention_mask.sum(dim=-1) - 1
            batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1]

        rewards = self.reward_head(last_hidden).squeeze(-1)
        return rewards

    def compute_reward_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Bradley-Terry reward model loss.

        Args:
            chosen_ids: Chosen response token IDs [batch, seq_len]
            rejected_ids: Rejected response token IDs [batch, seq_len]
            chosen_mask: Attention mask for chosen
            rejected_mask: Attention mask for rejected

        Returns:
            Dict with 'loss', 'chosen_reward', 'rejected_reward', 'accuracy'.
        """
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)

        # Bradley-Terry loss: chosen should score higher than rejected
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return {
            "loss": loss,
            "chosen_reward": chosen_rewards.mean(),
            "rejected_reward": rejected_rewards.mean(),
            "accuracy": accuracy,
        }
