"""Loss functions for Yaya model training.

Provides standard cross-entropy loss with label smoothing option,
and z-loss regularization for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalLMLoss(nn.Module):
    """Cross-entropy loss for causal language modeling.

    Features:
    - Ignores padding tokens (index -100)
    - Optional label smoothing
    - Optional z-loss for logit regularization (training stability)
    """

    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.0,
        z_loss_weight: float = 0.0,
    ):
        """Initialize loss function.

        Args:
            vocab_size: Vocabulary size.
            label_smoothing: Label smoothing factor (0 = no smoothing).
            z_loss_weight: Weight for z-loss regularization (0 = disabled).
                          Penalizes large logit values for training stability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.z_loss_weight = z_loss_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            labels: Target token IDs [batch, seq_len]

        Returns:
            Scalar loss tensor.
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Cross-entropy with label smoothing
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

        # Z-loss: penalize large logit magnitudes for stability
        if self.z_loss_weight > 0:
            # Only compute on non-padding positions
            mask = shift_labels != -100
            if mask.any():
                z_loss = torch.logsumexp(shift_logits[mask], dim=-1).pow(2).mean()
                loss = loss + self.z_loss_weight * z_loss

        return loss
