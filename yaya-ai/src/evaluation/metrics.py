"""Metric computations for model evaluation.

Provides accuracy, perplexity, F1, and other metrics
used in LLM benchmarking.
"""

import math
from typing import List, Dict, Optional
from collections import Counter


def accuracy(predictions: List[int], targets: List[int]) -> float:
    """Compute exact match accuracy."""
    if not predictions or not targets:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets)


def perplexity(avg_loss: float) -> float:
    """Compute perplexity from average cross-entropy loss."""
    return math.exp(min(avg_loss, 20))  # Clamp to prevent overflow


def f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """Check if prediction exactly matches reference."""
    if normalize:
        prediction = prediction.strip().lower()
        reference = reference.strip().lower()
    return 1.0 if prediction == reference else 0.0


def multiple_choice_accuracy(
    logits_per_choice: List[float],
    correct_index: int,
) -> float:
    """Evaluate multiple choice by selecting the highest-scoring option."""
    if not logits_per_choice:
        return 0.0
    predicted = max(range(len(logits_per_choice)), key=lambda i: logits_per_choice[i])
    return 1.0 if predicted == correct_index else 0.0


def aggregate_metrics(metric_lists: Dict[str, List[float]]) -> Dict[str, float]:
    """Aggregate per-sample metrics into summary statistics."""
    result = {}
    for name, values in metric_lists.items():
        if values:
            result[f"{name}_mean"] = sum(values) / len(values)
            result[f"{name}_count"] = len(values)
    return result
