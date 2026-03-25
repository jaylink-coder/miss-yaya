"""Curriculum Learning for Yaya.

Implements difficulty-aware training schedules:

  1. DifficultyScorer   — scores a (prompt, response) pair by multiple signals
  2. CurriculumDataset  — wraps a list of examples, sorted/filtered by difficulty
  3. CurriculumSchedule — controls which difficulty band is active at each step
  4. CurriculumSampler  — PyTorch Sampler that implements the schedule

Biological analogy: learning progresses from simple → complex, just as a child
learns addition before calculus.  The model builds a strong foundation on easy
examples first, then gradually exposes harder ones, preventing early catastrophic
confusion on examples it can't yet learn from.

Usage:
    scorer = DifficultyScorer()
    scores = scorer.score_batch(examples)          # list of floats in [0, 1]

    curriculum = CurriculumDataset(examples, scorer)
    curriculum.sort_by_difficulty()

    schedule = CurriculumSchedule(
        total_steps=100_000,
        warmup_easy_steps=10_000,   # first 10k steps: easy only
        strategy="linear",          # linearly expand difficulty window
    )

    sampler = CurriculumSampler(curriculum, schedule)
    loader  = DataLoader(curriculum, batch_sampler=...)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Difficulty signals
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Rough whitespace-based token estimate."""
    return len(text.split())


def _vocab_diversity(text: str) -> float:
    """Type-token ratio: unique_words / total_words.  Higher → harder."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _sentence_complexity(text: str) -> float:
    """Average sentence length (words).  Longer → harder."""
    import re
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    total = sum(len(s.split()) for s in sentences)
    return total / len(sentences)


def _rare_word_fraction(text: str, common_words: Optional[set] = None) -> float:
    """Fraction of words not in common word list.  More rare → harder."""
    if common_words is None:
        return 0.0
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    rare = sum(1 for t in tokens if t not in common_words)
    return rare / len(tokens)


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------

@dataclass
class DifficultyConfig:
    """Weights for each difficulty signal.  Must sum to ≤ 1.0."""
    length_weight: float = 0.25         # normalised sequence length
    diversity_weight: float = 0.20      # type-token ratio
    complexity_weight: float = 0.25     # avg sentence length
    rare_word_weight: float = 0.10      # fraction of rare words
    loss_weight: float = 0.20           # model loss (if available; else ignored)
    # Length normalisation reference (tokens)
    max_length_ref: int = 512
    max_sentence_length_ref: float = 40.0   # sentences this long → score 1.0


class DifficultyScorer:
    """Assigns difficulty scores in [0, 1] to training examples.

    An example is a dict with at minimum a 'text' key (or 'prompt'+'response').
    Loss-based scoring requires calling `score_with_loss()` instead of `score()`.

    Score 0.0 = very easy, 1.0 = very hard.
    """

    COMMON_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "do", "did", "does", "will", "would", "can", "could", "should", "may",
        "might", "it", "its", "this", "that", "these", "those", "i", "you",
        "he", "she", "we", "they", "my", "your", "his", "her", "our", "their",
        "what", "which", "who", "when", "where", "why", "how", "not", "with",
        "from", "by", "as", "if", "then", "than", "so", "up", "out", "about",
    }

    def __init__(self, config: Optional[DifficultyConfig] = None):
        self.config = config or DifficultyConfig()

    def _get_text(self, example: dict) -> str:
        if "text" in example:
            return example["text"]
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        return (prompt + " " + response).strip()

    def score(self, example: dict, model_loss: Optional[float] = None) -> float:
        """Score a single example.  Returns float in [0, 1]."""
        text = self._get_text(example)
        if not text:
            return 0.0

        cfg = self.config

        # 1. Length score
        n_tokens = _token_count(text)
        length_score = min(1.0, n_tokens / max(cfg.max_length_ref, 1))

        # 2. Vocabulary diversity
        diversity_score = _vocab_diversity(text)

        # 3. Sentence complexity
        avg_sent = _sentence_complexity(text)
        complexity_score = min(1.0, avg_sent / max(cfg.max_sentence_length_ref, 1.0))

        # 4. Rare words
        rare_score = _rare_word_fraction(text, self.COMMON_WORDS)

        # Weighted sum of heuristic signals
        total_heuristic_weight = (
            cfg.length_weight + cfg.diversity_weight +
            cfg.complexity_weight + cfg.rare_word_weight
        )
        score = (
            cfg.length_weight * length_score
            + cfg.diversity_weight * diversity_score
            + cfg.complexity_weight * complexity_score
            + cfg.rare_word_weight * rare_score
        )

        # 5. Optional model loss
        if model_loss is not None and cfg.loss_weight > 0.0:
            # Normalise: assume perplexity range ~[1, 20]; clamp to [0, 1]
            loss_score = min(1.0, max(0.0, (model_loss - 1.0) / 19.0))
            score = score + cfg.loss_weight * loss_score
            total_heuristic_weight += cfg.loss_weight

        # Normalise so total weight = 1.0
        if total_heuristic_weight > 0:
            score /= total_heuristic_weight

        return float(min(1.0, max(0.0, score)))

    def score_batch(
        self,
        examples: List[dict],
        losses: Optional[List[float]] = None,
    ) -> List[float]:
        """Score a batch of examples."""
        if losses is None:
            return [self.score(ex) for ex in examples]
        return [self.score(ex, loss) for ex, loss in zip(examples, losses)]


# ---------------------------------------------------------------------------
# CurriculumDataset
# ---------------------------------------------------------------------------

class CurriculumDataset:
    """A list of examples ordered by difficulty.

    Wraps a raw list[dict] (each dict having at minimum 'text' or
    'prompt'+'response') and annotates each with a difficulty score.

    Supports slicing by difficulty band for staged training.
    """

    def __init__(
        self,
        examples: List[dict],
        scorer: Optional[DifficultyScorer] = None,
        losses: Optional[List[float]] = None,
    ):
        self.scorer = scorer or DifficultyScorer()
        self.examples = list(examples)
        self.scores: List[float] = []
        if examples:
            self.scores = self.scorer.score_batch(examples, losses)

    def sort_by_difficulty(self) -> None:
        """Sort examples from easiest to hardest (in-place)."""
        paired = sorted(zip(self.scores, self.examples), key=lambda x: x[0])
        self.scores = [s for s, _ in paired]
        self.examples = [e for _, e in paired]

    def difficulty_band(self, low: float, high: float) -> "CurriculumDataset":
        """Return a new CurriculumDataset with only examples in [low, high] difficulty."""
        filtered = [
            (s, e) for s, e in zip(self.scores, self.examples) if low <= s <= high
        ]
        if not filtered:
            return CurriculumDataset([], self.scorer)
        scores, examples = zip(*filtered)
        ds = CurriculumDataset.__new__(CurriculumDataset)
        ds.scorer = self.scorer
        ds.examples = list(examples)
        ds.scores = list(scores)
        return ds

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# CurriculumSchedule
# ---------------------------------------------------------------------------

@dataclass
class CurriculumSchedule:
    """Controls which difficulty window is active at each training step.

    Strategies:
    - "linear"      — window expands linearly from [0, easy_ceiling] to [0, 1.0]
    - "step"        — fixed stages (easy / medium / hard) at step thresholds
    - "competence"  — window expands only when model loss is low enough
    """
    total_steps: int = 100_000
    warmup_easy_steps: int = 10_000    # steps before any hard examples
    strategy: str = "linear"           # "linear" | "step" | "competence"
    easy_ceiling: float = 0.4          # max difficulty during warmup
    # Step-based thresholds (only used when strategy="step")
    medium_start_step: int = 20_000    # step to include medium examples
    hard_start_step: int = 50_000      # step to include all examples
    medium_ceiling: float = 0.7        # max difficulty during medium phase
    # Competence-based: only advance when mean_loss falls below threshold
    competence_loss_threshold: float = 2.5

    def active_window(
        self,
        current_step: int,
        mean_loss: Optional[float] = None,
    ) -> tuple[float, float]:
        """Return (low, high) difficulty window for the current step."""

        if self.strategy == "linear":
            if current_step < self.warmup_easy_steps:
                progress = current_step / max(self.warmup_easy_steps, 1)
                high = self.easy_ceiling * progress
                high = max(self.easy_ceiling * 0.1, high)  # at least 10% of ceiling
            else:
                remaining = self.total_steps - self.warmup_easy_steps
                step_in_phase = current_step - self.warmup_easy_steps
                progress = min(1.0, step_in_phase / max(remaining, 1))
                high = self.easy_ceiling + (1.0 - self.easy_ceiling) * progress
            return (0.0, float(min(1.0, high)))

        elif self.strategy == "step":
            if current_step < self.warmup_easy_steps:
                return (0.0, self.easy_ceiling)
            elif current_step < self.medium_start_step:
                return (0.0, self.easy_ceiling)
            elif current_step < self.hard_start_step:
                return (0.0, self.medium_ceiling)
            else:
                return (0.0, 1.0)

        elif self.strategy == "competence":
            # Only advance when mean_loss is low enough
            if mean_loss is not None and mean_loss < self.competence_loss_threshold:
                return (0.0, 1.0)
            return (0.0, self.easy_ceiling)

        return (0.0, 1.0)  # fallback: all difficulties


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """Yields indices from CurriculumDataset according to the active schedule window.

    Designed to be used as the sampler in a DataLoader:
        sampler = CurriculumSampler(dataset, schedule)
        for idx in sampler:
            batch = dataset[idx]
    """

    def __init__(
        self,
        dataset: CurriculumDataset,
        schedule: CurriculumSchedule,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.schedule = schedule
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._step = 0
        self._mean_loss: Optional[float] = None

    def update_step(self, step: int, mean_loss: Optional[float] = None) -> None:
        """Call after each training step to update the active window."""
        self._step = step
        if mean_loss is not None:
            self._mean_loss = mean_loss

    def active_indices(self) -> List[int]:
        """Return the list of valid indices for the current step."""
        low, high = self.schedule.active_window(self._step, self._mean_loss)
        indices = [
            i for i, s in enumerate(self.dataset.scores) if low <= s <= high
        ]
        if not indices:
            # Fallback: include all
            indices = list(range(len(self.dataset)))
        return indices

    def __iter__(self) -> Iterator[int]:
        indices = self.active_indices()
        if self.shuffle:
            self._rng.shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        return len(self.active_indices())


# ---------------------------------------------------------------------------
# Convenience: build a curriculum from raw data
# ---------------------------------------------------------------------------

def build_curriculum(
    examples: List[dict],
    schedule_kwargs: Optional[dict] = None,
    scorer_config: Optional[DifficultyConfig] = None,
    losses: Optional[List[float]] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[CurriculumDataset, CurriculumSampler]:
    """One-shot helper to create a sorted CurriculumDataset and its Sampler.

    Args:
        examples:        List of training examples (dicts with 'text' or 'prompt'/'response').
        schedule_kwargs: Kwargs forwarded to CurriculumSchedule.
        scorer_config:   Custom DifficultyConfig; defaults if None.
        losses:          Optional per-example model losses for better scoring.
        shuffle:         Shuffle within the active difficulty window.
        seed:            RNG seed for reproducible shuffling.

    Returns:
        (CurriculumDataset, CurriculumSampler) — ready to plug into a DataLoader.
    """
    scorer = DifficultyScorer(config=scorer_config)
    dataset = CurriculumDataset(examples, scorer, losses=losses)
    dataset.sort_by_difficulty()

    schedule = CurriculumSchedule(**(schedule_kwargs or {}))
    sampler = CurriculumSampler(dataset, schedule, shuffle=shuffle, seed=seed)

    return dataset, sampler
