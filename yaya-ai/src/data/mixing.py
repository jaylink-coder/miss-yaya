"""Data mixing and sampling strategy.

Controls the proportion of different data sources during training
(web text, code, books, math, etc.) to shape model capabilities.
"""

import random
from typing import Dict, List, Optional, Iterator, Any
from torch.utils.data import IterableDataset, Dataset


# Default data mix ratios for pre-training
DEFAULT_MIX_RATIOS = {
    "web_text": 0.45,       # Filtered web crawl — general knowledge
    "code": 0.12,           # GitHub code — programming + logical reasoning
    "books": 0.12,          # Books — coherent long-form text
    "academic": 0.08,       # Academic papers — technical knowledge
    "math": 0.06,           # Math datasets — mathematical reasoning
    "conversational": 0.07, # Forums, Q&A — dialogue capability
    "domain_business": 0.05,# Business docs — enterprise specialization
    "multilingual": 0.05,   # Non-English text — multilingual capability
}


class MixedDataSource(IterableDataset):
    """Weighted mixture of multiple data sources.

    Samples from different data sources according to configurable
    mix ratios. This is critical for controlling model capabilities.

    Example:
        sources = {
            "web_text": StreamingTextDataset("data/web/"),
            "code": StreamingTextDataset("data/code/"),
            "books": StreamingTextDataset("data/books/"),
        }
        ratios = {"web_text": 0.5, "code": 0.3, "books": 0.2}
        mixed = MixedDataSource(sources, ratios)
    """

    def __init__(
        self,
        sources: Dict[str, IterableDataset],
        mix_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """Initialize mixed data source.

        Args:
            sources: Dict mapping source name to dataset.
            mix_ratios: Dict mapping source name to sampling weight.
                        Weights are normalized to sum to 1.
            seed: Random seed for reproducible sampling.
        """
        super().__init__()
        self.sources = sources
        self.seed = seed

        # Use provided ratios or defaults
        if mix_ratios is None:
            # Equal weights for all sources
            n = len(sources)
            self.mix_ratios = {name: 1.0 / n for name in sources}
        else:
            # Normalize weights to sum to 1
            total = sum(mix_ratios.get(name, 0.0) for name in sources)
            self.mix_ratios = {
                name: mix_ratios.get(name, 0.0) / total for name in sources
            }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield samples from mixed sources according to ratios."""
        rng = random.Random(self.seed)

        # Create iterators for each source
        iterators = {}
        for name, dataset in self.sources.items():
            iterators[name] = iter(dataset)

        source_names = list(self.sources.keys())
        weights = [self.mix_ratios[name] for name in source_names]

        while iterators:
            # Weighted random source selection
            chosen_name = rng.choices(source_names, weights=weights, k=1)[0]

            try:
                sample = next(iterators[chosen_name])
                yield sample
            except StopIteration:
                # Source exhausted — remove it and rebalance
                idx = source_names.index(chosen_name)
                source_names.pop(idx)
                weights.pop(idx)
                del iterators[chosen_name]

                if not source_names:
                    break

                # Re-normalize weights
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]


class CurriculumScheduler:
    """Adjust data mix ratios during training (curriculum learning).

    Can shift the data distribution over training steps, e.g.:
    - More web text early, more code/math later
    - Increase domain-specific data as training progresses
    """

    def __init__(
        self,
        initial_ratios: Dict[str, float],
        final_ratios: Dict[str, float],
        total_steps: int,
        warmup_steps: int = 0,
    ):
        """Initialize curriculum scheduler.

        Args:
            initial_ratios: Mix ratios at start of training.
            final_ratios: Mix ratios at end of training.
            total_steps: Total training steps.
            warmup_steps: Steps before curriculum starts changing.
        """
        self.initial_ratios = initial_ratios
        self.final_ratios = final_ratios
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_ratios(self, step: int) -> Dict[str, float]:
        """Get mix ratios for the current training step.

        Linearly interpolates between initial and final ratios.

        Args:
            step: Current training step.

        Returns:
            Dict of source name to sampling ratio.
        """
        if step <= self.warmup_steps:
            return self.initial_ratios.copy()

        # Linear interpolation
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))

        ratios = {}
        all_keys = set(self.initial_ratios.keys()) | set(self.final_ratios.keys())
        for key in all_keys:
            start = self.initial_ratios.get(key, 0.0)
            end = self.final_ratios.get(key, 0.0)
            ratios[key] = start + (end - start) * progress

        # Normalize
        total = sum(ratios.values())
        if total > 0:
            ratios = {k: v / total for k, v in ratios.items()}

        return ratios
