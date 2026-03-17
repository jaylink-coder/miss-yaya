"""Shard-level data mixer for combining tokenized data from multiple categories.

After tokenization, each category has its own binary shards. This module
combines them into final training shards according to configured mix ratios,
with support for shuffling and reproducibility.
"""

import os
import glob
import json
import random
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


class ShardMixer:
    """Combine tokenized shards from multiple categories into training data.

    Reads binary token shards from category-specific directories,
    samples according to mix ratios, shuffles, and writes final
    training shards.
    """

    def __init__(
        self,
        category_dirs: Dict[str, str],
        mix_ratios: Dict[str, float],
        output_dir: str,
        shard_size: int = 100_000_000,
        seed: int = 42,
        dtype: str = "uint16",
    ):
        """Initialize shard mixer.

        Args:
            category_dirs: Dict mapping category name to directory of .bin shards.
            mix_ratios: Dict mapping category name to target ratio (0-1).
            output_dir: Directory for final mixed shards.
            shard_size: Tokens per output shard.
            seed: Random seed for reproducibility.
            dtype: NumPy dtype for token storage.
        """
        self.category_dirs = category_dirs
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.seed = seed
        self.dtype = getattr(np, dtype)

        # Normalize ratios
        total = sum(mix_ratios.get(k, 0.0) for k in category_dirs)
        self.mix_ratios = {
            k: mix_ratios.get(k, 0.0) / max(total, 1e-8) for k in category_dirs
        }

        os.makedirs(output_dir, exist_ok=True)

    def _load_category_tokens(self, category: str, max_tokens: Optional[int] = None) -> np.ndarray:
        """Load all tokens from a category's shards."""
        cat_dir = self.category_dirs[category]
        shard_files = sorted(glob.glob(os.path.join(cat_dir, "*.bin")))

        if not shard_files:
            print(f"  Warning: no shards found in {cat_dir}")
            return np.array([], dtype=self.dtype)

        all_tokens = []
        total = 0
        for path in shard_files:
            tokens = np.fromfile(path, dtype=self.dtype)
            all_tokens.append(tokens)
            total += len(tokens)
            if max_tokens and total >= max_tokens:
                break

        combined = np.concatenate(all_tokens)
        if max_tokens and len(combined) > max_tokens:
            combined = combined[:max_tokens]
        return combined

    def mix(self, total_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Mix shards from all categories into final training data.

        Args:
            total_tokens: Target total tokens. If None, uses all available data.

        Returns:
            Dict with mixing statistics.
        """
        rng = random.Random(self.seed)
        np_rng = np.random.RandomState(self.seed)

        print(f"Mixing categories: {list(self.category_dirs.keys())}")
        print(f"Ratios: {self.mix_ratios}")
        print()

        # Calculate token budget per category
        if total_tokens is None:
            # Scan to find total available
            available = {}
            for cat, cat_dir in self.category_dirs.items():
                shard_files = glob.glob(os.path.join(cat_dir, "*.bin"))
                total_bytes = sum(os.path.getsize(f) for f in shard_files)
                available[cat] = total_bytes // np.dtype(self.dtype).itemsize
            total_tokens = sum(available.values())
            print(f"Available tokens: {total_tokens:,}")

        budgets = {}
        for cat, ratio in self.mix_ratios.items():
            budgets[cat] = int(total_tokens * ratio)

        print(f"Token budgets per category:")
        for cat, budget in budgets.items():
            print(f"  {cat}: {budget:,} ({self.mix_ratios[cat]*100:.1f}%)")
        print()

        # Load tokens per category
        category_tokens = {}
        for cat, budget in budgets.items():
            print(f"  Loading {cat}...", end="", flush=True)
            tokens = self._load_category_tokens(cat, max_tokens=budget)

            # If we have less than budget, use what we have
            # If we have more, subsample
            if len(tokens) > budget:
                tokens = tokens[:budget]

            category_tokens[cat] = tokens
            print(f" {len(tokens):,} tokens")

        # Combine all tokens
        all_tokens = np.concatenate(list(category_tokens.values()))
        print(f"\nTotal combined: {len(all_tokens):,} tokens")

        # Shuffle at document level (approximate: shuffle in large blocks)
        block_size = 2048  # Tokens per block
        num_blocks = len(all_tokens) // block_size
        if num_blocks > 1:
            blocks = all_tokens[:num_blocks * block_size].reshape(num_blocks, block_size)
            np_rng.shuffle(blocks)
            all_tokens[:num_blocks * block_size] = blocks.flatten()
            print(f"Shuffled {num_blocks:,} blocks of {block_size} tokens")

        # Write output shards
        shard_paths = []
        shard_idx = 0
        for offset in range(0, len(all_tokens), self.shard_size):
            chunk = all_tokens[offset:offset + self.shard_size]
            shard_path = os.path.join(self.output_dir, f"train_shard_{shard_idx:05d}.bin")
            chunk.tofile(shard_path)
            shard_paths.append(shard_path)
            shard_idx += 1

        print(f"Wrote {len(shard_paths)} shards to {self.output_dir}")

        # Stats
        stats = {
            "total_tokens": int(len(all_tokens)),
            "num_shards": len(shard_paths),
            "categories": {
                cat: {
                    "tokens": int(len(tokens)),
                    "ratio_target": float(self.mix_ratios.get(cat, 0)),
                    "ratio_actual": int(len(tokens)) / max(int(len(all_tokens)), 1),
                }
                for cat, tokens in category_tokens.items()
            },
        }

        # Save stats
        stats_path = os.path.join(self.output_dir, "mix_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return stats


class DatasetStatsReporter:
    """Generate comprehensive statistics for the training dataset.

    Reports on token counts, category distributions, shard sizes,
    vocabulary coverage, and data quality metrics.
    """

    def __init__(self, data_dir: str, tokenizer_path: Optional[str] = None):
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path

    def compute_shard_stats(self) -> Dict[str, Any]:
        """Compute statistics across all shards."""
        shard_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.bin"), recursive=True))

        if not shard_files:
            return {"error": f"No .bin files found in {self.data_dir}"}

        total_tokens = 0
        shard_sizes = []

        for path in shard_files:
            size = os.path.getsize(path) // 2  # uint16
            total_tokens += size
            shard_sizes.append(size)

        return {
            "total_tokens": total_tokens,
            "total_tokens_billions": total_tokens / 1e9,
            "num_shards": len(shard_files),
            "avg_shard_tokens": total_tokens // max(len(shard_files), 1),
            "min_shard_tokens": min(shard_sizes) if shard_sizes else 0,
            "max_shard_tokens": max(shard_sizes) if shard_sizes else 0,
            "total_size_gb": sum(os.path.getsize(f) for f in shard_files) / (1024**3),
        }

    def compute_vocab_coverage(self, sample_size: int = 10_000_000) -> Dict[str, Any]:
        """Estimate vocabulary coverage from a sample of tokens.

        Args:
            sample_size: Number of tokens to sample.

        Returns:
            Dict with vocabulary statistics.
        """
        shard_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.bin"), recursive=True))
        if not shard_files:
            return {}

        # Sample tokens from first shard(s)
        sampled = []
        for path in shard_files:
            tokens = np.fromfile(path, dtype=np.uint16)
            sampled.extend(tokens.tolist())
            if len(sampled) >= sample_size:
                break

        sampled = sampled[:sample_size]
        unique_tokens = set(sampled)

        # Token frequency distribution
        from collections import Counter
        freq = Counter(sampled)
        top_20 = freq.most_common(20)

        return {
            "sample_size": len(sampled),
            "unique_tokens": len(unique_tokens),
            "max_token_id": max(sampled) if sampled else 0,
            "top_20_tokens": [(int(t), c) for t, c in top_20],
        }

    def full_report(self) -> Dict[str, Any]:
        """Generate a comprehensive dataset report."""
        report = {
            "shard_stats": self.compute_shard_stats(),
        }

        # Try vocab coverage if data exists
        if report["shard_stats"].get("total_tokens", 0) > 0:
            report["vocab_coverage"] = self.compute_vocab_coverage()

        # Check for mix stats
        mix_stats_path = os.path.join(self.data_dir, "mix_stats.json")
        if os.path.exists(mix_stats_path):
            with open(mix_stats_path) as f:
                report["mix_stats"] = json.load(f)

        return report

    def print_report(self):
        """Print a human-readable report."""
        report = self.full_report()

        print(f"\n{'='*60}")
        print(f"  DATASET REPORT: {self.data_dir}")
        print(f"{'='*60}")

        stats = report.get("shard_stats", {})
        print(f"\n  Tokens: {stats.get('total_tokens', 0):,} ({stats.get('total_tokens_billions', 0):.2f}B)")
        print(f"  Shards: {stats.get('num_shards', 0)}")
        print(f"  Size: {stats.get('total_size_gb', 0):.2f} GB")
        print(f"  Avg shard: {stats.get('avg_shard_tokens', 0):,} tokens")

        vocab = report.get("vocab_coverage", {})
        if vocab:
            print(f"\n  Unique tokens (in sample): {vocab.get('unique_tokens', 0):,}")
            print(f"  Max token ID: {vocab.get('max_token_id', 0)}")

        mix = report.get("mix_stats", {})
        if mix:
            print(f"\n  Category breakdown:")
            for cat, cat_stats in mix.get("categories", {}).items():
                print(f"    {cat}: {cat_stats['tokens']:,} tokens ({cat_stats['ratio_actual']*100:.1f}%)")

        print(f"\n{'='*60}")
