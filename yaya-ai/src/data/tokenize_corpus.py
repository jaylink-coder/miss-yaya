"""Large-scale corpus tokenization into memory-mapped shards.

Reads processed JSONL files, tokenizes with the Yaya tokenizer,
and writes binary shards for efficient training data loading.
Supports parallel processing and progress tracking.
"""

import os
import json
import time
import glob
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def tokenize_file(
    input_path: str,
    tokenizer_path: str,
    text_field: str = "text",
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    """Tokenize a single JSONL file into a list of token IDs.

    Args:
        input_path: Path to JSONL file.
        tokenizer_path: Path to SentencePiece model.
        text_field: JSON field containing text.
        add_bos: Add beginning-of-sequence token.
        add_eos: Add end-of-sequence token.

    Returns:
        List of token IDs.
    """
    # Import here for multiprocessing compatibility
    from src.tokenizer.tokenizer import YayaTokenizer

    tokenizer = YayaTokenizer(tokenizer_path)
    all_tokens = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get(text_field, "")
                if not text or len(text) < 10:
                    continue
                tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
                all_tokens.extend(tokens)
            except (json.JSONDecodeError, KeyError):
                continue

    return all_tokens


def write_shard(tokens: List[int], output_path: str, dtype=np.uint16):
    """Write a list of tokens to a binary shard file.

    Args:
        tokens: List of token IDs.
        output_path: Output .bin file path.
        dtype: NumPy data type for storage.
    """
    arr = np.array(tokens, dtype=dtype)
    arr.tofile(output_path)
    return len(tokens)


class CorpusTokenizer:
    """Tokenize a large corpus into memory-mapped training shards.

    Reads JSONL files from the processed data directory,
    tokenizes them, and writes fixed-size binary shards.
    """

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        shard_size: int = 100_000_000,  # 100M tokens per shard
        dtype: str = "uint16",
        num_workers: int = 4,
    ):
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.dtype = getattr(np, dtype)
        self.num_workers = num_workers

        os.makedirs(output_dir, exist_ok=True)

    def tokenize_directory(
        self,
        input_dir: str,
        file_pattern: str = "*.jsonl",
        text_field: str = "text",
        split_name: str = "train",
        max_files: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Tokenize all files in a directory into shards.

        Args:
            input_dir: Directory containing JSONL files.
            file_pattern: Glob pattern for input files.
            text_field: JSON field containing text.
            split_name: Name prefix for output shards.
            max_files: Limit number of input files.

        Returns:
            Dict with tokenization statistics.
        """
        input_files = sorted(glob.glob(os.path.join(input_dir, "**", file_pattern), recursive=True))
        if max_files:
            input_files = input_files[:max_files]

        if not input_files:
            print(f"No files matching {file_pattern} in {input_dir}")
            return {"total_tokens": 0, "shards": 0}

        print(f"Tokenizing {len(input_files)} files from {input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Shard size: {self.shard_size:,} tokens")
        print(f"  Workers: {self.num_workers}")

        start_time = time.time()
        token_buffer = []
        shard_idx = 0
        shard_paths = []
        total_tokens = 0

        # Process files (sequentially for now — tokenizer isn't easily picklable)
        for file_idx, filepath in enumerate(input_files):
            rel_path = os.path.relpath(filepath, input_dir)
            print(f"  [{file_idx + 1}/{len(input_files)}] {rel_path}", end="", flush=True)

            tokens = tokenize_file(
                filepath, self.tokenizer_path, text_field=text_field
            )
            token_buffer.extend(tokens)
            print(f" -> {len(tokens):,} tokens")

            # Write full shards
            while len(token_buffer) >= self.shard_size:
                shard_tokens = token_buffer[:self.shard_size]
                token_buffer = token_buffer[self.shard_size:]

                shard_path = os.path.join(
                    self.output_dir, f"{split_name}_shard_{shard_idx:05d}.bin"
                )
                write_shard(shard_tokens, shard_path, self.dtype)
                shard_paths.append(shard_path)
                total_tokens += len(shard_tokens)
                print(f"    Shard {shard_idx}: {len(shard_tokens):,} tokens -> {shard_path}")
                shard_idx += 1

        # Write remainder
        if token_buffer:
            shard_path = os.path.join(
                self.output_dir, f"{split_name}_shard_{shard_idx:05d}.bin"
            )
            write_shard(token_buffer, shard_path, self.dtype)
            shard_paths.append(shard_path)
            total_tokens += len(token_buffer)
            print(f"    Final shard {shard_idx}: {len(token_buffer):,} tokens -> {shard_path}")

        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / max(elapsed, 1)

        stats = {
            "total_tokens": total_tokens,
            "shards": len(shard_paths),
            "shard_paths": shard_paths,
            "input_files": len(input_files),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
        }

        # Save stats
        stats_path = os.path.join(self.output_dir, f"{split_name}_stats.json")
        with open(stats_path, "w") as f:
            serializable = {k: v for k, v in stats.items() if k != "shard_paths"}
            serializable["shard_paths"] = [os.path.basename(p) for p in shard_paths]
            json.dump(serializable, f, indent=2)

        print(f"\nTokenization complete:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Shards: {len(shard_paths)}")
        print(f"  Time: {elapsed:.1f}s ({tokens_per_sec:,.0f} tokens/sec)")

        return stats

    def create_eval_split(
        self,
        input_dir: str,
        eval_ratio: float = 0.02,
        max_eval_tokens: int = 10_000_000,
        file_pattern: str = "*.jsonl",
        text_field: str = "text",
    ) -> Dict[str, Any]:
        """Create a held-out evaluation split.

        Takes a small portion of data for evaluation.

        Args:
            input_dir: Directory with JSONL files.
            eval_ratio: Fraction of data for eval.
            max_eval_tokens: Hard cap on eval tokens.
            file_pattern: Input file glob.
            text_field: JSON field for text.

        Returns:
            Stats dict.
        """
        input_files = sorted(glob.glob(os.path.join(input_dir, "**", file_pattern), recursive=True))
        if not input_files:
            return {"total_tokens": 0}

        # Use last few files for eval
        num_eval_files = max(1, int(len(input_files) * eval_ratio))
        eval_files = input_files[-num_eval_files:]

        print(f"Creating eval split from {len(eval_files)} files")

        eval_tokens = []
        for filepath in eval_files:
            tokens = tokenize_file(
                filepath, self.tokenizer_path, text_field=text_field
            )
            eval_tokens.extend(tokens)
            if len(eval_tokens) >= max_eval_tokens:
                eval_tokens = eval_tokens[:max_eval_tokens]
                break

        eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        eval_path = os.path.join(eval_dir, "eval_shard_00000.bin")
        write_shard(eval_tokens, eval_path, self.dtype)

        print(f"  Eval tokens: {len(eval_tokens):,} -> {eval_path}")
        return {"total_tokens": len(eval_tokens), "path": eval_path}
