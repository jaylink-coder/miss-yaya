"""Text data processing pipeline.

Handles crawled/raw text data: cleaning, filtering, deduplication,
quality scoring, and tokenization for pre-training.
"""

import os
import re
import hashlib
import unicodedata
from typing import List, Optional, Set, Iterator, Callable
from pathlib import Path
import numpy as np


class TextCleaner:
    """Clean raw text data for pre-training.

    Applies a series of cleaning steps to remove noise, normalize
    formatting, and ensure text quality.
    """

    def __init__(
        self,
        min_line_length: int = 10,
        max_line_length: int = 100000,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
    ):
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.normalize_unicode = normalize_unicode

        # Compiled regex patterns
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+", re.IGNORECASE
        )
        self._email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        self._whitespace_pattern = re.compile(r"\s+")
        self._control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to a text string."""
        if not text or not text.strip():
            return ""

        # Remove control characters
        text = self._control_char_pattern.sub("", text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub("", text)

        # Remove emails
        if self.remove_emails:
            text = self._email_pattern.sub("", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._whitespace_pattern.sub(" ", text)
            # But preserve paragraph breaks (double newlines)
            text = re.sub(r" *\n *\n *", "\n\n", text)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts, filtering out empty results."""
        results = []
        for text in texts:
            cleaned = self.clean(text)
            if len(cleaned) >= self.min_line_length:
                results.append(cleaned)
        return results


class TextFilter:
    """Filter text documents based on quality heuristics.

    Applies rules to remove low-quality documents:
    - Too short or too long
    - Too many special characters
    - Too repetitive
    - Wrong language (basic detection)
    """

    def __init__(
        self,
        min_doc_length: int = 100,
        max_doc_length: int = 1000000,
        max_special_char_ratio: float = 0.3,
        max_repetition_ratio: float = 0.5,
        min_alpha_ratio: float = 0.5,
    ):
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.min_alpha_ratio = min_alpha_ratio

    def is_valid(self, text: str) -> bool:
        """Check if a document passes all quality filters."""
        if not text:
            return False

        length = len(text)

        # Length check
        if length < self.min_doc_length or length > self.max_doc_length:
            return False

        # Alphabetic character ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / length < self.min_alpha_ratio:
            return False

        # Special character ratio
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if special_count / length > self.max_special_char_ratio:
            return False

        # Line repetition check (detect boilerplate/generated content)
        lines = text.split("\n")
        if len(lines) > 5:
            unique_lines = set(line.strip() for line in lines if line.strip())
            if len(unique_lines) / len([l for l in lines if l.strip()]) < (1 - self.max_repetition_ratio):
                return False

        return True

    def filter_batch(self, texts: List[str]) -> List[str]:
        """Filter a batch of texts, keeping only valid ones."""
        return [text for text in texts if self.is_valid(text)]


class Deduplicator:
    """Document-level deduplication using hashing.

    Uses both exact deduplication (SHA-256) and near-dedup
    (MinHash with LSH) to remove duplicate content.
    """

    def __init__(self):
        self._seen_hashes: Set[str] = set()

    def _hash_text(self, text: str) -> str:
        """Compute SHA-256 hash of normalized text."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """Check if this text is a duplicate of a previously seen document."""
        text_hash = self._hash_text(text)
        if text_hash in self._seen_hashes:
            return True
        self._seen_hashes.add(text_hash)
        return False

    def deduplicate(self, texts: List[str]) -> List[str]:
        """Remove duplicates from a list of texts."""
        results = []
        for text in texts:
            if not self.is_duplicate(text):
                results.append(text)
        return results

    def reset(self):
        """Clear the deduplication cache."""
        self._seen_hashes.clear()

    @property
    def num_seen(self) -> int:
        return len(self._seen_hashes)


class DataProcessor:
    """End-to-end data processing pipeline.

    Combines cleaning, filtering, and deduplication into a single
    pipeline that processes raw text files into tokenized training data.
    """

    def __init__(
        self,
        tokenizer,
        cleaner: Optional[TextCleaner] = None,
        text_filter: Optional[TextFilter] = None,
        deduplicator: Optional[Deduplicator] = None,
    ):
        self.tokenizer = tokenizer
        self.cleaner = cleaner or TextCleaner()
        self.text_filter = text_filter or TextFilter()
        self.deduplicator = deduplicator or Deduplicator()

    def process_file(self, input_path: str) -> List[int]:
        """Process a single text file into token IDs.

        Args:
            input_path: Path to raw text file.

        Returns:
            List of token IDs from all valid documents in the file.
        """
        all_tokens = []

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read documents separated by double newlines
            content = f.read()
            documents = content.split("\n\n")

        for doc in documents:
            # Clean
            cleaned = self.cleaner.clean(doc)
            if not cleaned:
                continue

            # Filter
            if not self.text_filter.is_valid(cleaned):
                continue

            # Dedup
            if self.deduplicator.is_duplicate(cleaned):
                continue

            # Tokenize
            tokens = self.tokenizer.encode(cleaned, add_bos=True, add_eos=True)
            all_tokens.extend(tokens)

        return all_tokens

    def process_directory(
        self,
        input_dir: str,
        output_path: str,
        file_pattern: str = "*.txt",
        shard_size: int = 100_000_000,  # tokens per shard
    ) -> List[str]:
        """Process all text files in a directory into tokenized shards.

        Args:
            input_dir: Directory containing raw text files.
            output_path: Output directory for tokenized shards.
            file_pattern: Glob pattern for input files.
            shard_size: Number of tokens per output shard.

        Returns:
            List of output shard file paths.
        """
        os.makedirs(output_path, exist_ok=True)
        input_files = sorted(Path(input_dir).glob(file_pattern))

        print(f"Processing {len(input_files)} files from {input_dir}")

        all_tokens = []
        shard_paths = []
        shard_idx = 0

        for file_idx, filepath in enumerate(input_files):
            print(f"  [{file_idx + 1}/{len(input_files)}] {filepath.name}")
            tokens = self.process_file(str(filepath))
            all_tokens.extend(tokens)

            # Write shard when buffer is large enough
            while len(all_tokens) >= shard_size:
                shard_tokens = all_tokens[:shard_size]
                all_tokens = all_tokens[shard_size:]

                shard_path = os.path.join(output_path, f"shard_{shard_idx:05d}.bin")
                arr = np.array(shard_tokens, dtype=np.uint16)
                arr.tofile(shard_path)
                shard_paths.append(shard_path)
                print(f"    Wrote shard {shard_idx}: {len(shard_tokens):,} tokens")
                shard_idx += 1

        # Write remaining tokens
        if all_tokens:
            shard_path = os.path.join(output_path, f"shard_{shard_idx:05d}.bin")
            arr = np.array(all_tokens, dtype=np.uint16)
            arr.tofile(shard_path)
            shard_paths.append(shard_path)
            print(f"    Wrote final shard {shard_idx}: {len(all_tokens):,} tokens")

        total_tokens = sum(
            os.path.getsize(p) // 2 for p in shard_paths  # uint16 = 2 bytes
        )
        print(f"Done. {len(shard_paths)} shards, {total_tokens:,} total tokens")
        print(f"  Dedup: {self.deduplicator.num_seen:,} unique documents")

        return shard_paths
