"""Tokenizer Training — Train a BPE tokenizer using SentencePiece.

Trains a byte-pair encoding tokenizer on a text corpus, with reserved
slots for special tokens (multimodal, tool use, chat formatting).
"""

import os
import glob
from typing import List, Optional
from pathlib import Path

import sentencepiece as spm

from src.tokenizer.tokenizer import SPECIAL_TOKENS_LIST


class TokenizerTrainer:
    """Train a SentencePiece BPE tokenizer for the Yaya model."""

    def __init__(
        self,
        vocab_size: int = 64000,
        model_prefix: str = "yaya_tokenizer",
        output_dir: str = "data/tokenizer",
        character_coverage: float = 0.9995,
    ):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.output_dir = output_dir
        self.character_coverage = character_coverage

    def train(
        self,
        input_files: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        file_pattern: str = "*.txt",
        num_threads: int = 4,
        max_sentence_length: int = 16384,
    ) -> str:
        """Train the tokenizer on a text corpus.

        Args:
            input_files: List of input text file paths.
            input_dir: Directory containing text files (alternative to input_files).
            file_pattern: Glob pattern for files in input_dir.
            num_threads: Number of training threads.
            max_sentence_length: Maximum sentence length in bytes.

        Returns:
            Path to the trained model file.
        """
        # Resolve input files
        if input_files is None and input_dir is not None:
            input_files = sorted(glob.glob(os.path.join(input_dir, file_pattern)))
        
        if not input_files:
            raise ValueError("No input files provided. Specify input_files or input_dir.")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        model_prefix_path = os.path.join(self.output_dir, self.model_prefix)

        # Build the user_defined_symbols string for special tokens
        # Exclude built-in SPM tokens (pad, unk, bos, eos) — they're handled by pad_id/unk_id/bos_id/eos_id
        builtin_tokens = {"<pad>", "<unk>", "<s>", "<" + chr(47) + "s>"}
        extra_tokens = [t for t in SPECIAL_TOKENS_LIST if t not in builtin_tokens]
        user_defined_symbols = ",".join(extra_tokens)

        print(f"Training tokenizer with vocab_size={self.vocab_size}")
        print(f"  Input files: {len(input_files)}")
        print(f"  Output: {model_prefix_path}.model")
        print(f"  Special tokens: {len(SPECIAL_TOKENS_LIST)}")

        # Train SentencePiece model
        spm.SentencePieceTrainer.Train(
            input=",".join(input_files),
            model_prefix=model_prefix_path,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=self.character_coverage,
            num_threads=num_threads,
            max_sentence_length=max_sentence_length,
            # Special tokens
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=user_defined_symbols,
            # Training settings
            byte_fallback=True,          # Handle unknown bytes
            split_digits=True,           # Treat digits as separate tokens
            allow_whitespace_only_pieces=True,
            normalization_rule_name="identity",  # No normalization
            remove_extra_whitespaces=False,
            # Input format
            input_format="text",
        )

        model_path = f"{model_prefix_path}.model"
        print(f"Tokenizer trained successfully: {model_path}")
        return model_path

    def train_from_iterator(
        self,
        text_iterator,
        temp_file: str = "data/tokenizer/_temp_training_data.txt",
    ) -> str:
        """Train from a Python iterator of strings.

        Writes iterator contents to a temp file, then trains.
        Useful for training from HuggingFace datasets or generators.

        Args:
            text_iterator: Iterator yielding text strings.
            temp_file: Temporary file path for writing data.

        Returns:
            Path to trained model file.
        """
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        
        print("Writing training data to temp file...")
        count = 0
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in text_iterator:
                if text.strip():
                    f.write(text.strip() + "\n")
                    count += 1
                    if count % 100000 == 0:
                        print(f"  Wrote {count:,} lines")

        print(f"  Total: {count:,} lines")
        
        result = self.train(input_files=[temp_file])
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return result
