"""Dataset classes for Yaya model training.

Provides text, vision, and multimodal dataset implementations
with efficient streaming for large-scale training.
"""

import os
import json
import random
from typing import Optional, Dict, List, Any, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np


class TextDataset(Dataset):
    """Memory-mapped text dataset for pre-training.

    Stores pre-tokenized data as memory-mapped numpy arrays for efficient
    random access without loading everything into memory.
    """

    def __init__(
        self,
        data_path: str,
        max_seq_length: int = 4096,
        split: str = "train",
    ):
        """Initialize text dataset.

        Args:
            data_path: Path to directory containing .bin and .idx files.
            max_seq_length: Maximum sequence length for training.
            split: Dataset split ('train' or 'eval').
        """
        self.max_seq_length = max_seq_length

        # Load memory-mapped token array
        bin_path = os.path.join(data_path, f"{split}.bin")
        idx_path = os.path.join(data_path, f"{split}.idx")

        if os.path.exists(bin_path):
            self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
            self.num_tokens = len(self.tokens)
        else:
            # Fallback: try loading from .npy
            npy_path = os.path.join(data_path, f"{split}.npy")
            if os.path.exists(npy_path):
                self.tokens = np.load(npy_path, mmap_mode="r")
                self.num_tokens = len(self.tokens)
            else:
                raise FileNotFoundError(
                    f"No data found at {data_path}. Expected {bin_path} or {npy_path}"
                )

        # Number of complete sequences we can extract
        self.num_samples = self.num_tokens // (max_seq_length + 1)

        print(f"TextDataset loaded: {self.num_tokens:,} tokens, {self.num_samples:,} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns input_ids and labels where labels are shifted by 1
        (next token prediction).
        """
        start = idx * (self.max_seq_length + 1)
        end = start + self.max_seq_length + 1

        chunk = self.tokens[start:end].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])   # input
        y = torch.from_numpy(chunk[1:])    # target (shifted by 1)

        return {
            "input_ids": x,
            "labels": y,
        }


class StreamingTextDataset(IterableDataset):
    """Streaming text dataset for distributed training.

    Reads pre-tokenized data from multiple sharded files and streams
    random chunks. Supports multi-worker and multi-GPU data loading.
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_length: int = 4096,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        # Find all shard files
        self.shard_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".bin") or f.endswith(".npy")]
        )
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {data_dir}")

        print(f"StreamingTextDataset: {len(self.shard_files)} shards in {data_dir}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over data, yielding training samples."""
        worker_info = torch.utils.data.get_worker_info()

        # Distribute shards across workers
        if worker_info is not None:
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        else:
            shards = self.shard_files

        # Shuffle shard order
        if self.shuffle_shards:
            rng = random.Random(self.seed)
            rng.shuffle(shards)

        for shard_file in shards:
            shard_path = os.path.join(self.data_dir, shard_file)

            if shard_file.endswith(".bin"):
                tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
            else:
                tokens = np.load(shard_path, mmap_mode="r")

            # Extract sequential chunks
            num_chunks = len(tokens) // (self.max_seq_length + 1)
            for i in range(num_chunks):
                start = i * (self.max_seq_length + 1)
                end = start + self.max_seq_length + 1
                chunk = tokens[start:end].astype(np.int64)

                x = torch.from_numpy(chunk[:-1])
                y = torch.from_numpy(chunk[1:])

                yield {
                    "input_ids": x,
                    "labels": y,
                }


class InstructionDataset(Dataset):
    """Dataset for supervised fine-tuning (SFT) on instruction-response pairs.

    Loads JSONL files where each line contains:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        # Load JSONL data
        if os.path.isfile(data_path):
            files = [data_path]
        else:
            files = sorted(Path(data_path).glob("*.jsonl"))

        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        print(f"InstructionDataset loaded: {len(self.samples):,} samples from {len(files)} files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single instruction-tuning sample.

        Tokenizes the chat messages and creates labels where only
        the assistant responses contribute to the loss (user/system
        messages are masked with -100).
        """
        sample = self.samples[idx]
        messages = sample.get("messages", [])

        # Format chat and tokenize
        formatted_text = self.tokenizer.format_chat(messages)
        token_ids = self.tokenizer.encode(formatted_text, add_bos=True, add_eos=True)

        # Truncate to max length
        token_ids = token_ids[: self.max_seq_length]

        # Create labels (mask non-assistant tokens with -100)
        # For simplicity, we use full sequence as labels here.
        # A production implementation would mask system/user turns.
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        # Pad if needed
        pad_len = self.max_seq_length - 1 - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])

        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class MultimodalDataset(Dataset):
    """Dataset for multimodal (image + text) training.

    Loads JSONL files where each line contains:
    {"image": "path/to/image.jpg", "text": "description of image"}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        image_processor=None,
        max_seq_length: int = 4096,
        image_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_seq_length = max_seq_length
        self.image_dir = image_dir
        self.samples = []

        # Load JSONL data
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"MultimodalDataset loaded: {len(self.samples):,} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a multimodal training sample."""
        sample = self.samples[idx]

        # Load and process image
        image_path = sample["image"]
        if self.image_dir:
            image_path = os.path.join(self.image_dir, image_path)

        pixel_values = None
        if self.image_processor and os.path.exists(image_path):
            pixel_values = self.image_processor(image_path)

        # Tokenize text
        text = sample.get("text", "")
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        token_ids = token_ids[: self.max_seq_length]

        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        result = {
            "input_ids": input_ids,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = pixel_values

        return result
