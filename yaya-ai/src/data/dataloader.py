"""Data loading utilities with distributed training support.

Provides collation, distributed sampling, and efficient data loading
for both pre-training and fine-tuning phases.
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset, IterableDataset
from typing import Optional, Dict, List, Any


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate a batch of samples into padded tensors.

    Handles variable-length sequences by padding to the longest
    sequence in the batch.

    Args:
        batch: List of sample dicts with 'input_ids', 'labels', etc.

    Returns:
        Batched dict with padded tensors.
    """
    # Find max length in this batch
    max_len = max(sample["input_ids"].shape[0] for sample in batch)

    input_ids = []
    labels = []
    attention_masks = []
    pixel_values_list = []
    has_images = False

    for sample in batch:
        seq_len = sample["input_ids"].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids with 0 (pad token)
        if pad_len > 0:
            input_ids.append(
                torch.cat([sample["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
            )
        else:
            input_ids.append(sample["input_ids"])

        # Pad labels with -100 (ignored in loss)
        if "labels" in sample:
            if pad_len > 0:
                labels.append(
                    torch.cat([sample["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                )
            else:
                labels.append(sample["labels"])

        # Build attention mask
        if "attention_mask" in sample:
            if pad_len > 0:
                attention_masks.append(
                    torch.cat([sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
            else:
                attention_masks.append(sample["attention_mask"])
        else:
            mask = torch.ones(seq_len, dtype=torch.long)
            if pad_len > 0:
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])
            attention_masks.append(mask)

        # Collect images if present
        if "pixel_values" in sample:
            has_images = True
            pixel_values_list.append(sample["pixel_values"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
    }

    if labels:
        result["labels"] = torch.stack(labels)

    if has_images and pixel_values_list:
        result["pixel_values"] = torch.stack(pixel_values_list)

    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    drop_last: bool = True,
    pin_memory: bool = True,
    sampler=None,
) -> DataLoader:
    """Create a DataLoader with optional distributed sampling.

    Args:
        dataset: The dataset to load from.
        batch_size: Per-device batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle data.
        distributed: Whether to use DistributedSampler.
        rank: Current process rank (for distributed).
        world_size: Total number of processes (for distributed).
        seed: Random seed for shuffling.
        drop_last: Drop the last incomplete batch.
        pin_memory: Pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader.
    """
    is_iterable = isinstance(dataset, IterableDataset)

    # Caller-provided sampler takes precedence over distributed sampler
    if sampler is None and distributed and not is_iterable:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if not is_iterable and sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
