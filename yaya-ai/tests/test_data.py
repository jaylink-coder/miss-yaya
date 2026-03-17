"""Tests for data pipeline components."""

import pytest
import torch
import tempfile
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import TextDataset
from src.data.dataloader import collate_fn
from src.data.processing import TextCleaner, TextFilter, Deduplicator
from src.data.mixing import CurriculumScheduler


class TestTextCleaner:
    def test_basic_cleaning(self):
        cleaner = TextCleaner()
        text = "  Hello   world  \x00\x01  "
        result = cleaner.clean(text)
        assert "\x00" not in result
        assert "Hello" in result

    def test_url_removal(self):
        cleaner = TextCleaner(remove_urls=True)
        text = "Visit https://example.com for more info"
        result = cleaner.clean(text)
        assert "https://example.com" not in result

    def test_email_removal(self):
        cleaner = TextCleaner(remove_emails=True)
        text = "Contact us at test@example.com"
        result = cleaner.clean(text)
        assert "test@example.com" not in result

    def test_empty_input(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""


class TestTextFilter:
    def test_valid_document(self):
        filt = TextFilter(min_doc_length=10, max_doc_length=10000)
        text = "This is a perfectly valid document with enough alphabetic content."
        assert filt.is_valid(text) is True

    def test_too_short(self):
        filt = TextFilter(min_doc_length=100)
        assert filt.is_valid("short") is False

    def test_too_many_special_chars(self):
        filt = TextFilter(max_special_char_ratio=0.1)
        text = "!!@@##$$%%^^&&**(())!!" * 10
        assert filt.is_valid(text) is False

    def test_low_alpha_ratio(self):
        filt = TextFilter(min_alpha_ratio=0.5)
        text = "12345 67890 12345 67890 12345 67890 12345 67890 12345 67890" * 5
        assert filt.is_valid(text) is False


class TestDeduplicator:
    def test_exact_dedup(self):
        dedup = Deduplicator()
        assert dedup.is_duplicate("hello world") is False
        assert dedup.is_duplicate("hello world") is True
        assert dedup.is_duplicate("different text") is False

    def test_deduplicate_list(self):
        dedup = Deduplicator()
        texts = ["aaa", "bbb", "aaa", "ccc", "bbb"]
        result = dedup.deduplicate(texts)
        assert len(result) == 3

    def test_reset(self):
        dedup = Deduplicator()
        dedup.is_duplicate("test")
        assert dedup.num_seen == 1
        dedup.reset()
        assert dedup.num_seen == 0


class TestCollation:
    def test_collate_same_length(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([5, 6, 7]), "labels": torch.tensor([6, 7, 8])},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_collate_different_lengths(self):
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4]), "labels": torch.tensor([2, 3, 4, 5])},
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([2, 3])},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 4)  # Padded to max length
        assert result["attention_mask"][1, 2].item() == 0  # Padding position


class TestCurriculumScheduler:
    def test_initial_ratios(self):
        scheduler = CurriculumScheduler(
            initial_ratios={"a": 0.7, "b": 0.3},
            final_ratios={"a": 0.3, "b": 0.7},
            total_steps=1000,
        )
        ratios = scheduler.get_ratios(0)
        assert abs(ratios["a"] - 0.7) < 0.01

    def test_final_ratios(self):
        scheduler = CurriculumScheduler(
            initial_ratios={"a": 0.7, "b": 0.3},
            final_ratios={"a": 0.3, "b": 0.7},
            total_steps=1000,
        )
        ratios = scheduler.get_ratios(1000)
        assert abs(ratios["a"] - 0.3) < 0.01

    def test_midpoint(self):
        scheduler = CurriculumScheduler(
            initial_ratios={"a": 1.0, "b": 0.0},
            final_ratios={"a": 0.0, "b": 1.0},
            total_steps=1000,
        )
        ratios = scheduler.get_ratios(500)
        assert abs(ratios["a"] - 0.5) < 0.01


class TestTextDataset:
    def test_memmap_dataset(self):
        """Test dataset with a temporary binary file."""
        tmpdir = tempfile.mkdtemp()
        try:
            # Create fake tokenized data
            tokens = np.arange(1000, dtype=np.uint16)
            bin_path = os.path.join(tmpdir, "train.bin")
            tokens.tofile(bin_path)

            dataset = TextDataset(data_path=tmpdir, max_seq_length=32, split="train")
            assert len(dataset) > 0

            sample = dataset[0]
            assert "input_ids" in sample
            assert "labels" in sample
            assert sample["input_ids"].shape[0] == 32
            assert sample["labels"].shape[0] == 32

            # Release memmap before cleanup (Windows holds file handles)
            del dataset
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
