"""Tests for the data engine components: quality pipeline, downloader, tokenization."""

import os
import json
import tempfile
import shutil

import pytest

from src.data.quality import (
    LanguageDetector,
    MinHasher,
    LSHIndex,
    NearDeduplicator,
    QualityScorer,
    PIIRemover,
    DataQualityPipeline,
)
from src.data.downloader import DownloadManager, DatasetInfo


# ── Language Detection ─────────────────────────────────────────


class TestLanguageDetector:
    def setup_method(self):
        self.detector = LanguageDetector(target_language="en")

    def test_detect_english(self):
        text = "The quick brown fox jumps over the lazy dog and that is a sentence."
        assert self.detector.detect(text) == "en"

    def test_detect_french(self):
        text = "Le chat est sur la table et les enfants sont dans le jardin."
        assert self.detector.detect(text) == "fr"

    def test_detect_german(self):
        text = "Der Hund ist in dem Garten und die Katze ist auf dem Tisch."
        assert self.detector.detect(text) == "de"

    def test_is_target_language_true(self):
        text = "This is a simple English sentence with common words for the test."
        assert self.detector.is_target_language(text) is True

    def test_is_target_language_false(self):
        text = "Le chat est sur la table et les enfants sont dans le jardin."
        assert self.detector.is_target_language(text) is False

    def test_short_text_unknown(self):
        text = "xyz"
        assert self.detector.detect(text) == "unknown"


# ── MinHash ────────────────────────────────────────────────────


class TestMinHasher:
    def setup_method(self):
        self.hasher = MinHasher(num_perm=64, ngram_size=5)

    def test_signature_length(self):
        sig = self.hasher.compute_signature("Hello world, this is a test document.")
        assert len(sig) == 64

    def test_identical_docs_high_similarity(self):
        text = "The quick brown fox jumps over the lazy dog."
        sig1 = self.hasher.compute_signature(text)
        sig2 = self.hasher.compute_signature(text)
        sim = MinHasher.estimate_similarity(sig1, sig2)
        assert sim == 1.0

    def test_similar_docs(self):
        text1 = "The quick brown fox jumps over the lazy dog near the river."
        text2 = "The quick brown fox jumps over the lazy dog near the lake."
        sig1 = self.hasher.compute_signature(text1)
        sig2 = self.hasher.compute_signature(text2)
        sim = MinHasher.estimate_similarity(sig1, sig2)
        assert sim > 0.5  # Similar but not identical

    def test_different_docs_low_similarity(self):
        text1 = "Machine learning is a subset of artificial intelligence."
        text2 = "Photosynthesis converts sunlight into chemical energy in plants."
        sig1 = self.hasher.compute_signature(text1)
        sig2 = self.hasher.compute_signature(text2)
        sim = MinHasher.estimate_similarity(sig1, sig2)
        assert sim < 0.5

    def test_empty_text(self):
        sig = self.hasher.compute_signature("")
        assert len(sig) == 64


# ── LSH Index ──────────────────────────────────────────────────


class TestLSHIndex:
    def test_insert_unique(self):
        lsh = LSHIndex(num_perm=64, threshold=0.8)
        hasher = MinHasher(num_perm=64)
        sig = hasher.compute_signature("A unique document about science.")
        is_dup = lsh.insert("doc1", sig)
        # First insert is never a dup
        assert is_dup is False

    def test_insert_duplicate(self):
        lsh = LSHIndex(num_perm=64, threshold=0.8)
        hasher = MinHasher(num_perm=64)
        text = "The exact same document about machine learning and AI."
        sig = hasher.compute_signature(text)
        lsh.insert("doc1", sig)
        is_dup = lsh.insert("doc2", sig)
        assert is_dup is True


# ── Near Deduplicator ─────────────────────────────────────────


class TestNearDeduplicator:
    def test_exact_duplicate(self):
        dedup = NearDeduplicator(num_perm=64, threshold=0.8)
        text = "This is a test document for deduplication testing purposes."
        assert dedup.is_duplicate(text) is False
        assert dedup.is_duplicate(text) is True

    def test_unique_documents(self):
        dedup = NearDeduplicator(num_perm=64, threshold=0.8)
        assert dedup.is_duplicate("Machine learning uses data to learn patterns.") is False
        assert dedup.is_duplicate("Photosynthesis is the process plants use for energy.") is False

    def test_stats(self):
        dedup = NearDeduplicator(num_perm=64, threshold=0.8)
        dedup.is_duplicate("Document one about topic A with enough words to hash.")
        dedup.is_duplicate("Document two about topic B with enough words to hash.")
        dedup.is_duplicate("Document one about topic A with enough words to hash.")
        stats = dedup.stats
        assert stats["total_docs"] == 3
        assert stats["duplicates"] == 1


# ── Quality Scorer ─────────────────────────────────────────────


class TestQualityScorer:
    def setup_method(self):
        self.scorer = QualityScorer()

    def test_high_quality_text(self):
        text = (
            "Machine learning is a method of data analysis that automates analytical "
            "model building. It is a branch of artificial intelligence based on the idea "
            "that systems can learn from data, identify patterns and make decisions with "
            "minimal human intervention. The process begins with observations or data."
        )
        score = self.scorer.score(text)
        assert score > 0.5

    def test_low_quality_short(self):
        score = self.scorer.score("hi")
        assert score < 0.3

    def test_empty_text(self):
        assert self.scorer.score("") == 0.0

    def test_boilerplate_penalty(self):
        clean = "This is a well-written article about science and technology."
        boilerplate = (
            "Click here to subscribe to our newsletter. "
            "Cookie policy and terms of service apply. "
            "Share on Facebook and Twitter. All rights reserved."
        )
        score_clean = self.scorer.score(clean)
        score_boilerplate = self.scorer.score(boilerplate)
        assert score_clean > score_boilerplate


# ── PII Remover ────────────────────────────────────────────────


class TestPIIRemover:
    def setup_method(self):
        self.remover = PIIRemover()

    def test_remove_email(self):
        text = "Contact us at john.doe@example.com for details."
        cleaned = self.remover.remove(text)
        assert "[EMAIL]" in cleaned
        assert "john.doe@example.com" not in cleaned

    def test_remove_phone(self):
        text = "Call us at (555) 123-4567 today."
        cleaned = self.remover.remove(text)
        assert "[PHONE]" in cleaned

    def test_remove_ip(self):
        text = "Server at 192.168.1.100 is down."
        cleaned = self.remover.remove(text)
        assert "[IP_ADDRESS]" in cleaned
        assert "192.168.1.100" not in cleaned

    def test_no_pii_unchanged(self):
        text = "The weather is nice today."
        cleaned = self.remover.remove(text)
        assert cleaned == text

    def test_stats_tracking(self):
        self.remover.remove("Email me at a@b.com or c@d.org")
        stats = self.remover.stats
        assert stats["email"] == 2


# ── Full Pipeline ──────────────────────────────────────────────


class TestDataQualityPipeline:
    def setup_method(self):
        self.pipeline = DataQualityPipeline(
            language="en",
            quality_threshold=0.3,
            dedup_threshold=0.8,
            min_doc_length=50,
            remove_pii=True,
        )

    def test_passes_good_english(self):
        text = (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data without being explicitly programmed. Deep learning "
            "uses neural networks with many layers to learn hierarchical representations."
        )
        result = self.pipeline.process_document(text)
        assert result is not None

    def test_rejects_non_english(self):
        text = (
            "Le chat est sur la table et les enfants sont dans le jardin. "
            "Les fleurs sont belles dans le parc en été."
        )
        result = self.pipeline.process_document(text)
        assert result is None

    def test_rejects_duplicate(self):
        text = (
            "This is a test document that should only be included once in the dataset. "
            "It contains enough words to generate meaningful hash signatures."
        )
        result1 = self.pipeline.process_document(text)
        result2 = self.pipeline.process_document(text)
        assert result1 is not None
        assert result2 is None

    def test_rejects_too_short(self):
        result = self.pipeline.process_document("Short text.")
        assert result is None

    def test_removes_pii(self):
        text = (
            "Please contact us at support@example.com for more information about "
            "our machine learning platform. Our server is at 10.0.0.1 for access."
        )
        result = self.pipeline.process_document(text)
        if result:
            assert "support@example.com" not in result
            assert "[EMAIL]" in result

    def test_stats(self):
        self.pipeline.process_document(
            "A good English document about science and technology with enough words."
        )
        self.pipeline.process_document("Trop court.")
        stats = self.pipeline.get_stats()
        assert stats["total_input"] == 2
        assert "dedup_stats" in stats


# ── Download Manager ───────────────────────────────────────────


class TestDownloadManager:
    def test_manifest_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DownloadManager(output_dir=tmpdir, cache_dir=tmpdir)
            assert manager.manifest == {"datasets": {}, "last_updated": None}

    def test_manifest_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DownloadManager(output_dir=tmpdir, cache_dir=tmpdir)
            manager.manifest["datasets"]["test"] = {"status": "complete"}
            manager._save_manifest()

            manager2 = DownloadManager(output_dir=tmpdir, cache_dir=tmpdir)
            assert manager2.manifest["datasets"]["test"]["status"] == "complete"

    def test_is_downloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DownloadManager(output_dir=tmpdir, cache_dir=tmpdir)
            assert manager.is_downloaded("test") is False
            manager.manifest["datasets"]["test"] = {"status": "complete"}
            assert manager.is_downloaded("test") is True

    def test_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DownloadManager(output_dir=tmpdir, cache_dir=tmpdir)
            manager.manifest["datasets"]["a"] = {"status": "complete"}
            manager.manifest["datasets"]["b"] = {"status": "failed"}
            status = manager.status()
            assert "a" in status["complete"]
            assert "b" in status["failed"]

    def test_load_sources_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "datasets": {
                    "test_ds": {
                        "name": "test/dataset",
                        "source": "huggingface",
                        "category": "web_text",
                        "format": "jsonl",
                        "text_field": "text",
                    }
                }
            }
            config_path = os.path.join(tmpdir, "sources.yaml")
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            datasets = DownloadManager.load_sources_config(config_path)
            assert "test_ds" in datasets
            assert datasets["test_ds"].name == "test/dataset"
            assert datasets["test_ds"].category == "web_text"
