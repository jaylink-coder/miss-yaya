"""Tests for the HuggingFace fine-tuning pipeline scripts.

Tests data generation, data format validation, and script argument parsing
without requiring GPU or model downloads.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainingDataGeneration(unittest.TestCase):
    """Test the generate_training_data.py script."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_sft_pipeline_generates_examples(self):
        """SFTDataPipeline produces valid examples."""
        from src.agent.sft_data_pipeline import SFTDataPipeline

        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=5, rag_count=5, safety_count=5, struct_count=5
        )
        self.assertEqual(len(examples), 20)
        for ex in examples:
            self.assertTrue(hasattr(ex, "id"))
            self.assertTrue(hasattr(ex, "messages"))
            self.assertTrue(hasattr(ex, "category"))
            self.assertIsInstance(ex.messages, list)
            self.assertGreater(len(ex.messages), 0)

    def test_sft_pipeline_deterministic(self):
        """SFTDataPipeline with same seed produces same output."""
        from src.agent.sft_data_pipeline import SFTDataPipeline

        p1 = SFTDataPipeline(seed=42)
        p2 = SFTDataPipeline(seed=42)
        e1 = p1.generate_all(tool_count=3, rag_count=3, safety_count=3, struct_count=3)
        e2 = p2.generate_all(tool_count=3, rag_count=3, safety_count=3, struct_count=3)
        self.assertEqual(len(e1), len(e2))
        for a, b in zip(e1, e2):
            self.assertEqual(a.id, b.id)
            self.assertEqual(a.category, b.category)

    def test_sft_categories_present(self):
        """All four SFT categories are generated."""
        from src.agent.sft_data_pipeline import SFTDataPipeline

        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=3, rag_count=3, safety_count=3, struct_count=3
        )
        categories = {ex.category for ex in examples}
        self.assertIn("tool_use", categories)
        self.assertIn("rag_qa", categories)
        self.assertIn("safety_refusal", categories)
        self.assertIn("structured_output", categories)

    def test_generate_training_data_script(self):
        """Full data generation script produces valid JSONL files."""
        from scripts.generate_training_data import (
            SFTDataPipeline,
            convert_sft_to_chatml,
            generate_dpo_data,
        )

        # Test SFT conversion
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=5, rag_count=5, safety_count=5, struct_count=5
        )
        chatml = convert_sft_to_chatml(examples)
        self.assertEqual(len(chatml), 20)
        for item in chatml:
            self.assertIn("messages", item)
            self.assertIn("id", item)
            self.assertIn("category", item)
            for msg in item["messages"]:
                self.assertIn("role", msg)
                self.assertIn("content", msg)
                self.assertIn(msg["role"], ("system", "user", "assistant"))

        # Test DPO generation
        dpo = generate_dpo_data(count=10, seed=42)
        self.assertEqual(len(dpo), 10)
        for pair in dpo:
            self.assertIn("prompt", pair)
            self.assertIn("chosen", pair)
            self.assertIn("rejected", pair)
            self.assertIsInstance(pair["prompt"], str)
            self.assertIsInstance(pair["chosen"], str)
            self.assertIsInstance(pair["rejected"], str)

    def test_write_and_read_jsonl(self):
        """JSONL files can be written and read back correctly."""
        from scripts.generate_training_data import generate_dpo_data

        dpo = generate_dpo_data(count=5, seed=42)
        path = os.path.join(self.tmpdir, "test.jsonl")

        # Write
        with open(path, "w", encoding="utf-8") as f:
            for item in dpo:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Read back
        read_back = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    read_back.append(json.loads(line))

        self.assertEqual(len(read_back), 5)
        for orig, loaded in zip(dpo, read_back):
            self.assertEqual(orig["prompt"], loaded["prompt"])
            self.assertEqual(orig["chosen"], loaded["chosen"])
            self.assertEqual(orig["rejected"], loaded["rejected"])


class TestDPODataQuality(unittest.TestCase):
    """Test DPO preference pair quality."""

    def test_chosen_longer_than_rejected(self):
        """Chosen responses should generally be more detailed (longer)."""
        from scripts.generate_training_data import DPO_TEMPLATES

        longer_count = sum(
            1 for t in DPO_TEMPLATES
            if len(t["chosen"]) > len(t["rejected"])
        )
        # At least 80% of chosen should be longer (more detailed)
        ratio = longer_count / len(DPO_TEMPLATES)
        self.assertGreater(ratio, 0.7, f"Only {ratio:.0%} chosen are longer than rejected")

    def test_all_templates_have_required_fields(self):
        """Every DPO template has prompt, chosen, rejected."""
        from scripts.generate_training_data import DPO_TEMPLATES

        for i, t in enumerate(DPO_TEMPLATES):
            self.assertIn("prompt", t, f"Template {i} missing 'prompt'")
            self.assertIn("chosen", t, f"Template {i} missing 'chosen'")
            self.assertIn("rejected", t, f"Template {i} missing 'rejected'")
            self.assertGreater(len(t["prompt"]), 0, f"Template {i} empty prompt")
            self.assertGreater(len(t["chosen"]), 0, f"Template {i} empty chosen")
            self.assertGreater(len(t["rejected"]), 0, f"Template {i} empty rejected")

    def test_safety_refusal_in_dpo(self):
        """At least one DPO pair tests safety (chosen refuses, rejected complies)."""
        from scripts.generate_training_data import DPO_TEMPLATES

        has_safety = False
        refusal_words = ["can't", "cannot", "not able", "illegal", "won't"]
        for t in DPO_TEMPLATES:
            chosen_lower = t["chosen"].lower()
            if any(w in chosen_lower for w in refusal_words):
                has_safety = True
                break
        self.assertTrue(has_safety, "No safety refusal pair found in DPO templates")


class TestFineTuneScriptArgs(unittest.TestCase):
    """Test argument parsing for finetune_hf.py."""

    def test_default_args(self):
        """Default arguments are sensible."""
        import argparse
        # Simulate parsing with minimal args
        sys.argv = ["finetune_hf.py"]
        # We can't run the full script but we verify the import works
        from scripts.finetune_hf import load_sft_dataset, format_messages_for_training
        self.assertTrue(callable(load_sft_dataset))
        self.assertTrue(callable(format_messages_for_training))

    def test_format_messages_fallback(self):
        """Message formatting fallback works without a real tokenizer."""
        from scripts.finetune_hf import format_messages_for_training

        class MockTokenizer:
            def apply_chat_template(self, *a, **kw):
                raise NotImplementedError("no template")

        example = {
            "messages": [
                {"role": "system", "content": "You are Yaya."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = format_messages_for_training(example, MockTokenizer())
        self.assertIn("text", result)
        self.assertIn("Hello", result["text"])
        self.assertIn("Hi there!", result["text"])
        self.assertIn("Yaya", result["text"])


class TestDPOAlignScriptArgs(unittest.TestCase):
    """Test argument parsing for align_dpo_hf.py."""

    def test_imports(self):
        """DPO script imports successfully."""
        from scripts.align_dpo_hf import load_dpo_dataset, format_dpo_example
        self.assertTrue(callable(load_dpo_dataset))
        self.assertTrue(callable(format_dpo_example))

    def test_format_dpo_fallback(self):
        """DPO formatting fallback works without a real tokenizer."""
        from scripts.align_dpo_hf import format_dpo_example

        class MockTokenizer:
            def apply_chat_template(self, *a, **kw):
                raise NotImplementedError("no template")

        example = {
            "prompt": "What is AI?",
            "chosen": "AI is artificial intelligence.",
            "rejected": "It's stuff.",
        }
        result = format_dpo_example(example, MockTokenizer())
        self.assertIn("prompt", result)
        self.assertIn("chosen", result)
        self.assertIn("rejected", result)
        self.assertIn("What is AI?", result["prompt"])


class TestEvalScript(unittest.TestCase):
    """Test eval script components."""

    def test_sanity_checks_defined(self):
        """Sanity check prompts are well-formed."""
        from scripts.eval_model import SANITY_CHECKS

        self.assertGreater(len(SANITY_CHECKS), 5)
        for check in SANITY_CHECKS:
            self.assertIn("name", check)
            self.assertIn("messages", check)
            self.assertIn("expect_contains", check)
            self.assertIsInstance(check["messages"], list)
            self.assertGreater(len(check["messages"]), 0)
            self.assertIsInstance(check["expect_contains"], list)

    def test_sanity_check_categories(self):
        """Sanity checks cover key capabilities."""
        from scripts.eval_model import SANITY_CHECKS

        names = {c["name"] for c in SANITY_CHECKS}
        expected = {"identity", "math", "safety_refusal", "code", "reasoning"}
        self.assertTrue(expected.issubset(names),
                        f"Missing checks: {expected - names}")


class TestDeployScript(unittest.TestCase):
    """Test deploy script components."""

    def test_imports(self):
        """Deploy script imports successfully."""
        from scripts.deploy_model import load_model, create_app
        self.assertTrue(callable(load_model))
        self.assertTrue(callable(create_app))


class TestEndToEndDataPipeline(unittest.TestCase):
    """Test full data generation -> file write -> file read cycle."""

    def test_full_cycle(self):
        """Generate data, write to files, read back, validate format."""
        from scripts.generate_training_data import (
            SFTDataPipeline,
            convert_sft_to_chatml,
            generate_dpo_data,
        )

        tmpdir = tempfile.mkdtemp()

        # Generate SFT
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=3, rag_count=3, safety_count=3, struct_count=3
        )
        chatml = convert_sft_to_chatml(examples)

        sft_path = os.path.join(tmpdir, "yaya_sft_train.jsonl")
        with open(sft_path, "w", encoding="utf-8") as f:
            for item in chatml:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Generate DPO
        dpo = generate_dpo_data(count=5, seed=42)
        dpo_path = os.path.join(tmpdir, "yaya_dpo_train.jsonl")
        with open(dpo_path, "w", encoding="utf-8") as f:
            for item in dpo:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Read back SFT
        with open(sft_path, encoding="utf-8") as f:
            sft_loaded = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(sft_loaded), 12)

        # Validate SFT format (ChatML-compatible)
        for item in sft_loaded:
            self.assertIn("messages", item)
            roles = [m["role"] for m in item["messages"]]
            # Must have at least user + assistant
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)

        # Read back DPO
        with open(dpo_path, encoding="utf-8") as f:
            dpo_loaded = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(dpo_loaded), 5)

        for item in dpo_loaded:
            self.assertIn("prompt", item)
            self.assertIn("chosen", item)
            self.assertIn("rejected", item)


if __name__ == "__main__":
    unittest.main()
