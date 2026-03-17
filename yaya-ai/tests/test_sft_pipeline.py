"""Tests for the SFT data pipeline: tool use, RAG, safety, structured output generators."""

import os
import json
import tempfile

import pytest

from src.agent.sft_data_pipeline import (
    SFTExample,
    ToolUseSFTGenerator,
    RAGSFTGenerator,
    SafetySFTGenerator,
    StructuredOutputSFTGenerator,
    SFTDataPipeline,
)


class TestToolUseSFTGenerator:
    def test_generate_examples(self):
        gen = ToolUseSFTGenerator()
        examples = gen.generate(count=5)
        assert len(examples) == 5
        assert all(ex.category == "tool_use" for ex in examples)

    def test_examples_have_tool_calls(self):
        gen = ToolUseSFTGenerator()
        examples = gen.generate(count=3)
        for ex in examples:
            assert "tool" in ex.metadata
            assert len(ex.messages) >= 3  # user, assistant+tool, tool_result, assistant
            assert ex.formatted  # Non-empty formatted string

    def test_examples_have_unique_ids(self):
        gen = ToolUseSFTGenerator()
        examples = gen.generate(count=10)
        ids = [ex.id for ex in examples]
        assert len(set(ids)) == len(ids)


class TestRAGSFTGenerator:
    def test_generate_examples(self):
        gen = RAGSFTGenerator()
        examples = gen.generate(count=5)
        assert len(examples) == 5
        assert all(ex.category == "rag_qa" for ex in examples)

    def test_examples_have_context(self):
        gen = RAGSFTGenerator()
        examples = gen.generate(count=3)
        for ex in examples:
            assert ex.metadata.get("has_context") is True
            # System message should contain context
            system_msg = [m for m in ex.messages if m["role"] == "system"]
            assert len(system_msg) == 1
            assert "Context:" in system_msg[0]["content"]


class TestSafetySFTGenerator:
    def test_generate_examples(self):
        gen = SafetySFTGenerator()
        examples = gen.generate(count=5)
        assert len(examples) == 5
        assert all(ex.category == "safety_refusal" for ex in examples)

    def test_examples_have_refusals(self):
        gen = SafetySFTGenerator()
        examples = gen.generate(count=3)
        for ex in examples:
            assert "harm_category" in ex.metadata
            # Last assistant message should be a refusal
            assistant_msgs = [m for m in ex.messages if m["role"] == "assistant"]
            assert len(assistant_msgs) >= 1
            assert len(assistant_msgs[-1]["content"]) > 10  # Non-trivial refusal


class TestStructuredOutputSFTGenerator:
    def test_generate_examples(self):
        gen = StructuredOutputSFTGenerator()
        examples = gen.generate(count=5)
        assert len(examples) == 5
        assert all(ex.category == "structured_output" for ex in examples)

    def test_examples_have_valid_json_responses(self):
        gen = StructuredOutputSFTGenerator()
        examples = gen.generate(count=6)
        for ex in examples:
            assistant_msgs = [m for m in ex.messages if m["role"] == "assistant"]
            # Response should be valid JSON
            parsed = json.loads(assistant_msgs[-1]["content"])
            assert parsed is not None


class TestSFTDataPipeline:
    def test_generate_all(self):
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=5, rag_count=5, safety_count=5, struct_count=5
        )
        assert len(examples) == 20

    def test_category_distribution(self):
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=10, rag_count=8, safety_count=6, struct_count=4
        )
        stats = pipeline.get_stats(examples)
        assert stats["categories"]["tool_use"] == 10
        assert stats["categories"]["rag_qa"] == 8
        assert stats["categories"]["safety_refusal"] == 6
        assert stats["categories"]["structured_output"] == 4

    def test_save_and_load_jsonl(self):
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=3, rag_count=3, safety_count=2, struct_count=2
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sft_data.jsonl")
            pipeline.save_jsonl(examples, path)

            # File should exist and have correct number of lines
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 10

            # Load back
            loaded = pipeline.load_jsonl(path)
            assert len(loaded) == 10
            assert loaded[0].id == examples[0].id
            assert loaded[0].category == examples[0].category

    def test_stats(self):
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=5, rag_count=5, safety_count=5, struct_count=5
        )
        stats = pipeline.get_stats(examples)
        assert stats["total"] == 20
        assert stats["approx_tokens"] > 0
        assert len(stats["categories"]) == 4

    def test_all_examples_have_formatted_text(self):
        pipeline = SFTDataPipeline(seed=42)
        examples = pipeline.generate_all(
            tool_count=3, rag_count=3, safety_count=3, struct_count=3
        )
        for ex in examples:
            assert len(ex.formatted) > 0
            assert len(ex.messages) >= 2  # At least user + assistant

    def test_deterministic_with_seed(self):
        p1 = SFTDataPipeline(seed=123)
        p2 = SFTDataPipeline(seed=123)
        e1 = p1.generate_all(tool_count=3, rag_count=3, safety_count=3, struct_count=3)
        e2 = p2.generate_all(tool_count=3, rag_count=3, safety_count=3, struct_count=3)
        # Same seed -> same order (after shuffle)
        assert [ex.id for ex in e1] == [ex.id for ex in e2]
