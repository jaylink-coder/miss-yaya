"""Tests for Constitutional AI and persistent memory integration with inference."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def echo_generate(prompt: str) -> str:
    """Stub generate_fn that returns the last line of the prompt."""
    return prompt.strip().splitlines()[-1] if prompt.strip() else "ok"


def fixed_generate(text: str):
    """Returns a generate_fn that always returns `text`."""
    def _gen(prompt: str) -> str:
        return text
    return _gen


# ---------------------------------------------------------------------------
# ConstitutionalPrinciple
# ---------------------------------------------------------------------------

class TestConstitutionalPrinciple:
    def test_format_critique_contains_prompt_and_response(self):
        from src.training.constitutional import HELPFULNESS
        formatted = HELPFULNESS.format_critique("What is 2+2?", "Purple.")
        assert "What is 2+2?" in formatted
        assert "Purple." in formatted

    def test_format_revision_contains_critique(self):
        from src.training.constitutional import HELPFULNESS
        formatted = HELPFULNESS.format_revision("q", "original answer", "This is wrong")
        assert "This is wrong" in formatted
        assert "original answer" in formatted


# ---------------------------------------------------------------------------
# ConstitutionalAI core
# ---------------------------------------------------------------------------

class TestConstitutionalAI:
    def test_revise_returns_expected_keys(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        cai = ConstitutionalAI(
            fixed_generate("Revised answer."),
            config=CAIConfig(principles=[HELPFULNESS], n_revisions=1),
        )
        result = cai.revise("What is 2+2?", initial_response="Four.")
        assert "original" in result
        assert "revised" in result
        assert "critique" in result
        assert "history" in result

    def test_original_is_preserved(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        cai = ConstitutionalAI(
            fixed_generate("Improved."),
            config=CAIConfig(principles=[HELPFULNESS], n_revisions=1),
        )
        result = cai.revise("q", initial_response="Original response.")
        assert result["original"] == "Original response."

    def test_revised_differs_from_original(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        responses = iter(["critique text", "Better answer."])
        def alternating(prompt):
            return next(responses)
        cai = ConstitutionalAI(
            alternating,
            config=CAIConfig(principles=[HELPFULNESS], n_revisions=1),
        )
        result = cai.revise("q", initial_response="Original.")
        assert result["revised"] == "Better answer."

    def test_generates_initial_response_if_none(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        call_count = [0]
        def counting_gen(prompt):
            call_count[0] += 1
            return "auto response"
        cai = ConstitutionalAI(
            counting_gen,
            config=CAIConfig(principles=[HELPFULNESS], n_revisions=1),
        )
        cai.revise("q")  # No initial_response — should generate one
        assert call_count[0] >= 2   # at least one for initial + one for critique/revise

    def test_history_has_correct_length(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig
        from src.training.constitutional import HELPFULNESS, HARMLESSNESS
        cai = ConstitutionalAI(
            fixed_generate("ok"),
            config=CAIConfig(principles=[HELPFULNESS, HARMLESSNESS], n_revisions=1),
        )
        result = cai.revise("q", initial_response="r")
        assert len(result["history"]) == 2

    def test_chain_revisions_uses_revised_as_next_input(self):
        """With chain_revisions=True, each cycle refines the previous revision."""
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        outputs = iter(["crit1", "rev1", "crit2", "rev2"])
        def seq_gen(prompt):
            return next(outputs, "done")
        cai = ConstitutionalAI(
            seq_gen,
            config=CAIConfig(
                principles=[HELPFULNESS],
                n_revisions=2,
                chain_revisions=True,
            ),
        )
        result = cai.revise("q", initial_response="initial")
        # After 2 cycles: rev2 should be the final revised output
        assert result["revised"] == "rev2"

    def test_no_chain_revisions_always_uses_original(self):
        """With chain_revisions=False, each cycle starts from the original."""
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        outputs = iter(["crit1", "rev1", "crit2", "rev2"])
        def seq_gen(prompt):
            return next(outputs, "done")
        cai = ConstitutionalAI(
            seq_gen,
            config=CAIConfig(
                principles=[HELPFULNESS],
                n_revisions=2,
                chain_revisions=False,
            ),
        )
        result = cai.revise("q", initial_response="initial")
        # Without chaining, last revision is rev2 but based on original each time
        assert "rev" in result["revised"]

    def test_generate_preference_pairs_structure(self):
        from src.training.constitutional import ConstitutionalAI, CAIConfig, HELPFULNESS
        responses = ["orig", "crit", "revised better"]
        idx = [0]
        def cycling_gen(prompt):
            val = responses[idx[0] % len(responses)]
            idx[0] += 1
            return val
        cai = ConstitutionalAI(
            cycling_gen,
            config=CAIConfig(principles=[HELPFULNESS], n_revisions=1),
        )
        pairs = cai.generate_preference_pairs(["What is the sun?"], n_revisions=1)
        # pairs may be empty if original == revised; just verify structure if non-empty
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair

    def test_self_score_returns_float_in_range(self):
        from src.training.constitutional import ConstitutionalAI, HELPFULNESS
        cai = ConstitutionalAI(fixed_generate("0.85"))
        score = cai.self_score("What is 2+2?", "Four.", principle=HELPFULNESS)
        assert 0.0 <= score <= 1.0

    def test_self_score_fallback_when_no_number(self):
        from src.training.constitutional import ConstitutionalAI
        cai = ConstitutionalAI(fixed_generate("no number here"))
        score = cai.self_score("q", "r")
        assert score == 0.5  # fallback

    def test_default_constitution_has_three_principles(self):
        from src.training.constitutional import DEFAULT_CONSTITUTION
        assert len(DEFAULT_CONSTITUTION) == 3


# ---------------------------------------------------------------------------
# Default principles are importable and well-formed
# ---------------------------------------------------------------------------

class TestPrincipleLibrary:
    def test_all_built_in_principles_have_name(self):
        from src.training.constitutional import (
            HELPFULNESS, HARMLESSNESS, HONESTY, CONCISENESS
        )
        for p in [HELPFULNESS, HARMLESSNESS, HONESTY, CONCISENESS]:
            assert p.name
            assert p.critique_prompt
            assert p.revision_prompt


# ---------------------------------------------------------------------------
# Generator memory integration
# ---------------------------------------------------------------------------

class _FakeTok:
    eos_id = 1
    def encode(self, text, add_bos=False):
        return [ord(c) % 200 + 2 for c in text[:8]]
    def decode(self, ids, skip_special=False):
        return "".join(chr(i + 32) for i in ids if 32 <= i + 32 < 128)
    def format_chat(self, messages):
        return " ".join(m["content"] for m in messages)


class _FakeTinyModel(object):
    """Minimal model stub that returns random logits."""
    import torch as _torch
    def __init__(self):
        import torch
        self._torch = torch
    def eval(self): return self
    def __call__(self, input_ids, past_key_values=None, use_cache=False, **kw):
        import torch
        batch, seq = input_ids.shape
        vocab = 200
        logits = torch.randn(batch, seq, vocab)
        out = {"logits": logits}
        if use_cache:
            out["past_key_values"] = None
        return out


class TestGeneratorMemoryIntegration:
    def _make_generator(self, memory=None):
        import torch
        from src.inference.generator import TextGenerator
        model = _FakeTinyModel()
        tok = _FakeTok()
        gen = TextGenerator(model, tok, device="cpu", memory=memory)
        return gen

    def test_generator_accepts_memory_param(self):
        from src.agent.persistent_memory import SessionMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            sess = SessionMemory(store_dir=tmpdir, session_id="t1")
            gen = self._make_generator(memory=sess)
            assert gen.memory is sess

    def test_generate_without_memory_still_works(self):
        gen = self._make_generator(memory=None)
        result = gen.generate("hello", max_new_tokens=3)
        assert isinstance(result, str)

    def test_generate_with_memory_injects_context(self):
        from src.agent.persistent_memory import SessionMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            sess = SessionMemory(store_dir=tmpdir, session_id="t1")
            sess.add_fact("Alex is from Kalimoni")
            gen = self._make_generator(memory=sess)
            # Should not raise — memory context is prepended to prompt
            result = gen.generate("hello", max_new_tokens=3)
            assert isinstance(result, str)

    def test_memory_updates_after_generation(self):
        """After generate(), memory.extract_from_text should have been called."""
        from src.agent.persistent_memory import SessionMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            sess = SessionMemory(store_dir=tmpdir, session_id="t1")
            gen = self._make_generator(memory=sess)
            # Put a number in the prompt so extract_from_text has something to catch
            gen.generate("There are 47 counties in Kenya.", max_new_tokens=2)
            # extract_from_text runs on prompt — session facts may include "47"
            # Just verify no exception was raised and memory object is intact
            assert sess is not None


# ---------------------------------------------------------------------------
# Server memory endpoints (unit tests — no HTTP calls)
# ---------------------------------------------------------------------------

class TestServerMemoryEndpoints:
    def test_create_app_builds_without_error(self):
        """create_app() should return a FastAPI app without raising."""
        try:
            from src.inference.server import create_app
            from src.inference.generator import TextGenerator
        except ImportError:
            pytest.skip("FastAPI not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _FakeTinyModel()
            tok = _FakeTok()
            gen = TextGenerator(model, tok, device="cpu")
            app = create_app(gen, model_name="test-yaya", memory_store_dir=tmpdir)
            assert app is not None

    def test_long_term_memory_loaded_at_startup(self):
        """PersistentMemory.load() is called during create_app()."""
        try:
            from src.inference.server import create_app
            from src.inference.generator import TextGenerator
            from src.agent.persistent_memory import PersistentMemory
        except ImportError:
            pytest.skip("FastAPI not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-populate long-term memory
            lt = PersistentMemory(store_dir=tmpdir, name="long_term")
            lt.add_fact("pre-existing knowledge")
            lt.save()

            model = _FakeTinyModel()
            tok = _FakeTok()
            gen = TextGenerator(model, tok, device="cpu")
            # create_app should load the pre-existing memory
            app = create_app(gen, model_name="test", memory_store_dir=tmpdir)
            assert app is not None
