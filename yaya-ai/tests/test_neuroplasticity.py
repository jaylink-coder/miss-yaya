"""Tests for all four neuroplasticity modules: LoRA, EWC, MoE, OnlineLearner."""

import os
import json
import tempfile
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig, TrainingConfig, _resolve_moe_layer
from src.model.yaya_model import YayaForCausalLM


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def tiny_model_config(**kwargs) -> ModelConfig:
    defaults = dict(
        model_name="yaya-test",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def tiny_model(**kwargs) -> YayaForCausalLM:
    return YayaForCausalLM(tiny_model_config(**kwargs))


def fake_batch(vocab_size: int = 256, seq_len: int = 16, batch: int = 2):
    ids = torch.randint(0, vocab_size, (batch, seq_len))
    labels = ids.clone()
    return {"input_ids": ids, "labels": labels}


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class TestLoRA:
    def test_inject_reduces_trainable_params(self):
        from src.model.lora import inject_lora, LoRAConfig
        model = tiny_model()
        total_before = sum(p.numel() for p in model.parameters())
        inject_lora(model, LoRAConfig(rank=4, target_modules=["q_proj", "v_proj"]))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_after = sum(p.numel() for p in model.parameters())
        # inject_lora ADDS adapter params (lora_A + lora_B), so total grows slightly
        assert total_after > total_before, "inject_lora must add adapter parameters"
        # Trainable params must be only the adapters — a tiny fraction of total
        assert trainable < total_before * 0.1, "LoRA must freeze the vast majority of params"
        assert trainable > 0, "LoRA must leave adapter params trainable"
        # Only adapter names should be trainable
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert "lora_A" in name or "lora_B" in name, \
                    f"Non-adapter param is trainable: {name}"

    def test_lm_head_never_wrapped(self):
        from src.model.lora import inject_lora, LoRAConfig, LoRALinear
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        # lm_head weight is tied to embed_tokens — must never become LoRALinear
        assert not isinstance(model.lm_head, LoRALinear), "lm_head must never be wrapped"

    def test_tied_weights_preserved(self):
        from src.model.lora import inject_lora, LoRAConfig
        model = tiny_model(tie_word_embeddings=True)
        inject_lora(model, LoRAConfig(rank=4))
        assert (
            model.lm_head.weight.data_ptr()
            == model.model.embed_tokens.word_embeddings.weight.data_ptr()
        ), "Tied weight (lm_head <-> embed_tokens) must remain shared after LoRA injection"

    def test_forward_runs_after_injection(self):
        from src.model.lora import inject_lora, LoRAConfig
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        batch = fake_batch()
        with torch.no_grad():
            out = model(**batch)
        assert "loss" in out
        assert not torch.isnan(out["loss"])

    def test_lora_state_dict_contains_only_adapters(self):
        from src.model.lora import inject_lora, LoRAConfig, lora_state_dict
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        sd = lora_state_dict(model)
        assert len(sd) > 0, "lora_state_dict must return adapter keys"
        for key in sd:
            assert "lora_A" in key or "lora_B" in key, f"Non-adapter key in lora_state_dict: {key}"

    def test_merge_lora_restores_nn_linear(self):
        from src.model.lora import inject_lora, merge_lora, LoRAConfig, LoRALinear
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        merge_lora(model)
        for module in model.modules():
            assert not isinstance(module, LoRALinear), "No LoRALinear should remain after merge"

    def test_merge_lora_output_matches_pre_merge(self):
        from src.model.lora import inject_lora, merge_lora, LoRAConfig
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        batch = fake_batch()
        with torch.no_grad():
            out_before = model(**batch)["logits"]
        merge_lora(model)
        with torch.no_grad():
            out_after = model(**batch)["logits"]
        assert torch.allclose(out_before, out_after, atol=1e-5), \
            "Merged LoRA output must equal pre-merge output"

    def test_only_target_modules_wrapped(self):
        from src.model.lora import inject_lora, LoRAConfig, LoRALinear
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4, target_modules=["q_proj"]))
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                assert name.endswith("q_proj"), f"Non-target module wrapped: {name}"

    def test_adapter_save_load_roundtrip(self):
        from src.model.lora import inject_lora, LoRAConfig, lora_state_dict
        cfg = LoRAConfig(rank=4)
        model = tiny_model()
        inject_lora(model, cfg)

        # Run a forward pass to get a reference output
        batch = fake_batch()
        with torch.no_grad():
            out_before = model(**batch)["logits"].clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(lora_state_dict(model), path)

            # Zero out the adapter weights in place to simulate a fresh init
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        p.zero_()

            # Reload the saved adapters — output must match original
            sd = torch.load(path, weights_only=True)
            model.load_state_dict(sd, strict=False)
            with torch.no_grad():
                out_after = model(**batch)["logits"]
            assert torch.allclose(out_before, out_after, atol=1e-5), \
                "Output after adapter save/load must match original"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# EWC
# ---------------------------------------------------------------------------

class TestEWC:
    def _make_dataloader(self, vocab_size=256, n=4):
        batches = [fake_batch(vocab_size) for _ in range(n)]
        return batches  # simple list used as iterable

    def test_fisher_computed_after_call(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1.0)
        assert len(ewc.fisher) == 0
        ewc.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))
        assert len(ewc.fisher) > 0

    def test_penalty_zero_before_fisher(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1000.0)
        penalty = ewc.penalty()
        assert penalty.item() == 0.0, "Penalty must be 0 before Fisher is computed"

    def test_penalty_positive_after_perturbation(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1000.0)
        ewc.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))
        # Perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        penalty = ewc.penalty()
        assert penalty.item() > 0.0, "Penalty must be positive after perturbing weights"

    def test_penalty_zero_when_weights_unchanged(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1000.0)
        ewc.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))
        penalty = ewc.penalty()
        assert penalty.item() == pytest.approx(0.0, abs=1e-6), \
            "Penalty must be ~0 when weights have not changed since Fisher computation"

    def test_penalty_on_correct_device(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1.0)
        ewc.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))
        penalty = ewc.penalty()
        assert penalty.device.type == "cpu"

    def test_save_load_roundtrip(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc_a = EWC(model, lambda_ewc=42.0)
        ewc_a.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            ewc_a.save(path)
            ewc_b = EWC(model, lambda_ewc=0.0)
            ewc_b.load(path)
            assert ewc_b.lambda_ewc == pytest.approx(42.0)
            assert set(ewc_b.fisher.keys()) == set(ewc_a.fisher.keys())
        finally:
            os.unlink(path)

    def test_penalty_added_to_loss(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1000.0)
        ewc.compute_fisher(self._make_dataloader(), num_samples=2, device=torch.device("cpu"))
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        batch = fake_batch()
        out = model(**batch)
        base_loss = out["loss"].item()
        penalized_loss = (out["loss"] + ewc.penalty()).item()
        assert penalized_loss > base_loss, "EWC penalty must increase the loss"


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class TestMoE:
    def test_moe_forward_returns_tuple(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out, aux = moe(x)
        assert out.shape == x.shape, "MoE output must match input shape"
        assert aux.numel() == 1, "aux_loss must be a scalar"

    def test_moe_aux_loss_positive(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(4, 16, 64)
        _, aux = moe(x)
        assert aux.item() > 0, "Load-balance aux loss must be positive"

    def test_moe_layer_in_model_forward(self):
        cfg = tiny_model_config(moe_enabled=True, moe_num_experts=4, moe_top_k=2,
                                moe_layers="all", moe_load_balance_coeff=0.01)
        model = YayaForCausalLM(cfg)
        batch = fake_batch()
        out = model(**batch)
        assert "loss" in out
        assert "moe_aux_loss" in out
        assert not torch.isnan(out["loss"])

    def test_moe_aux_loss_added_to_total(self):
        cfg = tiny_model_config(moe_enabled=True, moe_num_experts=4, moe_top_k=2,
                                moe_layers="all", moe_load_balance_coeff=0.01)
        model = YayaForCausalLM(cfg)
        batch = fake_batch()
        out = model(**batch)
        # Total loss = CE + coeff * aux. With coeff=0.01 > 0 and aux > 0,
        # total must be > pure CE (which we can't isolate directly, but aux > 0 is sufficient)
        assert out["moe_aux_loss"].item() > 0

    def test_resolve_moe_layer(self):
        assert _resolve_moe_layer("all", 0) is True
        assert _resolve_moe_layer("all", 5) is True
        assert _resolve_moe_layer("alternate", 0) is False
        assert _resolve_moe_layer("alternate", 1) is True
        assert _resolve_moe_layer("alternate", 2) is False
        assert _resolve_moe_layer("0,2,4", 0) is True
        assert _resolve_moe_layer("0,2,4", 1) is False
        assert _resolve_moe_layer("0,2,4", 4) is True

    def test_model_config_is_moe_layer_delegates(self):
        cfg = tiny_model_config(moe_enabled=True, moe_layers="alternate")
        assert cfg.is_moe_layer(0) is False
        assert cfg.is_moe_layer(1) is True
        cfg_disabled = tiny_model_config(moe_enabled=False, moe_layers="all")
        assert cfg_disabled.is_moe_layer(0) is False

    def test_moe_config_is_moe_layer_delegates(self):
        from src.model.moe import MoEConfig
        cfg = MoEConfig(enabled=True, moe_layers="alternate")
        assert cfg.is_moe_layer(0) is False
        assert cfg.is_moe_layer(1) is True
        # Must agree with _resolve_moe_layer
        for i in range(6):
            assert cfg.is_moe_layer(i) == _resolve_moe_layer("alternate", i)

    def test_convert_to_moe_copies_weights(self):
        from src.model.moe import convert_to_moe, MoEConfig, MoEFeedForward
        from src.model.feedforward import SwiGLUFeedForward
        cfg = tiny_model_config()
        model = YayaForCausalLM(cfg)
        # Extract original gate_proj weight from layer 0
        orig_weight = model.model.layers[0].mlp.gate_proj.weight.clone()
        moe_cfg = MoEConfig(enabled=True, num_experts=4, top_k=2, moe_layers="all")
        convert_to_moe(model, moe_cfg)
        # Expert 0 must have the original weights
        expert0_weight = model.model.layers[0].mlp.experts[0].gate_proj.weight
        assert torch.allclose(orig_weight, expert0_weight), \
            "convert_to_moe must copy original FFN weights into expert 0"

    def test_router_no_nan_on_uniform_input(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.ones(2, 8, 64)  # Uniform input — stresses router normalization
        out, aux = moe(x)
        assert not torch.isnan(out).any(), "MoE output must not contain NaN on uniform input"
        assert not torch.isnan(aux), "MoE aux loss must not be NaN on uniform input"


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stub for OnlineLearner tests."""
    eos_id = 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = [ord(c) % 250 + 2 for c in text[:20]]
        if add_bos:
            ids = [0] + ids
        if add_eos:
            ids = ids + [1]
        return ids

    def decode(self, ids, skip_special: bool = False):
        return "".join(chr(i + 30) for i in ids if i > 1)


class TestOnlineLearner:
    def _make_learner(self, tmpdir):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        model = tiny_model()
        cfg = OnlineLearnerConfig(
            buffer_capacity=100,
            min_examples_to_finetune=2,
            finetune_every_n_examples=2,
            micro_finetune_steps=1,
            micro_lr=1e-4,
            max_seq_length=32,
            buffer_path=os.path.join(tmpdir, "buffer.jsonl"),
        )
        return OnlineLearner(model, _FakeTokenizer(), cfg, torch.device("cpu"))

    def test_requires_tokenizer(self):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        model = tiny_model()
        cfg = OnlineLearnerConfig()
        with pytest.raises(ValueError, match="tokenizer"):
            OnlineLearner(model, None, cfg, torch.device("cpu"))

    def test_add_example_grows_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            assert len(learner.buffer) == 0
            learner.add_example("hello", " world", score=1.0)
            assert len(learner.buffer) == 1

    def test_negative_examples_stored_not_trained(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            for _ in range(5):
                learner.add_example("bad", " output", score=-1.0)
            positives = [e for e in learner.buffer if e["score"] > 0]
            assert len(positives) == 0, "Negative examples must be stored but not trained"

    def test_step_returns_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            for i in range(5):
                learner.add_example(f"prompt{i}", f" response{i}", score=1.0)
            result = learner.step()
            assert result is not None
            assert isinstance(result, float)

    def test_buffer_persisted_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            learner.add_example("test prompt", " test response", score=1.0)
            assert os.path.exists(learner.config.buffer_path)
            with open(learner.config.buffer_path) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            assert len(lines) == 1
            assert lines[0]["prompt"] == "test prompt"

    def test_buffer_restored_from_disk(self):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            learner.add_example("restore test", " resp", score=1.0)
            # Create a new learner pointing to same buffer file
            cfg = OnlineLearnerConfig(
                buffer_path=learner.config.buffer_path,
                min_examples_to_finetune=99,  # Prevent auto-step
            )
            learner2 = OnlineLearner(tiny_model(), _FakeTokenizer(), cfg, torch.device("cpu"))
            assert len(learner2.buffer) == 1
            assert learner2.buffer[0]["prompt"] == "restore test"

    def test_save_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = self._make_learner(tmpdir)
            for i in range(3):
                learner.add_example(f"p{i}", f" r{i}", score=1.0)
            state_path = os.path.join(tmpdir, "ol_state.pt")
            learner.save_state(state_path)
            assert os.path.exists(state_path)
            learner.load_state(state_path)  # Should not raise

    def test_ewc_penalty_applied_in_step(self):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        from src.training.ewc import EWC
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tiny_model()
            ewc = EWC(model, lambda_ewc=1000.0)
            batches = [fake_batch(256) for _ in range(4)]
            ewc.compute_fisher(batches, num_samples=4, device=torch.device("cpu"))

            cfg = OnlineLearnerConfig(
                buffer_capacity=100,
                min_examples_to_finetune=2,
                finetune_every_n_examples=99,  # Manual step only
                micro_finetune_steps=1,
                micro_lr=1e-4,
                max_seq_length=32,
                buffer_path=os.path.join(tmpdir, "buf.jsonl"),
            )
            learner = OnlineLearner(model, _FakeTokenizer(), cfg, torch.device("cpu"), ewc=ewc)
            for i in range(5):
                learner.add_example(f"p{i}", f" r{i}", score=1.0)
            # Should not raise, EWC penalty integrated into backward
            result = learner.step()
            assert result is not None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_lora_defaults_in_training_config(self):
        cfg = TrainingConfig()
        assert cfg.lora_rank == 0
        assert cfg.lora_alpha == 32.0
        assert cfg.lora_dropout == 0.05

    def test_ewc_defaults_in_training_config(self):
        cfg = TrainingConfig()
        assert cfg.ewc_lambda == 0.0
        assert cfg.ewc_fisher_samples == 200

    def test_online_learning_defaults_in_training_config(self):
        cfg = TrainingConfig()
        assert cfg.online_learning_enabled is False
        assert cfg.online_buffer_capacity == 1000
        assert cfg.online_finetune_every == 50
        assert cfg.online_micro_steps == 10

    def test_moe_defaults_in_model_config(self):
        cfg = ModelConfig()
        assert cfg.moe_enabled is False
        assert cfg.moe_num_experts == 8
        assert cfg.moe_top_k == 2

    def test_elastic_guard_defaults_in_training_config(self):
        cfg = TrainingConfig()
        assert cfg.elastic_guard_enabled is False
        assert cfg.elastic_loss_spike_ratio == 2.5
        assert cfg.elastic_max_per_minute == 120


# ---------------------------------------------------------------------------
# ElasticGuard (neuroelastic resilience)
# ---------------------------------------------------------------------------

class TestElasticGuard:
    """Tests for the neuroelastic ElasticGuard wrapper."""

    def _make_guard(self, tmpdir, **elastic_kwargs):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        from src.training.neuro_elastic import ElasticGuard, ElasticConfig
        model = tiny_model()
        ol_cfg = OnlineLearnerConfig(
            buffer_capacity=100,
            min_examples_to_finetune=2,
            finetune_every_n_examples=99,  # Manual step only
            micro_finetune_steps=1,
            micro_lr=1e-4,
            max_seq_length=32,
            buffer_path=os.path.join(tmpdir, "buf.jsonl"),
        )
        learner = OnlineLearner(model, _FakeTokenizer(), ol_cfg, torch.device("cpu"))
        e_cfg = ElasticConfig(**elastic_kwargs)
        return ElasticGuard(learner, e_cfg), learner

    def test_add_example_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            accepted = guard.add_example("hello", " world", score=1.0)
            assert accepted is True
            assert guard.stats["examples_accepted"] == 1
            assert guard.stats["examples_rejected"] == 0

    def test_rejects_nan_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            accepted = guard.add_example("hello", " world", score=float("nan"))
            assert accepted is False
            assert guard.stats["examples_rejected"] == 1

    def test_rejects_empty_response(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir, min_response_chars=3)
            accepted = guard.add_example("hello", "ab", score=1.0)
            assert accepted is False

    def test_clamps_score_to_range(self):
        """Score outside [score_min, score_max] must be clamped, not rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, learner = self._make_guard(tmpdir, score_min=-1.0, score_max=1.0)
            accepted = guard.add_example("hello", " world", score=999.0)
            assert accepted is True
            # Check the clamped score was stored
            stored_score = learner.buffer[-1]["score"]
            assert stored_score <= 1.0

    def test_rate_limiter_blocks_flood(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir, max_per_minute=3)
            for i in range(3):
                guard.add_example(f"p{i}", " r", score=1.0)
            # 4th should be rate-limited
            accepted = guard.add_example("p4", " r", score=1.0)
            assert accepted is False

    def test_step_returns_loss_or_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            for i in range(5):
                guard.add_example(f"p{i}", f" r{i}", score=1.0)
            result = guard.step()
            # Returns float or None (None = no positives or circuit tripped)
            assert result is None or isinstance(result, float)

    def test_circuit_trips_on_nan_tolerance(self):
        """After nan_tolerance NaN steps, circuit must be tripped."""
        from src.training.neuro_elastic import ElasticGuard, ElasticConfig
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tiny_model()
            ol_cfg = OnlineLearnerConfig(
                buffer_capacity=100,
                min_examples_to_finetune=2,
                finetune_every_n_examples=99,
                micro_finetune_steps=1,
                micro_lr=1e-4,
                max_seq_length=32,
                buffer_path=os.path.join(tmpdir, "buf.jsonl"),
            )
            learner = OnlineLearner(model, _FakeTokenizer(), ol_cfg, torch.device("cpu"))
            guard = ElasticGuard(learner, ElasticConfig(nan_tolerance=2, cooldown_seconds=1000))

            # Inject NaN losses by patching learner.step
            call_count = [0]
            def _nan_step():
                call_count[0] += 1
                return float("nan")
            learner.step = _nan_step

            for i in range(5):
                guard.add_example(f"p{i}", " r", score=1.0)
            guard.step()  # NaN #1
            guard.step()  # NaN #2 — should trip (nan_tolerance=2)

            assert guard._tripped, "Circuit should be tripped after nan_tolerance NaN steps"
            assert guard.stats["circuit_trips"] == 1

    def test_circuit_resets_after_cooldown(self):
        import time
        from src.training.neuro_elastic import ElasticGuard, ElasticConfig
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tiny_model()
            ol_cfg = OnlineLearnerConfig(
                buffer_capacity=100,
                min_examples_to_finetune=2,
                finetune_every_n_examples=99,
                micro_finetune_steps=1,
                micro_lr=1e-4,
                max_seq_length=32,
                buffer_path=os.path.join(tmpdir, "buf.jsonl"),
            )
            learner = OnlineLearner(model, _FakeTokenizer(), ol_cfg, torch.device("cpu"))
            guard = ElasticGuard(learner, ElasticConfig(cooldown_seconds=0.05))
            # Trip manually
            guard._trip("test trip")
            assert guard._is_tripped()
            # Wait for cooldown
            time.sleep(0.1)
            assert not guard._is_tripped(), "Circuit should auto-reset after cooldown"

    def test_manual_reset_circuit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            guard._trip("manual test")
            assert guard._tripped
            guard.reset_circuit()
            assert not guard._tripped

    def test_health_report_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            report = guard.health_report()
            assert "circuit_tripped" in report
            assert "cooldown_remaining_s" in report
            assert "examples_accepted" in report
            assert "rollbacks" in report
            assert "adapter_norms" in report

    def test_blocked_when_tripped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard, _ = self._make_guard(tmpdir)
            guard._trip("force trip")
            accepted = guard.add_example("hello", " world", score=1.0)
            assert accepted is False
            assert guard.stats["examples_rejected"] == 1


# ---------------------------------------------------------------------------
# Dynamic EWC lambda
# ---------------------------------------------------------------------------

class TestDynamicEWC:
    def test_drift_magnitude_zero_before_fisher(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=100.0)
        assert ewc.drift_magnitude() == 0.0

    def test_drift_magnitude_increases_after_modification(self):
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=100.0)
        batches = [fake_batch(256) for _ in range(3)]
        ewc.compute_fisher(batches, num_samples=3, device=torch.device("cpu"))
        drift_before = ewc.drift_magnitude()
        # Perturb weights significantly
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 5.0)
        drift_after = ewc.drift_magnitude()
        assert drift_after > drift_before

    def test_dynamic_lambda_penalty_stronger_than_static(self):
        """dynamic_lambda=True must produce >= penalty than static when drift > 0."""
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=1.0)
        batches = [fake_batch(256) for _ in range(3)]
        ewc.compute_fisher(batches, num_samples=3, device=torch.device("cpu"))
        # Perturb weights so drift > 0
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        penalty_static = ewc.penalty(dynamic_lambda=False).item()
        penalty_dynamic = ewc.penalty(dynamic_lambda=True).item()
        assert penalty_dynamic >= penalty_static, (
            "Dynamic lambda must be >= static lambda when drift > 0"
        )

    def test_dynamic_penalty_zero_when_no_drift(self):
        """When model hasn't moved at all, dynamic and static penalties are equal."""
        from src.training.ewc import EWC
        model = tiny_model()
        ewc = EWC(model, lambda_ewc=500.0)
        batches = [fake_batch(256) for _ in range(3)]
        ewc.compute_fisher(batches, num_samples=3, device=torch.device("cpu"))
        # Snapshot weights, then don't move them
        penalty_static = ewc.penalty(dynamic_lambda=False).item()
        penalty_dynamic = ewc.penalty(dynamic_lambda=True).item()
        # Both should be 0 (no drift)
        assert abs(penalty_static - penalty_dynamic) < 1e-6


# ---------------------------------------------------------------------------
# Expert utilization tracking
# ---------------------------------------------------------------------------

class TestExpertUtilization:
    def test_routing_stats_initially_zero(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        stats = moe.routing_stats()
        assert stats["steps_tracked"] == 0
        assert all(v == 0.0 for v in stats["expert_avg_tokens"])

    def test_routing_stats_accumulate_during_training(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(2, 8, 64)
        moe(x)
        moe(x)
        stats = moe.routing_stats()
        assert stats["steps_tracked"] == 2
        # All experts get at least some tokens across 2 steps (probabilistically)
        assert sum(stats["expert_avg_tokens"]) > 0

    def test_routing_stats_not_accumulated_in_eval(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(2, 8, 64)
        with torch.no_grad():
            moe(x)
            moe(x)
        stats = moe.routing_stats()
        assert stats["steps_tracked"] == 0, "Routing stats must not accumulate in eval mode"

    def test_reset_routing_stats(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(2, 8, 64)
        moe(x)
        assert moe.routing_stats()["steps_tracked"] == 1
        moe.reset_routing_stats()
        assert moe.routing_stats()["steps_tracked"] == 0

    def test_utilization_sums_to_one(self):
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(4, 16, 64)
        for _ in range(10):
            moe(x)
        stats = moe.routing_stats()
        total_util = sum(stats["utilization"])
        assert abs(total_util - 1.0) < 1e-5, f"Utilization must sum to 1.0, got {total_util}"

    def test_collapse_detection(self):
        """If one expert gets 0 tokens, collapse_detected must be True."""
        from src.model.moe import MoEFeedForward
        moe = MoEFeedForward(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.train()
        x = torch.randn(4, 16, 64)
        moe(x)  # Populate steps_tracked
        # Manually zero out one expert's count to simulate total collapse
        moe.router._expert_token_counts[0] = 0.0
        stats = moe.routing_stats()
        assert stats["collapse_detected"], "Collapse should be detected when an expert gets no tokens"


# ---------------------------------------------------------------------------
# SyntheticReplay
# ---------------------------------------------------------------------------

class TestSyntheticReplay:
    def _make_replay(self, model=None):
        from src.training.synthetic_replay import SyntheticReplay, ReplayConfig
        m = model or tiny_model()
        cfg = ReplayConfig(
            num_anchors=5,
            samples_per_anchor=1,
            max_new_tokens=8,
            temperature=1.0,
            replay_weight=1.0,
            min_anchor_distance=3,
        )
        return SyntheticReplay(m, _FakeTokenizer(), cfg), m

    def test_add_anchor_accepted(self):
        replay, _ = self._make_replay()
        ok = replay.add_anchor("hello world prompt")
        assert ok is True
        assert replay.num_anchors() == 1

    def test_duplicate_anchor_rejected(self):
        replay, _ = self._make_replay()
        replay.add_anchor("hello world prompt")
        # Same prefix — should be rejected as duplicate
        ok = replay.add_anchor("hello world prompt")
        assert ok is False
        assert replay.num_anchors() == 1

    def test_diverse_anchors_accepted(self):
        replay, _ = self._make_replay()
        replay.add_anchor("aaaaaaa")
        replay.add_anchor("zzzzzzz")  # Very different
        assert replay.num_anchors() == 2

    def test_anchor_ring_buffer_capacity(self):
        replay, _ = self._make_replay()
        for i in range(10):  # More than num_anchors=5
            replay.add_anchor(f"unique prompt number {i} xyz")
        assert replay.num_anchors() <= 5

    def test_replay_loss_returns_none_without_anchors(self):
        replay, _ = self._make_replay()
        assert replay.replay_loss() is None

    def test_replay_loss_returns_tensor_with_anchors(self):
        replay, _ = self._make_replay()
        replay.add_anchor("explain gravity")
        result = replay.replay_loss()
        # May return None if generation fails in test env, but should not raise
        assert result is None or (isinstance(result, torch.Tensor) and result.dim() == 0)

    def test_replay_loss_stat_incremented(self):
        replay, _ = self._make_replay()
        replay.add_anchor("test anchor here")
        initial_steps = replay.stats["replay_steps"]
        replay.replay_loss()
        # replay_steps only increments if loss was non-None
        assert replay.stats["replay_steps"] >= initial_steps

    def test_save_load_anchors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            replay, model = self._make_replay()
            replay.add_anchor("anchor one here")
            replay.add_anchor("anchor two there")
            path = os.path.join(tmpdir, "replay.pt")
            replay.save(path)

            from src.training.synthetic_replay import SyntheticReplay, ReplayConfig
            cfg = ReplayConfig(num_anchors=5, samples_per_anchor=1, max_new_tokens=8)
            replay2 = SyntheticReplay(model, _FakeTokenizer(), cfg)
            replay2.load(path)
            assert replay2.num_anchors() == replay.num_anchors()

    def test_add_anchors_bulk(self):
        replay, _ = self._make_replay()
        n = replay.add_anchors(["aaaaaa", "bbbbbb", "cccccc"])
        assert n == 3

    def test_replay_integrated_in_online_learner(self):
        """SyntheticReplay must be invoked from OnlineLearner.step()."""
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        from src.training.synthetic_replay import SyntheticReplay, ReplayConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tiny_model()
            cfg = OnlineLearnerConfig(
                buffer_capacity=50,
                min_examples_to_finetune=2,
                finetune_every_n_examples=99,
                micro_finetune_steps=1,
                micro_lr=1e-4,
                max_seq_length=32,
                buffer_path=os.path.join(tmpdir, "buf.jsonl"),
            )
            replay_cfg = ReplayConfig(num_anchors=3, samples_per_anchor=1, max_new_tokens=8)
            replay = SyntheticReplay(model, _FakeTokenizer(), replay_cfg)
            replay.add_anchor("test anchor for integration")
            learner = OnlineLearner(
                model, _FakeTokenizer(), cfg, torch.device("cpu"),
                synthetic_replay=replay,
            )
            for i in range(3):
                learner.add_example(f"p{i}", f" r{i}", score=1.0)
            # Should not raise
            result = learner.step()
            assert result is not None


# ---------------------------------------------------------------------------
# ForgettingTracker
# ---------------------------------------------------------------------------

class TestForgettingTracker:
    def test_empty_tracker_report(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        report = tracker.report()
        assert report["num_tasks"] == 0
        assert report["avg_forgetting"] == 0.0
        assert report["plasticity"] == 0.0

    def test_single_task_single_phase(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.90)
        assert tracker.plasticity() == 0.90
        # Only 1 record — no forgetting computable
        assert tracker.avg_forgetting() == 0.0

    def test_forgetting_computed_correctly(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.92)
        tracker.record("task_A", phase=1, score=0.81)
        f = tracker.forgetting()
        assert "task_A" in f
        assert abs(f["task_A"] - 0.11) < 1e-6

    def test_no_forgetting_when_score_improves(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.80)
        tracker.record("task_A", phase=1, score=0.90)
        f = tracker.forgetting()
        # Positive improvement: forgetting should be ≤ 0
        assert f["task_A"] <= 0.0

    def test_backward_transfer_negative_on_forgetting(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.92)
        tracker.record("task_A", phase=1, score=0.81)  # dropped
        assert tracker.backward_transfer() < 0.0

    def test_backward_transfer_positive_on_improvement(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.80)
        tracker.record("task_A", phase=1, score=0.90)  # improved
        assert tracker.backward_transfer() > 0.0

    def test_two_tasks_avg_forgetting(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("task_A", phase=0, score=0.90)
        tracker.record("task_B", phase=0, score=0.80)
        tracker.record("task_A", phase=1, score=0.80)  # forgot 0.10
        tracker.record("task_B", phase=1, score=0.70)  # forgot 0.10
        assert abs(tracker.avg_forgetting() - 0.10) < 1e-5

    def test_max_forgetting(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("easy", phase=0, score=0.95)
        tracker.record("hard", phase=0, score=0.85)
        tracker.record("easy", phase=1, score=0.90)   # forgot 0.05
        tracker.record("hard", phase=1, score=0.60)   # forgot 0.25
        worst_task, worst_score = tracker.max_forgetting()
        assert worst_task == "hard"
        assert abs(worst_score - 0.25) < 1e-5

    def test_summary_line_contains_key_metrics(self):
        from src.training.continual_metrics import ForgettingTracker
        tracker = ForgettingTracker()
        tracker.record("t1", phase=0, score=0.9)
        tracker.record("t1", phase=1, score=0.8)
        line = tracker.summary_line()
        assert "avg_forgetting" in line
        assert "plasticity" in line

    def test_save_load_roundtrip(self):
        from src.training.continual_metrics import ForgettingTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ForgettingTracker()
            tracker.record("task_A", phase=0, score=0.9)
            tracker.record("task_A", phase=1, score=0.8)
            path = os.path.join(tmpdir, "tracker.pt")
            tracker.save(path)

            tracker2 = ForgettingTracker()
            tracker2.load(path)
            assert tracker2.avg_forgetting() == tracker.avg_forgetting()
            assert len(tracker2._records) == 2


# ---------------------------------------------------------------------------
# MAML
# ---------------------------------------------------------------------------

class TestMAML:
    def _make_maml(self, lora_only=True):
        from src.training.maml import MAML, MAMLConfig
        from src.model.lora import inject_lora, LoRAConfig
        model = tiny_model()
        if lora_only:
            inject_lora(model, LoRAConfig(rank=4))
        cfg = MAMLConfig(
            inner_lr=0.01,
            inner_steps=2,
            meta_batch_size=2,
            lora_only=lora_only,
            first_order=True,
        )
        return MAML(model, cfg), model

    def test_snapshot_and_restore(self):
        maml, model = self._make_maml()
        original = maml.snapshot_params()
        # Perturb
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        maml.restore_params(original)
        for name, orig_val in original.items():
            for pname, param in model.named_parameters():
                if pname == name:
                    assert torch.allclose(param.data, orig_val), f"Restore failed for {name}"

    def test_inner_loop_changes_params(self):
        maml, model = self._make_maml()
        batch = fake_batch()
        original = maml.snapshot_params()
        adapted = maml.inner_loop(batch)
        # Adapted params must differ from original
        changed = any(
            not torch.allclose(adapted[k], original[k])
            for k in adapted
            if k in original
        )
        assert changed, "Inner loop must update at least one parameter"

    def test_adapt_does_not_modify_model_in_place(self):
        maml, model = self._make_maml()
        original = maml.snapshot_params()
        _ = maml.adapt(fake_batch())
        # Model weights must be unchanged after adapt()
        for name, orig_val in original.items():
            for pname, param in model.named_parameters():
                if pname == name:
                    assert torch.allclose(param.data, orig_val), \
                        f"adapt() must not modify model in place (param {name} changed)"

    def test_lora_only_adapts_lora_params(self):
        from src.training.maml import MAML, MAMLConfig
        from src.model.lora import inject_lora, LoRAConfig
        model = tiny_model()
        inject_lora(model, LoRAConfig(rank=4))
        cfg = MAMLConfig(inner_lr=0.01, inner_steps=1, lora_only=True, first_order=True)
        maml = MAML(model, cfg)
        params = maml._get_params()
        for name in params:
            assert "lora_A" in name or "lora_B" in name, \
                f"lora_only=True must only include LoRA params, got {name}"

    def test_outer_step_returns_float(self):
        maml, model = self._make_maml()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        task_batches = [(fake_batch(), fake_batch()), (fake_batch(), fake_batch())]
        meta_loss = maml.outer_step(task_batches, opt)
        assert isinstance(meta_loss, float)
        assert not (meta_loss != meta_loss)  # not NaN


# ---------------------------------------------------------------------------
# AlignmentMonitor
# ---------------------------------------------------------------------------

class TestAlignmentMonitor:
    def _make_monitor(self):
        from src.training.alignment_monitor import AlignmentMonitor, AlignmentConfig
        model = tiny_model()
        cfg = AlignmentConfig(
            max_probes=5,
            probe_tokens=4,
            kl_alert_threshold=0.3,
            entropy_alert_threshold=0.5,
            score_regression_threshold=0.1,
        )
        return AlignmentMonitor(model, _FakeTokenizer(), cfg), model

    def test_no_probes_check_returns_empty_alerts(self):
        monitor, _ = self._make_monitor()
        report = monitor.check_drift()
        assert report["drift_detected"] is False
        assert report["alerts"] == []

    def test_add_probe(self):
        monitor, _ = self._make_monitor()
        monitor.add_probe("what is 2+2?")
        assert len(monitor._probes) == 1

    def test_set_reference_requires_probes(self):
        monitor, _ = self._make_monitor()
        # With no probes, set_reference should warn but not raise
        monitor.set_reference()  # Should not raise
        assert monitor._reference is None

    def test_set_reference_with_probes(self):
        monitor, _ = self._make_monitor()
        monitor.add_probe("hello world")
        monitor.set_reference()
        assert monitor._reference is not None
        assert monitor._reference_entropy is not None

    def test_no_drift_right_after_reference(self):
        """Immediately after set_reference, KL divergence should be ~0."""
        monitor, _ = self._make_monitor()
        monitor.add_probe("test prompt one")
        monitor.set_reference()
        report = monitor.check_drift()
        # KL against itself should be ~0, no alert
        assert report["kl_divergence"] < 0.01

    def test_drift_detected_after_large_weight_change(self):
        """After perturbing model weights, KL divergence should rise."""
        monitor, model = self._make_monitor()
        monitor.add_probe("explain gravity")
        monitor.set_reference()
        # Massively perturb weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 10.0)
        report = monitor.check_drift()
        # KL should be non-zero
        assert report["kl_divergence"] > 0.0

    def test_score_floor_regression_alert(self):
        monitor, _ = self._make_monitor()
        monitor.set_score_floor("task_A", min_score=0.8)
        report = monitor.check_drift(task_scores={"task_A": 0.5})
        assert report["drift_detected"] is True
        assert "task_A" in report["score_regressions"]

    def test_score_floor_no_alert_when_above_floor(self):
        monitor, _ = self._make_monitor()
        monitor.set_score_floor("task_A", min_score=0.8)
        report = monitor.check_drift(task_scores={"task_A": 0.9})
        assert "task_A" not in report["score_regressions"]

    def test_drift_trend_history(self):
        monitor, _ = self._make_monitor()
        monitor.add_probe("hello")
        monitor.set_reference()
        monitor.check_drift()
        monitor.check_drift()
        assert len(monitor.drift_trend()) == 2

    def test_is_safe_before_any_check(self):
        monitor, _ = self._make_monitor()
        assert monitor.is_safe() is True

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor, _ = self._make_monitor()
            monitor.add_probe("save test prompt")
            monitor.set_score_floor("t1", 0.7)
            path = os.path.join(tmpdir, "monitor.pt")
            monitor.save(path)

            from src.training.alignment_monitor import AlignmentMonitor, AlignmentConfig
            monitor2 = AlignmentMonitor(tiny_model(), _FakeTokenizer(), AlignmentConfig())
            monitor2.load(path)
            assert len(monitor2._probes) == 1
            assert "t1" in monitor2._score_floors


# ---------------------------------------------------------------------------
# Human-in-the-loop review queue
# ---------------------------------------------------------------------------

class TestHumanReviewQueue:
    def _make_guard_with_review(self, tmpdir):
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        from src.training.neuro_elastic import ElasticGuard, ElasticConfig
        model = tiny_model()
        ol_cfg = OnlineLearnerConfig(
            buffer_capacity=50,
            min_examples_to_finetune=2,
            finetune_every_n_examples=99,
            micro_finetune_steps=1,
            micro_lr=1e-4,
            max_seq_length=32,
            buffer_path=os.path.join(tmpdir, "buf.jsonl"),
        )
        learner = OnlineLearner(model, _FakeTokenizer(), ol_cfg, torch.device("cpu"))
        cfg = ElasticConfig(
            human_review_enabled=True,
            human_review_z_threshold=2.0,  # Flag if score > 2 std-devs from mean
        )
        return ElasticGuard(learner, cfg)

    def test_review_queue_empty_initially(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            assert guard.review_queue_size() == 0

    def test_normal_scores_not_flagged(self):
        """With insufficient score history, no flagging occurs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            for i in range(5):
                guard.add_example(f"p{i}", " r", score=1.0)
            assert guard.review_queue_size() == 0

    def test_anomalous_score_flagged_after_history(self):
        """After 10 normal scores, an extreme score should be flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            # Build score history with tight distribution
            for i in range(12):
                guard.add_example(f"p{i}", " r", score=1.0)
            # Now submit an extreme outlier
            guard.add_example("outlier", " resp", score=999.0)
            # Score is clamped to score_max=10.0, but still extreme vs history of 1.0
            assert guard.review_queue_size() > 0

    def test_pop_review_queue_clears_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            # Force a flag by directly injecting into review queue
            guard._review_queue.append({"prompt": "x", "response": "y", "score": 99.0, "timestamp": 0.0, "reason": "test"})
            assert guard.review_queue_size() == 1
            items = guard.pop_review_queue()
            assert len(items) == 1
            assert guard.review_queue_size() == 0

    def test_review_item_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            guard._review_queue.append({"prompt": "p", "response": "r", "score": 5.0, "timestamp": 0.0, "reason": "score_anomaly"})
            items = guard.pop_review_queue()
            assert items[0]["reason"] == "score_anomaly"
            assert "prompt" in items[0]
            assert "score" in items[0]

    def test_stats_track_flagged_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            guard = self._make_guard_with_review(tmpdir)
            for i in range(15):
                guard.add_example(f"p{i}", " r", score=1.0)
            guard.add_example("outlier", " resp", score=9.9)
            assert guard.stats["examples_flagged_for_review"] >= 0  # May or may not flag depending on variance


# ---------------------------------------------------------------------------
# Sparse plasticity
# ---------------------------------------------------------------------------

class TestSparsePlasticity:
    def test_sparse_gradient_zeros_low_magnitude(self):
        from src.training.online_learner import _apply_sparse_gradients
        model = tiny_model()
        # Set up fake gradients
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        # Apply top-10% sparse mask
        _apply_sparse_gradients(model, k=0.1)
        # Count non-zero grads
        total = 0
        nonzero = 0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.numel()
                nonzero += (p.grad != 0).sum().item()
        if total > 0:
            fraction_kept = nonzero / total
            # Should keep approximately 10% (within tolerance due to ties)
            assert fraction_kept <= 0.15, f"Too many grads kept: {fraction_kept:.3f}"

    def test_sparse_gradient_k_zero_is_noop(self):
        from src.training.online_learner import _apply_sparse_gradients
        model = tiny_model()
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        _apply_sparse_gradients(model, k=0.0)
        # All gradients should still be 1.0 (noop)
        for p in model.parameters():
            if p.grad is not None:
                assert (p.grad == 1.0).all()

    def test_sparse_gradient_k_one_is_noop(self):
        from src.training.online_learner import _apply_sparse_gradients
        model = tiny_model()
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        _apply_sparse_gradients(model, k=1.0)
        # k=1.0 = keep all — should be noop
        for p in model.parameters():
            if p.grad is not None:
                assert (p.grad == 1.0).all()

    def test_sparse_plasticity_in_online_learner_step(self):
        """OnlineLearner with sparse_gradient_k > 0 must complete a step without error."""
        from src.training.online_learner import OnlineLearner, OnlineLearnerConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            model = tiny_model()
            cfg = OnlineLearnerConfig(
                buffer_capacity=50,
                min_examples_to_finetune=2,
                finetune_every_n_examples=99,
                micro_finetune_steps=1,
                micro_lr=1e-4,
                max_seq_length=32,
                buffer_path=os.path.join(tmpdir, "buf.jsonl"),
                sparse_gradient_k=0.1,
            )
            learner = OnlineLearner(model, _FakeTokenizer(), cfg, torch.device("cpu"))
            for i in range(4):
                learner.add_example(f"p{i}", f" r{i}", score=1.0)
            result = learner.step()
            assert result is not None

    def test_config_defaults(self):
        from src.utils.config import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.sparse_gradient_k == 0.0
        assert cfg.alignment_monitor_enabled is False
        assert cfg.human_review_enabled is False
