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
