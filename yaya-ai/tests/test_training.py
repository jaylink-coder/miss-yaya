"""Tests for training infrastructure components."""

import pytest
import torch
import tempfile
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ModelConfig
from src.model.yaya_model import YayaForCausalLM
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.checkpointing import CheckpointManager
from src.training.loss import CausalLMLoss
from src.training.distributed import is_main_process, get_deepspeed_config
from src.utils.config import TrainingConfig


def get_test_config() -> ModelConfig:
    return ModelConfig(
        model_name="yaya-test",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )


class TestOptimizer:
    def test_create_optimizer(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)
        assert optimizer is not None
        assert len(optimizer.param_groups) == 2  # decay + no_decay

    def test_weight_decay_groups(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
        # Group 0: decay, Group 1: no_decay
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_optimizer_step(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)

        input_ids = torch.randint(0, 256, (2, 16))
        labels = torch.randint(0, 256, (2, 16))
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()

        # Capture params before step
        params_before = {n: p.clone() for n, p in model.named_parameters() if p.grad is not None}

        optimizer.step()

        # Verify params changed
        changed = False
        for n, p in model.named_parameters():
            if n in params_before and not torch.equal(p.data, params_before[n]):
                changed = True
                break
        assert changed, "Optimizer step should change at least some parameters"


class TestScheduler:
    def test_warmup(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(optimizer, warmup_steps=100, max_steps=1000)

        # At step 0, LR should be ~0
        lr_0 = scheduler.get_last_lr()[0]
        assert lr_0 < 1e-4

        # Simulate warmup
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        lr_50 = scheduler.get_last_lr()[0]

        for _ in range(50):
            optimizer.step()
            scheduler.step()
        lr_100 = scheduler.get_last_lr()[0]

        # LR should increase during warmup
        assert lr_50 > lr_0
        assert lr_100 > lr_50

    def test_cosine_decay(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer, warmup_steps=10, max_steps=100, schedule_type="cosine"
        )

        # Complete warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        lr_peak = scheduler.get_last_lr()[0]

        # Run to midpoint
        for _ in range(45):
            optimizer.step()
            scheduler.step()
        lr_mid = scheduler.get_last_lr()[0]

        # LR should decay after warmup
        assert lr_mid < lr_peak

    def test_constant_schedule(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)
        scheduler = create_scheduler(
            optimizer, warmup_steps=10, max_steps=100, schedule_type="constant"
        )

        # After warmup, LR should stay constant
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        lr_after_warmup = scheduler.get_last_lr()[0]

        for _ in range(40):
            optimizer.step()
            scheduler.step()
        lr_later = scheduler.get_last_lr()[0]

        assert abs(lr_after_warmup - lr_later) < 1e-6


class TestCheckpointManager:
    def test_save_and_load(self):
        config = get_test_config()
        model = YayaForCausalLM(config)
        optimizer = create_optimizer(model, learning_rate=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=3)

            # Save checkpoint
            ckpt_path = manager.save(
                model, optimizer, step=100, epoch=1, loss=2.5
            )
            assert os.path.exists(ckpt_path)
            assert os.path.exists(os.path.join(ckpt_path, "model.pt"))
            assert os.path.exists(os.path.join(ckpt_path, "optimizer.pt"))
            assert os.path.exists(os.path.join(ckpt_path, "metadata.json"))

            # Load into a fresh model
            model2 = YayaForCausalLM(config)
            metadata = manager.load(model2, checkpoint_path=ckpt_path)

            assert metadata["step"] == 100
            assert metadata["epoch"] == 1
            assert metadata["loss"] == 2.5

            # Weights should match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(), model2.named_parameters()
            ):
                assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_latest_checkpoint(self):
        config = get_test_config()
        model = YayaForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=5)
            manager.save(model, step=100, epoch=1, loss=3.0)
            manager.save(model, step=200, epoch=2, loss=2.5)
            manager.save(model, step=300, epoch=3, loss=2.0)

            latest = manager.get_latest_checkpoint()
            assert "00000300" in latest

    def test_cleanup_old_checkpoints(self):
        config = get_test_config()
        model = YayaForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir, keep_last_n=2)
            manager.save(model, step=100, epoch=1, loss=3.0)
            manager.save(model, step=200, epoch=2, loss=2.5)
            manager.save(model, step=300, epoch=3, loss=2.0)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 2  # Only last 2 kept

    def test_resume_no_checkpoint(self):
        config = get_test_config()
        model = YayaForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(save_dir=tmpdir)
            metadata = manager.load(model)
            assert metadata["step"] == 0  # Fresh start


class TestCausalLMLoss:
    def test_basic_loss(self):
        loss_fn = CausalLMLoss(vocab_size=256)
        logits = torch.randn(2, 16, 256)
        labels = torch.randint(0, 256, (2, 16))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_ignore_padding(self):
        loss_fn = CausalLMLoss(vocab_size=256)
        logits = torch.randn(2, 16, 256)
        # Partially padded labels — padding positions should not contribute
        labels_partial = torch.randint(0, 256, (2, 16))
        labels_partial[:, 8:] = -100  # Mask second half
        loss_partial = loss_fn(logits, labels_partial)

        labels_full = labels_partial.clone()
        labels_full[:, 8:] = torch.randint(0, 256, (2, 8))  # Fill second half
        loss_full = loss_fn(logits, labels_full)

        # Losses should differ since they use different numbers of valid tokens
        assert loss_partial.item() != loss_full.item()

    def test_label_smoothing(self):
        loss_fn_no_smooth = CausalLMLoss(vocab_size=256, label_smoothing=0.0)
        loss_fn_smooth = CausalLMLoss(vocab_size=256, label_smoothing=0.1)

        logits = torch.randn(2, 16, 256)
        labels = torch.randint(0, 256, (2, 16))

        loss_no = loss_fn_no_smooth(logits, labels)
        loss_yes = loss_fn_smooth(logits, labels)

        # With smoothing, loss is typically higher on random data
        assert loss_no.item() != loss_yes.item()

    def test_z_loss(self):
        loss_fn_no_z = CausalLMLoss(vocab_size=256, z_loss_weight=0.0)
        loss_fn_z = CausalLMLoss(vocab_size=256, z_loss_weight=1e-4)

        logits = torch.randn(2, 16, 256) * 10  # Large logits
        labels = torch.randint(0, 256, (2, 16))

        loss_no = loss_fn_no_z(logits, labels)
        loss_z = loss_fn_z(logits, labels)

        # Z-loss adds penalty for large logits
        assert loss_z.item() > loss_no.item()


class TestDistributedUtils:
    def test_is_main_process(self):
        # Without distributed init, should return True
        assert is_main_process() is True

    def test_deepspeed_config_generation(self):
        train_config = TrainingConfig(
            per_device_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=3e-4,
            max_steps=100000,
            warmup_steps=2000,
            max_grad_norm=1.0,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            lr_scheduler="cosine",
            dtype="bfloat16",
        )
        ds_config = get_deepspeed_config(train_config, zero_stage=2)
        assert ds_config["train_micro_batch_size_per_gpu"] == 4
        assert ds_config["gradient_accumulation_steps"] == 8
        assert ds_config["bf16"]["enabled"] is True
        assert ds_config["zero_optimization"]["stage"] == 2

    def test_deepspeed_config_zero3(self):
        train_config = TrainingConfig(
            per_device_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            dtype="bfloat16",
        )
        ds_config = get_deepspeed_config(train_config, zero_stage=3)
        assert ds_config["zero_optimization"]["stage"] == 3
        assert "stage3_max_live_parameters" in ds_config["zero_optimization"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
