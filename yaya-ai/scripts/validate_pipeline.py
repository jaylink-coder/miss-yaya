"""End-to-end pipeline validation for Yaya AI.

Runs the full pipeline on synthetic data:
1. Generate synthetic corpus
2. Train tokenizer
3. Tokenize data into shards
4. Train tiny model
5. Run generation

Usage:
    python scripts/validate_pipeline.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "test")


def step(name):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")


def main():
    start_time = time.time()

    # ── Step 1: Generate synthetic data ──────────────────────
    step("1/5 — Generate synthetic training corpus")
    from scripts.generate_test_data import generate_corpus

    corpus_path = generate_corpus(TEST_DATA_DIR, num_docs=500, seed=42)

    # ── Step 2: Train tokenizer ──────────────────────────────
    step("2/5 — Train BPE tokenizer")
    from src.tokenizer.trainer import TokenizerTrainer

    tokenizer_dir = os.path.join(TEST_DATA_DIR, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    trainer = TokenizerTrainer(
        output_dir=tokenizer_dir,
        vocab_size=4096,
        model_prefix="yaya_tokenizer",
    )
    trainer.train(input_files=[corpus_path])

    tokenizer_model_path = os.path.join(tokenizer_dir, "yaya_tokenizer.model")
    assert os.path.exists(tokenizer_model_path), "Tokenizer model not created!"

    # Verify tokenizer
    from src.tokenizer.tokenizer import YayaTokenizer
    tokenizer = YayaTokenizer(tokenizer_model_path)
    test_text = "The process of photosynthesis converts sunlight into energy."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"  Test encode: '{test_text[:50]}...'")
    print(f"  Token count: {len(encoded)}")
    print(f"  Decoded: '{decoded[:50]}...'")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Step 3: Tokenize data into binary shards ─────────────
    step("3/5 — Tokenize corpus into training shards")

    train_shard_dir = os.path.join(TEST_DATA_DIR, "processed", "train")
    eval_shard_dir = os.path.join(TEST_DATA_DIR, "processed", "eval")
    os.makedirs(train_shard_dir, exist_ok=True)
    os.makedirs(eval_shard_dir, exist_ok=True)

    # Read and tokenize corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    all_tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(all_tokens):,}")

    # Split 90/10 train/eval
    split_idx = int(len(all_tokens) * 0.9)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    eval_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)

    train_path = os.path.join(train_shard_dir, "train.bin")
    eval_path = os.path.join(eval_shard_dir, "eval.bin")
    train_tokens.tofile(train_path)
    eval_tokens.tofile(eval_path)

    print(f"  Train tokens: {len(train_tokens):,} -> {train_path}")
    print(f"  Eval tokens: {len(eval_tokens):,} -> {eval_path}")

    # ── Step 4: Train tiny model ─────────────────────────────
    step("4/5 — Train tiny model (200 steps)")

    from src.utils.config import load_model_config, ModelConfig
    from src.model.yaya_model import YayaForCausalLM
    from src.data.dataset import TextDataset
    from src.data.dataloader import collate_fn, create_dataloader
    from src.training.optimizer import create_optimizer, create_scheduler
    from src.training.loss import CausalLMLoss
    from src.training.checkpointing import CheckpointManager
    from src.utils.io_utils import count_parameters, format_num

    # Load model config
    model_config = load_model_config(
        os.path.join(PROJECT_ROOT, "configs", "model", "yaya_125m.yaml")
    )
    # Override vocab size to match our tokenizer
    model_config.vocab_size = tokenizer.vocab_size

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YayaForCausalLM(model_config).to(device)
    params = count_parameters(model)
    print(f"  Model: {model_config.model_name}")
    print(f"  Parameters: {format_num(params['total'])} ({params['total']:,})")
    print(f"  Device: {device}")

    # Dataset
    max_seq_length = 128
    train_dataset = TextDataset(
        data_path=train_shard_dir,
        max_seq_length=max_seq_length,
        split="train",
    )
    print(f"  Train samples: {len(train_dataset):,}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # Optimizer & scheduler
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    scheduler = create_scheduler(optimizer, warmup_steps=20, max_steps=200)

    # Checkpoint manager
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "yaya-tiny")
    ckpt_manager = CheckpointManager(save_dir=ckpt_dir, keep_last_n=2)

    # Training loop
    model.train()
    total_loss = 0.0
    log_interval = 20
    max_steps = 200

    data_iter = iter(train_loader)
    for step_num in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if step_num % log_interval == 0:
            avg_loss = total_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step_num:>4}/{max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            total_loss = 0.0

        if step_num % 100 == 0:
            ckpt_manager.save(model, optimizer, step=step_num, loss=loss.item())

    # Save final
    ckpt_manager.save(model, optimizer, step=max_steps, loss=loss.item())
    print(f"\n  Training complete! Final loss: {loss.item():.4f}")

    # ── Step 5: Generate text ────────────────────────────────
    step("5/5 — Generate text from trained model")

    model.eval()
    prompts = [
        "The process of",
        "Machine learning is",
        "The capital of",
        "In mathematics,",
    ]

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=device)

        generated = list(input_ids)
        with torch.no_grad():
            for _ in range(64):
                outputs = model(input_ids=input_tensor)
                logits = outputs["logits"][:, -1, :]
                # Temperature sampling
                logits = logits / 0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_id = next_token.item()
                generated.append(next_id)
                if next_id == tokenizer.eos_id:
                    break
                input_tensor = torch.tensor([[next_id]], device=device)

        text = tokenizer.decode(generated)
        print(f"  Prompt: '{prompt}'")
        print(f"  Output: '{text[:150]}...'")
        print()

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  PIPELINE VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Tokenizer vocab: {tokenizer.vocab_size}")
    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Model params: {format_num(params['total'])}")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"\n  All pipeline stages passed! ✓")


if __name__ == "__main__":
    main()
