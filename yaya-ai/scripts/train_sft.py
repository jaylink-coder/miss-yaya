"""Supervised Fine-Tuning (SFT) entry point for Yaya.

Loads a pretrained checkpoint and fine-tunes on instruction data.

Usage:
    python scripts/train_sft.py \
        --model_config configs/model/yaya_125m.yaml \
        --train_config configs/training/sft_125m.yaml \
        --pretrain_checkpoint checkpoints/yaya-125m/checkpoint-XXXXXXXX
"""

import argparse
import sys
import os

# Ensure UTF-8 output on Windows (box-drawing chars in model summary)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config, load_training_config
from src.model.yaya_model import YayaForCausalLM
from src.data.dataset import InstructionDataset
from src.data.dataloader import create_dataloader
from src.tokenizer.tokenizer import YayaTokenizer
from src.training.trainer import Trainer
from src.training.checkpointing import CheckpointManager
from src.training.distributed import is_main_process


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Yaya")
    parser.add_argument("--model_config",        type=str, required=True)
    parser.add_argument("--train_config",        type=str, required=True)
    parser.add_argument("--pretrain_checkpoint", type=str, default=None,
                        help="Pretrained checkpoint to start from")
    parser.add_argument("--resume",              nargs="?", const=True, default=None,
                        help="Resume from checkpoint. Pass a path to resume from a specific checkpoint, or pass without a value to auto-resume from latest.")
    parser.add_argument("--compute_ewc_fisher",  action="store_true",
                        help="After loading checkpoint, compute EWC Fisher before training "
                             "(use for continual learning Phase 2)")
    # ── CLI overrides (take priority over YAML config) ───────────────────────
    parser.add_argument("--data_file",                  type=str,   default=None,
                        help="Override train_config.train_data (and eval_data)")
    parser.add_argument("--output_dir",                 type=str,   default=None,
                        help="Override train_config.save_dir")
    parser.add_argument("--max_steps",                  type=int,   default=None)
    parser.add_argument("--learning_rate",              type=float, default=None)
    parser.add_argument("--per_device_batch_size",      type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps",type=int,   default=None)
    parser.add_argument("--max_seq_length",             type=int,   default=None)
    parser.add_argument("--save_steps",                 type=int,   default=None)
    parser.add_argument("--warmup_steps",               type=int,   default=None)
    parser.add_argument("--lr_scheduler",               type=str,   default=None)
    parser.add_argument("--weight_decay",               type=float, default=None)
    parser.add_argument("--max_grad_norm",              type=float, default=None)
    parser.add_argument("--dataloader_num_workers",     type=int,   default=None)
    parser.add_argument("--fp16",  action="store_true", help="Use float16 mixed precision")
    parser.add_argument("--bf16",  action="store_true", help="Use bfloat16 mixed precision")
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    train_config = load_training_config(args.train_config)

    # Apply CLI overrides
    if args.data_file:
        train_config.train_data = args.data_file
        train_config.eval_data  = args.data_file
    if args.output_dir:
        train_config.save_dir = args.output_dir
    if args.max_steps                  is not None: train_config.max_steps                   = args.max_steps
    if args.learning_rate              is not None: train_config.learning_rate               = args.learning_rate
    if args.per_device_batch_size      is not None: train_config.per_device_batch_size       = args.per_device_batch_size
    if args.gradient_accumulation_steps is not None: train_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_seq_length             is not None: train_config.max_seq_length              = args.max_seq_length
    if args.save_steps                 is not None: train_config.save_steps                  = args.save_steps
    if args.warmup_steps               is not None: train_config.warmup_steps                = args.warmup_steps
    if args.lr_scheduler               is not None: train_config.lr_scheduler                = args.lr_scheduler
    if args.weight_decay               is not None: train_config.weight_decay                = args.weight_decay
    if args.max_grad_norm              is not None: train_config.max_grad_norm               = args.max_grad_norm
    if args.dataloader_num_workers     is not None: train_config.num_workers                 = args.dataloader_num_workers
    if args.bf16:  train_config.dtype = "bfloat16"
    if args.fp16:  train_config.dtype = "float16"

    torch.manual_seed(train_config.seed)

    # Build model
    model = YayaForCausalLM(model_config)

    # Load pretrained weights if provided
    if args.pretrain_checkpoint and not args.resume:
        ckpt = args.pretrain_checkpoint
        if is_main_process():
            print(f"Loading pretrained weights from: {ckpt}")
        if os.path.isfile(ckpt):
            # Direct file path (model.pt) — load state dict straight from file
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if is_main_process():
                print(f"  Weights loaded OK. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        elif os.path.isdir(ckpt):
            # Directory path — use CheckpointManager
            ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(ckpt))
            ckpt_mgr.load(model, checkpoint_path=ckpt)
        else:
            print(f"  ERROR: pretrain_checkpoint not found: {ckpt!r} — training from random init!")

    if is_main_process():
        print(model.generate_summary())

    # Load tokenizer
    tokenizer = YayaTokenizer(train_config.tokenizer_path)

    # Build instruction datasets
    if is_main_process():
        print(f"Loading instruction data from: {train_config.train_data}")

    train_dataset = InstructionDataset(
        data_path=train_config.train_data,
        tokenizer=tokenizer,
        max_seq_length=train_config.max_seq_length,
    )

    eval_dataset = InstructionDataset(
        data_path=train_config.eval_data,
        tokenizer=tokenizer,
        max_seq_length=train_config.max_seq_length,
    ) if train_config.eval_data else None

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=train_config.per_device_batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
    )

    eval_dataloader = create_dataloader(
        eval_dataset,
        batch_size=train_config.per_device_batch_size,
        num_workers=train_config.num_workers,
        shuffle=False,
    ) if eval_dataset else None

    trainer = Trainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        tokenizer=tokenizer,
    )

    # Continual learning: anchor current weights before Phase 2 fine-tuning
    if args.compute_ewc_fisher:
        if trainer.ewc is None:
            print("WARNING: --compute_ewc_fisher passed but ewc_lambda=0 in config. Skipping.")
        else:
            if is_main_process():
                print("Computing EWC Fisher information matrix on training data...")
            trainer.compute_ewc_fisher()

    resume_from = None if args.resume is True else args.resume
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
