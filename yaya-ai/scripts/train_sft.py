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
    parser.add_argument("--resume",              type=str, default=None,
                        help="SFT checkpoint to resume from")
    parser.add_argument("--compute_ewc_fisher",  action="store_true",
                        help="After loading checkpoint, compute EWC Fisher before training "
                             "(use for continual learning Phase 2)")
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    train_config = load_training_config(args.train_config)

    torch.manual_seed(train_config.seed)

    # Build model
    model = YayaForCausalLM(model_config)

    # Load pretrained weights if provided
    if args.pretrain_checkpoint and args.resume is None:
        if is_main_process():
            print(f"Loading pretrained weights from: {args.pretrain_checkpoint}")
        ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(args.pretrain_checkpoint))
        ckpt_mgr.load(model, checkpoint_path=args.pretrain_checkpoint)

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
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
