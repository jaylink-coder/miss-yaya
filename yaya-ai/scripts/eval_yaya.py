"""Evaluate Yaya with a standard set of prompts.

Runs a fixed test suite and prints responses so you can quickly
judge model quality after pretraining or SFT.

Usage:
    python scripts/eval_yaya.py \
        --model_config configs/model/yaya_125m.yaml \
        --checkpoint checkpoints/yaya-125m/checkpoint-XXXXXXXX
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig
from src.training.checkpointing import CheckpointManager

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
)

# Standard test prompts — covers jokes, facts, reasoning, creativity
TEST_PROMPTS = [
    ("Joke",        "Tell me a joke."),
    ("Greeting",    "Hello! Who are you?"),
    ("Factual",     "What is the capital of France?"),
    ("Science",     "What is photosynthesis?"),
    ("AI",          "What is a neural network?"),
    ("Reasoning",   "If I have 5 apples and eat 2, how many do I have?"),
    ("Creative",    "Write a short poem about the moon."),
    ("Explain",     "Explain what the Internet is in simple terms."),
    ("Identity",    "What can you do?"),
    ("Open",        "The history of artificial intelligence"),
]


def evaluate(model, tokenizer, generator, use_chat_format: bool = False):
    print("\n" + "=" * 70)
    print("  YAYA EVALUATION")
    print("=" * 70)

    for label, prompt in TEST_PROMPTS:
        if use_chat_format:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            full_prompt = tokenizer.format_chat(messages) + "<|assistant|>\n"
        else:
            full_prompt = prompt

        response = generator.generate(
            full_prompt,
            config=GenerationConfig(max_new_tokens=100, temperature=0.8, top_p=0.9),
        )

        # Strip everything up to and including <|assistant|>
        tag = "<|assistant|>"
        if tag in response:
            response = response.split(tag)[-1]
        elif full_prompt in response:
            response = response[len(full_prompt):]

        # Clean up any trailing chat tags
        for end_tag in ["<|user|>", "<|system|>", "</s>"]:
            response = response.split(end_tag)[0]

        response = response.strip()

        print(f"\n[{label}] {prompt}")
        print(f"Yaya: {response}")
        print("-" * 70)

    print("\nEvaluation complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--chat",         action="store_true",
                        help="Use chat format (for SFT models)")
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya on {device}...")

    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)

    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(args.checkpoint))
    ckpt_mgr.load(model, checkpoint_path=args.checkpoint)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    generator = TextGenerator(model, tokenizer, device=device)
    generator.temperature = args.temperature
    generator.top_p = args.top_p

    evaluate(model, tokenizer, generator, use_chat_format=args.chat)


if __name__ == "__main__":
    main()
