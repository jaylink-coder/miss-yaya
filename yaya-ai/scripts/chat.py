"""Interactive chat with Yaya.

Usage:
    python scripts/chat.py \
        --model_config configs/model/yaya_125m.yaml \
        --checkpoint checkpoints/yaya-125m-sft/checkpoint-XXXXXXXX
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer
from src.inference.generator import TextGenerator
from src.training.checkpointing import CheckpointManager

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
)


def main():
    parser = argparse.ArgumentParser(description="Chat with Yaya")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--max_tokens",   type=int, default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya on {device}...")

    model_config = load_model_config(args.model_config)
    model = YayaForCausalLM(model_config)

    ckpt_dir = os.path.dirname(args.checkpoint)
    ckpt_mgr = CheckpointManager(save_dir=ckpt_dir)
    ckpt_mgr.load(model, checkpoint_path=args.checkpoint)

    model.eval().to(device)

    tokenizer = YayaTokenizer(
        os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)),
                     "data/tokenizer/yaya_tokenizer.model")
        if not os.path.exists("data/tokenizer/yaya_tokenizer.model")
        else "data/tokenizer/yaya_tokenizer.model"
    )

    generator = TextGenerator(model, tokenizer, device=device)

    print("\n" + "=" * 50)
    print("  Chat with Yaya  (type 'quit' to exit)")
    print("=" * 50 + "\n")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Yaya: Goodbye! It was great chatting with you.")
            break

        history.append({"role": "user", "content": user_input})

        # Format as chat prompt
        prompt = tokenizer.format_chat(history) + "<|assistant|>\n"

        response = generator.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Strip the prompt and any trailing tags from response
        if prompt in response:
            response = response[len(prompt):]
        response = response.replace("</|assistant|>", "").strip()

        print(f"Yaya: {response}\n")
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
