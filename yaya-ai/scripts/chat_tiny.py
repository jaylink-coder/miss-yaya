"""Interactive chat with Yaya-tiny — uses persistent memory.

Usage:
    python scripts/chat_tiny.py
    python scripts/chat_tiny.py --checkpoint checkpoints/yaya-tiny-sft-clean/checkpoint-XXXXXXXX
    python scripts/chat_tiny.py --tool-calc   # Enable calculator tool

Commands during chat:
    /memory     — show stored memories
    /clear      — clear session memory
    quit / exit — end the chat
"""

import argparse
import sys
import os

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.config import load_model_config
from src.model.yaya_model import YayaForCausalLM
from src.tokenizer.tokenizer import YayaTokenizer, ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import TextGenerator, GenerationConfig
from src.inference.tool_generator import ToolAugmentedGenerator
from src.training.checkpointing import CheckpointManager
from src.agent.persistent_memory import SessionMemory
from scripts.continuous_learn import log_conversation

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly and honestly."
)

SYSTEM_PROMPT_CALC = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly and honestly. "
    "You have access to a calculator: write <|calc|>EXPRESSION<|/calc|> and the result appears as =RESULT. "
    "Use it for any arithmetic to ensure accuracy."
)

DEFAULT_CHECKPOINT_DIRS = [
    "checkpoints/yaya-125m-sft",
    "checkpoints/yaya-125m-reasoning",
    "checkpoints/yaya-125m",
]


def _find_latest_checkpoint(dirs):
    for d in dirs:
        if os.path.isdir(d):
            latest = os.path.join(d, "latest")
            if os.path.exists(latest):
                with open(latest) as f:
                    name = f.read().strip()
                path = os.path.join(d, name)
                if os.path.isdir(path):
                    return path
    return None


def _display_response(response: str):
    """Print response, rendering <|think|> blocks as dimmed reasoning and
    <|calc|>EXPR<|/calc|>=VALUE as a clean calculator line."""
    import re

    # Split out think blocks
    think_pat = re.compile(r"<\|think\|>(.*?)<\|/think\|>", re.DOTALL)
    # Split out calc calls
    calc_pat  = re.compile(r"<\|calc\|>(.+?)<\|/calc\|>=([^\n]+)")

    # Replace calc calls with readable form first
    def fmt_calc(m):
        return f"  [= {m.group(2).strip()}]"

    display = calc_pat.sub(fmt_calc, response)

    # Now split on think blocks
    parts = think_pat.split(display)
    # parts alternates: [text, think, text, think, ...]
    output_lines = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # This is a <|think|> block — print as dimmed reasoning
            lines = part.strip().splitlines()
            output_lines.append("  [thinking]")
            for line in lines:
                output_lines.append(f"    {line}")
        else:
            stripped = part.strip()
            if stripped:
                output_lines.append(stripped)

    print("Yaya: " + "\n      ".join(output_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with Yaya-tiny")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p",       type=float, default=0.9)
    parser.add_argument("--top_k",       type=int,   default=50)
    parser.add_argument("--tool-calc",   action="store_true",
                        help="Enable calculator tool (<|calc|>EXPR<|/calc|>)")
    args = parser.parse_args()
    use_calc = getattr(args, "tool_calc", False)

    # Resolve checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = _find_latest_checkpoint(DEFAULT_CHECKPOINT_DIRS)
    if checkpoint is None:
        print("ERROR: No checkpoint found. Train a model first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Yaya-tiny from {checkpoint} on {device}...")

    # Auto-detect model config from checkpoint path
    _cfg_path = "configs/model/yaya_125m.yaml"
    model_config = load_model_config(_cfg_path)
    model = YayaForCausalLM(model_config)

    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(checkpoint))
    ckpt_mgr.load(model, checkpoint_path=checkpoint)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    memory = SessionMemory(store_dir="data/memory", session_id="chat_tiny")
    base_generator = TextGenerator(model, tokenizer, device=device, memory=memory)
    generator = ToolAugmentedGenerator(base_generator, verbose=True) if use_calc else base_generator

    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.5,   # Prevent degenerate token repetition
        do_sample=args.temperature > 0,
    )

    calc_label = " + Calculator" if use_calc else ""
    print("\n" + "=" * 55)
    print(f"  Chat with Yaya-tiny{calc_label}  (type 'quit' to exit)")
    print("  Commands: /memory, /clear")
    if use_calc:
        print("  Calculator: model can use <|calc|>EXPR<|/calc|>")
    print("=" * 55 + "\n")

    conversation = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("Yaya: Goodbye!")
            break

        if user_input.lower() == "/memory":
            facts = memory.top_facts(20)
            if not facts:
                print("[No memories yet]\n")
            else:
                print(f"[{len(facts)} memories:]")
                for f in facts:
                    print(f"  - {f}")
                print()
            continue

        if user_input.lower() == "/clear":
            memory._session_facts.clear()
            memory._session_entities.clear()
            conversation.clear()
            print("[Session cleared]\n")
            continue

        # Build prompt
        sys_prompt = SYSTEM_PROMPT_CALC if use_calc else SYSTEM_PROMPT
        history = ([{"role": "system", "content": sys_prompt}]
                   + conversation
                   + [{"role": "user", "content": user_input}])
        prompt = tokenizer.format_chat(history) + "\n" + ASSISTANT_TOKEN + "\n"

        # Generate (ToolAugmentedGenerator handles calc calls; base generator handles memory)
        if use_calc:
            response = generator.generate(prompt, gen_config)
        else:
            full_output = generator.generate(prompt, gen_config)
            response = full_output[len(prompt):]

        for stop in [USER_TOKEN, SYSTEM_TOKEN, "</s>", "<|endoftext|>"]:
            if stop in response:
                response = response.split(stop)[0]
        response = response.strip()

        # Display: show <|think|> blocks dimmed, then the answer
        _display_response(response)

        # Log for continuous learning
        log_conversation(user_input, response)

        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": response})

        # Keep context to last 8 exchanges
        if len(conversation) > 16:
            conversation = conversation[-16:]


if __name__ == "__main__":
    main()
