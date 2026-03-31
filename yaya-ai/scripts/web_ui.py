"""Simple web chat UI for Yaya.

Runs a local web server with a chat interface in the browser.

Usage:
    python scripts/web_ui.py
    python scripts/web_ui.py --checkpoint checkpoints/yaya-tiny-sft-focused/checkpoint-00005000

Then open http://localhost:7860 in your browser.
"""

import argparse
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.tokenizer.tokenizer import ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN
from src.inference.generator import GenerationConfig

SYSTEM_PROMPT = (
    "You are Yaya, a helpful and friendly AI assistant. "
    "You answer questions clearly, tell jokes when asked, and are always honest."
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


def load_model(model_config_path, checkpoint_path, device):
    from src.utils.config import load_model_config
    from src.model.yaya_model import YayaForCausalLM
    from src.tokenizer.tokenizer import YayaTokenizer
    from src.inference.generator import TextGenerator
    from src.training.checkpointing import CheckpointManager
    from src.agent.persistent_memory import SessionMemory

    print(f"Loading Yaya from {checkpoint_path} on {device}...")
    model_config = load_model_config(model_config_path)
    model = YayaForCausalLM(model_config)

    ckpt_mgr = CheckpointManager(save_dir=os.path.dirname(checkpoint_path))
    ckpt_mgr.load(model, checkpoint_path=checkpoint_path)
    model.eval().to(device)

    tokenizer = YayaTokenizer("data/tokenizer/yaya_tokenizer.model")
    memory = SessionMemory(store_dir="data/memory", session_id="web_ui")
    generator = TextGenerator(model, tokenizer, device=device, memory=memory)
    print("Yaya ready.")
    return generator, tokenizer


def run_gradio(generator, tokenizer, args):
    try:
        import gradio as gr
    except ImportError:
        print("Installing gradio...")
        os.system(f"{sys.executable} -m pip install gradio -q")
        import gradio as gr

    from scripts.continuous_learn import log_conversation

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=50,
        repetition_penalty=1.5,
        do_sample=args.temperature > 0,
    )

    def chat(message, history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, bot_msg in history:
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})

        prompt = tokenizer.format_chat(messages) + "\n" + ASSISTANT_TOKEN + "\n"
        full_output = generator.generate(prompt, gen_cfg)
        response = full_output[len(prompt):]

        for stop in [USER_TOKEN, SYSTEM_TOKEN, "</s>", "<|endoftext|>"]:
            if stop in response:
                response = response.split(stop)[0]
        response = response.strip()

        # Log for continuous learning
        log_conversation(message, response)

        return response

    with gr.Blocks(title="Chat with Yaya", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 💬 Chat with Yaya\nYaya is a custom AI assistant built from scratch.")
        chatbot = gr.ChatInterface(
            fn=chat,
            examples=[
                "Tell me a joke.",
                "What is artificial intelligence?",
                "Write a short poem about the stars.",
                "What can you do?",
            ],
            title="",
        )

    print(f"\nYaya web UI running at http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/model/yaya_125m.yaml")
    parser.add_argument("--checkpoint",   type=str, default=None)
    parser.add_argument("--port",         type=int, default=7860)
    parser.add_argument("--share",        action="store_true",
                        help="Create public Gradio link (useful on Colab)")
    parser.add_argument("--max_tokens",   type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = _find_latest_checkpoint(DEFAULT_CHECKPOINT_DIRS)
    if checkpoint is None:
        print("ERROR: No checkpoint found. Train a model first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator, tokenizer = load_model(args.model_config, checkpoint, device)
    run_gradio(generator, tokenizer, args)


if __name__ == "__main__":
    main()
