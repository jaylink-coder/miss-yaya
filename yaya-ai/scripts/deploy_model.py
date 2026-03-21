"""One-command deploy and serve script for Yaya.

Starts a FastAPI server with OpenAI-compatible API endpoints for the
fine-tuned Yaya model, including safety guardrails and streaming support.

Usage:
    # Basic serve
    python scripts/deploy_model.py --model_path outputs/yaya-dpo

    # With custom port and quantization
    python scripts/deploy_model.py \
        --model_path outputs/yaya-dpo-merged \
        --port 8000 \
        --quantize int8

    # CPU-only (slower but works everywhere)
    python scripts/deploy_model.py --model_path outputs/yaya-dpo --device cpu

Requirements:
    pip install transformers fastapi uvicorn pydantic
"""

import argparse
import json
import os
import sys
import time
import uuid
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(model_path: str, device: str = "auto", quantize: str = "none"):
    """Load model and tokenizer with optional quantization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {model_path}")
    print(f"Device: {device}, Quantization: {quantize}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True, "use_cache": True}

    if device == "auto" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif device == "cpu":
        model_kwargs["torch_dtype"] = torch.float32
    else:
        model_kwargs["device_map"] = device
        model_kwargs["torch_dtype"] = torch.bfloat16

    if quantize == "int8":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "int4":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e9:.2f}B parameters")
    return model, tokenizer


def create_app(model, tokenizer, enable_safety: bool = True):
    """Create FastAPI application with OpenAI-compatible endpoints."""
    import torch
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="Yaya AI API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Safety filter (lightweight) ────────────────────────────────────────────
    BLOCKED_PATTERNS = [
        "how to make a bomb", "how to synthesize", "how to hack into",
        "generate malware", "create a virus", "write ransomware",
    ]

    def safety_check(text: str) -> bool:
        if not enable_safety:
            return True
        text_lower = text.lower()
        return not any(p in text_lower for p in BLOCKED_PATTERNS)

    # ── Pydantic models ────────────────────────────────────────────────────────
    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "yaya"
        messages: List[Message]
        max_tokens: int = Field(default=1024, le=4096)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        stream: bool = False
        stop: Optional[List[str]] = None

    class CompletionRequest(BaseModel):
        model: str = "yaya"
        prompt: str
        max_tokens: int = Field(default=256, le=4096)
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False

    # ── Generation helper ──────────────────────────────────────────────────────
    def generate_text(messages_dicts, max_tokens, temperature, top_p):
        try:
            input_text = tokenizer.apply_chat_template(
                messages_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = ""
            for msg in messages_dicts:
                input_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            input_text += "<|im_start|>assistant\n"

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response_ids = outputs[0][input_len:]
        return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # ── Endpoints ──────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "yaya"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": "yaya",
                "object": "model",
                "owned_by": "yaya-ai",
            }]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # Safety check on last user message
        user_msgs = [m for m in request.messages if m.role == "user"]
        if user_msgs and not safety_check(user_msgs[-1].content):
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "yaya",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I'm not able to help with that request. If you have other questions, I'm happy to assist."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

        if request.stream:
            async def stream_response():
                response = generate_text(
                    messages_dicts, request.max_tokens,
                    request.temperature, request.top_p
                )
                # Simulate streaming by yielding word by word
                words = response.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "yaya",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                # Final chunk
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        start = time.time()
        response = generate_text(
            messages_dicts, request.max_tokens,
            request.temperature, request.top_p
        )
        elapsed = time.time() - start

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "yaya",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(tokenizer.encode(response)),
                "total_tokens": 0,
            },
            "timing": {"generation_seconds": round(elapsed, 2)}
        }

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response_ids = outputs[0][input_len:]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "yaya",
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop"
            }]
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Deploy Yaya model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "int8", "int4"],
                        help="Quantization method")
    parser.add_argument("--no_safety", action="store_true",
                        help="Disable safety filter")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.device, args.quantize)
    app = create_app(model, tokenizer, enable_safety=not args.no_safety)

    print(f"\n{'=' * 60}")
    print(f"Yaya AI Server")
    print(f"{'=' * 60}")
    print(f"  Model:    {args.model_path}")
    print(f"  Endpoint: http://{args.host}:{args.port}")
    print(f"  Docs:     http://{args.host}:{args.port}/docs")
    print(f"  Safety:   {'enabled' if not args.no_safety else 'DISABLED'}")
    print(f"{'=' * 60}")
    print()
    print("OpenAI-compatible API:")
    print(f"  POST http://localhost:{args.port}/v1/chat/completions")
    print(f"  POST http://localhost:{args.port}/v1/completions")
    print(f"  GET  http://localhost:{args.port}/v1/models")
    print(f"  GET  http://localhost:{args.port}/health")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
