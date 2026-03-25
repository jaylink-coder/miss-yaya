"""FastAPI serving endpoint for Yaya model.

Provides a REST API for text generation, chat completion,
and model health checks. Compatible with OpenAI API format.
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from src.inference.generator import TextGenerator, GenerationConfig
from src.tokenizer.tokenizer import ASSISTANT_TOKEN
from src.agent.persistent_memory import PersistentMemory, SessionMemory


# --- Pydantic models for API request/response ---

if FASTAPI_AVAILABLE:

    class ChatMessage(BaseModel):
        role: str = "user"
        content: str = ""

    class CompletionRequest(BaseModel):
        model: str = "yaya"
        prompt: Optional[str] = None
        messages: Optional[List[ChatMessage]] = None
        max_tokens: int = Field(default=256, ge=1, le=4096)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_p: float = Field(default=0.9, ge=0.0, le=1.0)
        top_k: int = Field(default=50, ge=0)
        repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
        stream: bool = False
        session_id: Optional[str] = None  # For persistent memory: same id = same session

    class CompletionChoice(BaseModel):
        index: int = 0
        message: Optional[ChatMessage] = None
        text: Optional[str] = None
        finish_reason: str = "stop"

    class UsageInfo(BaseModel):
        prompt_tokens: int = 0
        completion_tokens: int = 0
        total_tokens: int = 0

    class CompletionResponse(BaseModel):
        id: str = ""
        object: str = "chat.completion"
        created: int = 0
        model: str = "yaya"
        choices: List[CompletionChoice] = []
        usage: UsageInfo = UsageInfo()


def create_app(generator: TextGenerator, model_name: str = "yaya") -> "FastAPI":
    """Create FastAPI application for model serving.

    Args:
        generator: Configured TextGenerator instance.
        model_name: Model name to report in API responses.

    Returns:
        FastAPI application.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI required for serving: pip install fastapi uvicorn")

    app = FastAPI(
        title="Yaya AI API",
        description="Multimodal Foundation Model API",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model": model_name}

    @app.get("/v1/models")
    async def list_models():
        return {
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "yaya-ai",
                }
            ]
        }

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Text completion endpoint."""
        if not request.prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
        )

        generated_text = generator.generate(request.prompt, config)
        # Remove prompt from output
        completion_text = generated_text[len(request.prompt):]

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            object="text_completion",
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=completion_text,
                    finish_reason="stop",
                )
            ],
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: CompletionRequest):
        """Chat completion endpoint (OpenAI-compatible)."""
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # Format chat messages into prompt, then open the assistant turn
        # so the model knows to generate an assistant response.
        prompt = generator.tokenizer.format_chat(
            [{"role": m.role, "content": m.content} for m in request.messages]
        ) + "\n" + ASSISTANT_TOKEN + "\n"

        config = GenerationConfig(
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
        )

        if request.stream:
            # Streaming response
            async def stream_tokens():
                for token in generator.stream_generate(prompt, config):
                    chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": token}}],
                    }
                    yield f"data: {__import__('json').dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_tokens(),
                media_type="text/event-stream",
            )

        # Non-streaming response
        generated_text = generator.generate(prompt, config)
        response_text = generated_text[len(prompt):]

        return CompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
        )

    return app
