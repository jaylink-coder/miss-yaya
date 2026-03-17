"""Production-grade API server for Yaya AI.

Extends the base FastAPI server with production middleware:
- Safety guardrails on input/output
- Rate limiting per client
- API key authentication
- Metrics collection and reporting
- Structured request logging
- Health checks with model status
"""

import time
import uuid
import json
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from src.serving.middleware import (
    RateLimiter,
    RateLimitConfig,
    APIKeyAuth,
    MetricsCollector,
    RequestMetrics,
    RequestLogger,
    ServerConfig,
)
from src.safety.filters import GuardrailsEngine


class ProductionServer:
    """Production server wrapping model inference with middleware.

    Integrates safety, rate limiting, auth, and metrics into
    a cohesive serving layer. Can be used standalone or with FastAPI.

    Usage:
        server = ProductionServer(config=ServerConfig(...))
        server.setup(generator=my_generator)

        # Process a request
        result = server.handle_completion(
            prompt="Hello",
            client_id="user-123",
            api_key="sk-xxx",
        )
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()

        # Initialize middleware components
        self.rate_limiter = RateLimiter(self.config.rate_limit) if self.config.enable_rate_limiting else None
        self.auth = APIKeyAuth(self.config.api_keys, enabled=self.config.enable_auth) if self.config.enable_auth else None
        self.metrics = MetricsCollector() if self.config.enable_metrics else None
        self.logger = RequestLogger() if self.config.log_requests else None
        self.safety = GuardrailsEngine() if self.config.enable_safety else None

        self._generator = None
        self._model_loaded = False

    def setup(self, generator=None):
        """Configure the server with a model generator.

        Args:
            generator: TextGenerator instance (or any object with .generate() method).
        """
        self._generator = generator
        self._model_loaded = generator is not None

    def _check_auth(self, api_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Validate API key. Returns error dict if invalid, None if ok."""
        if self.auth and not self.auth.validate(api_key):
            return {"error": "Invalid or missing API key", "status": 401}
        return None

    def _check_rate_limit(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Check rate limit. Returns error dict if exceeded, None if ok."""
        if self.rate_limiter and not self.rate_limiter.check(client_id):
            remaining = self.rate_limiter.get_remaining(client_id)
            return {
                "error": "Rate limit exceeded",
                "status": 429,
                "remaining": remaining,
            }
        return None

    def _check_input_safety(self, text: str) -> Optional[Dict[str, Any]]:
        """Check input safety. Returns error dict if blocked, None if ok."""
        if not self.safety:
            return None

        result = self.safety.check_input(text)
        if not result.is_safe:
            refusal = self.safety.get_refusal(result)
            return {
                "error": "Content policy violation",
                "status": 400,
                "message": refusal,
                "categories": [c.value for c in result.categories],
            }
        return None

    def _sanitize_output(self, text: str) -> str:
        """Sanitize model output."""
        if self.safety:
            return self.safety.sanitize_output(text)
        return text

    def handle_completion(
        self,
        prompt: str,
        client_id: str = "anonymous",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Handle a completion request with full middleware pipeline.

        Args:
            prompt: Input text.
            client_id: Client identifier for rate limiting.
            api_key: API key for authentication.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stream: Whether to stream tokens.

        Returns:
            Dict with 'response' or 'error'.
        """
        request_id = f"req-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        # Log request
        if self.logger:
            self.logger.log_request(request_id, "POST", "/v1/completions", client_id)

        # Auth check
        auth_error = self._check_auth(api_key)
        if auth_error:
            self._record_metrics("/v1/completions", "POST", auth_error["status"], start_time, client_id)
            return auth_error

        # Rate limit check
        rate_error = self._check_rate_limit(client_id)
        if rate_error:
            self._record_metrics("/v1/completions", "POST", rate_error["status"], start_time, client_id)
            return rate_error

        # Input safety check
        safety_error = self._check_input_safety(prompt)
        if safety_error:
            self._record_metrics("/v1/completions", "POST", safety_error["status"], start_time, client_id)
            return safety_error

        # Generate
        if not self._generator:
            response_text = f"[Model not loaded. Prompt received: {len(prompt)} chars]"
        else:
            try:
                response_text = self._generator.generate(prompt)
                # Remove prompt from output if present
                if response_text.startswith(prompt):
                    response_text = response_text[len(prompt):]
            except Exception as e:
                self._record_metrics("/v1/completions", "POST", 500, start_time, client_id)
                return {"error": f"Generation error: {type(e).__name__}", "status": 500}

        # Sanitize output
        response_text = self._sanitize_output(response_text)

        # Record metrics
        latency = (time.monotonic() - start_time) * 1000
        self._record_metrics(
            "/v1/completions", "POST", 200, start_time, client_id,
            input_tokens=len(prompt.split()),
            output_tokens=len(response_text.split()),
        )

        # Log response
        if self.logger:
            self.logger.log_response(request_id, 200, latency)

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.config.model_name,
            "response": response_text,
            "status": 200,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
            },
        }

    def handle_chat(
        self,
        messages: List[Dict[str, str]],
        client_id: str = "anonymous",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Handle a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} messages.
            client_id: Client identifier.
            api_key: API key.
            max_tokens: Max generation tokens.
            temperature: Sampling temperature.

        Returns:
            Dict with chat completion response.
        """
        # Extract prompt from messages
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        prompt = "\n".join(parts) + "\nassistant:"

        # Check safety on last user message
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        request_id = f"req-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        if self.logger:
            self.logger.log_request(request_id, "POST", "/v1/chat/completions", client_id)

        # Auth + rate limit
        auth_error = self._check_auth(api_key)
        if auth_error:
            return auth_error

        rate_error = self._check_rate_limit(client_id)
        if rate_error:
            return rate_error

        # Safety on user input
        if last_user:
            safety_error = self._check_input_safety(last_user)
            if safety_error:
                self._record_metrics("/v1/chat/completions", "POST", safety_error["status"], start_time, client_id)
                return safety_error

        # Generate
        if not self._generator:
            response_text = f"[Model not loaded. {len(messages)} messages received]"
        else:
            try:
                response_text = self._generator.generate(prompt)
                if response_text.startswith(prompt):
                    response_text = response_text[len(prompt):]
            except Exception as e:
                return {"error": f"Generation error: {type(e).__name__}", "status": 500}

        response_text = self._sanitize_output(response_text)
        latency = (time.monotonic() - start_time) * 1000

        self._record_metrics(
            "/v1/chat/completions", "POST", 200, start_time, client_id,
            output_tokens=len(response_text.split()),
        )

        if self.logger:
            self.logger.log_response(request_id, 200, latency)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "status": 200,
        }

    def health_check(self) -> Dict[str, Any]:
        """Return server health status."""
        return {
            "status": "healthy" if self._model_loaded else "degraded",
            "model_loaded": self._model_loaded,
            "model_name": self.config.model_name,
            "safety_enabled": self.config.enable_safety,
            "auth_enabled": self.config.enable_auth,
            "rate_limiting_enabled": self.config.enable_rate_limiting,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return collected metrics."""
        if self.metrics:
            return self.metrics.get_summary()
        return {"error": "Metrics not enabled"}

    def _record_metrics(
        self,
        endpoint: str,
        method: str,
        status: int,
        start_time: float,
        client_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        if self.metrics:
            latency = (time.monotonic() - start_time) * 1000
            self.metrics.record(RequestMetrics(
                endpoint=endpoint,
                method=method,
                status_code=status,
                latency_ms=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                client_id=client_id,
                timestamp=time.time(),
            ))
