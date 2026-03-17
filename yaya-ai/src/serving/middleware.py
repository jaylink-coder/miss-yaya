"""Production middleware for Yaya API server.

Provides rate limiting, request logging, metrics collection,
API key authentication, and CORS handling.
"""

import time
import hashlib
import threading
from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict


# ── Rate Limiter ───────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100_000
    burst_multiplier: float = 1.5


class TokenBucket:
    """Token bucket rate limiter.

    Allows bursts up to bucket capacity while enforcing
    a sustained rate over time.
    """

    def __init__(self, rate: float, capacity: float):
        """Initialize token bucket.

        Args:
            rate: Tokens added per second.
            capacity: Maximum bucket size (burst capacity).
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def consume(self, count: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        with self._lock:
            self._refill()
            if self.tokens >= count:
                self.tokens -= count
                return True
            return False

    @property
    def available(self) -> float:
        with self._lock:
            self._refill()
            return self.tokens


class RateLimiter:
    """Per-client rate limiter using token buckets.

    Tracks rate limits per API key or IP address.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, client_id: str) -> TokenBucket:
        with self._lock:
            if client_id not in self._buckets:
                rate = self.config.requests_per_minute / 60.0
                capacity = self.config.requests_per_minute * self.config.burst_multiplier
                self._buckets[client_id] = TokenBucket(rate, capacity)
            return self._buckets[client_id]

    def check(self, client_id: str, cost: float = 1.0) -> bool:
        """Check if a request is allowed.

        Args:
            client_id: Client identifier (API key or IP).
            cost: Request cost in tokens.

        Returns:
            True if request is allowed.
        """
        bucket = self._get_bucket(client_id)
        return bucket.consume(cost)

    def get_remaining(self, client_id: str) -> float:
        """Get remaining request budget for a client."""
        bucket = self._get_bucket(client_id)
        return bucket.available

    def reset(self, client_id: str):
        """Reset rate limit for a client."""
        with self._lock:
            self._buckets.pop(client_id, None)


# ── API Key Authentication ─────────────────────────────────────

class APIKeyAuth:
    """Simple API key authentication.

    Validates API keys against a set of known keys.
    Supports key hashing for secure storage.
    """

    def __init__(self, api_keys: Optional[List[str]] = None, enabled: bool = True):
        self.enabled = enabled
        self._key_hashes: set = set()
        if api_keys:
            for key in api_keys:
                self._key_hashes.add(self._hash_key(key))

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def add_key(self, key: str):
        self._key_hashes.add(self._hash_key(key))

    def revoke_key(self, key: str):
        self._key_hashes.discard(self._hash_key(key))

    def validate(self, key: Optional[str]) -> bool:
        """Validate an API key.

        Returns True if auth is disabled or key is valid.
        """
        if not self.enabled:
            return True
        if key is None:
            return False
        return self._hash_key(key) in self._key_hashes

    @property
    def num_keys(self) -> int:
        return len(self._key_hashes)


# ── Metrics Collector ──────────────────────────────────────────

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    client_id: str = ""
    timestamp: float = 0.0


class MetricsCollector:
    """Collect and aggregate API metrics.

    Tracks request counts, latencies, token usage, and error rates.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_history: int = 10000):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._latencies: List[float] = []
        self._endpoint_counts: Dict[str, int] = defaultdict(int)
        self._status_counts: Dict[int, int] = defaultdict(int)
        self._max_history = max_history
        self._start_time = time.time()

    def record(self, metrics: RequestMetrics):
        """Record metrics for a completed request."""
        with self._lock:
            self._total_requests += 1
            self._endpoint_counts[metrics.endpoint] += 1
            self._status_counts[metrics.status_code] += 1
            self._total_input_tokens += metrics.input_tokens
            self._total_output_tokens += metrics.output_tokens

            if metrics.status_code >= 400:
                self._total_errors += 1

            self._latencies.append(metrics.latency_ms)
            if len(self._latencies) > self._max_history:
                self._latencies = self._latencies[-self._max_history:]

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        with self._lock:
            uptime = time.time() - self._start_time
            latencies = sorted(self._latencies) if self._latencies else [0]

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": round(self._total_errors / max(self._total_requests, 1), 4),
                "requests_per_second": round(self._total_requests / max(uptime, 1), 2),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "latency_ms": {
                    "p50": round(latencies[len(latencies) // 2], 1),
                    "p95": round(latencies[int(len(latencies) * 0.95)], 1),
                    "p99": round(latencies[int(len(latencies) * 0.99)], 1),
                    "avg": round(sum(latencies) / len(latencies), 1),
                },
                "endpoints": dict(self._endpoint_counts),
                "status_codes": dict(self._status_counts),
            }

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._total_requests = 0
            self._total_errors = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._latencies = []
            self._endpoint_counts = defaultdict(int)
            self._status_counts = defaultdict(int)
            self._start_time = time.time()


# ── Request Logger ─────────────────────────────────────────────

class RequestLogger:
    """Structured request logging for observability.

    Logs request/response metadata for debugging and auditing.
    """

    def __init__(self, log_fn: Optional[Callable] = None, log_body: bool = False):
        """Initialize logger.

        Args:
            log_fn: Custom logging function. Uses print if None.
            log_body: Whether to log request/response bodies.
        """
        self.log_fn = log_fn or print
        self.log_body = log_body

    def log_request(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        client_id: str = "",
        body: Optional[Dict] = None,
    ):
        """Log an incoming request."""
        msg = f"[REQ] {request_id} {method} {endpoint} client={client_id}"
        if self.log_body and body:
            msg += f" body={body}"
        self.log_fn(msg)

    def log_response(
        self,
        request_id: str,
        status_code: int,
        latency_ms: float,
        body: Optional[Dict] = None,
    ):
        """Log an outgoing response."""
        msg = f"[RES] {request_id} status={status_code} latency={latency_ms:.1f}ms"
        if self.log_body and body:
            preview = str(body)[:200]
            msg += f" body={preview}"
        self.log_fn(msg)


# ── Server Config ──────────────────────────────────────────────

@dataclass
class ServerConfig:
    """Complete server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    model_path: str = ""
    model_name: str = "yaya"
    max_batch_size: int = 8
    max_sequence_length: int = 4096
    enable_safety: bool = True
    enable_rate_limiting: bool = True
    enable_auth: bool = False
    enable_metrics: bool = True
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_keys: List[str] = field(default_factory=list)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    log_requests: bool = True
