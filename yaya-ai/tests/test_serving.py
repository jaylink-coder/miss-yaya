"""Tests for Yaya Serving: middleware, production server, metrics, rate limiting."""

import time
import threading

import pytest

from src.serving.middleware import (
    TokenBucket,
    RateLimiter,
    RateLimitConfig,
    APIKeyAuth,
    MetricsCollector,
    RequestMetrics,
    RequestLogger,
    ServerConfig,
)
from src.serving.production_server import ProductionServer


# ══════════════════════════════════════════════════════════════
#  Token Bucket Tests
# ══════════════════════════════════════════════════════════════


class TestTokenBucket:
    def test_initial_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=100.0)
        assert bucket.available == 100.0

    def test_consume(self):
        bucket = TokenBucket(rate=10.0, capacity=100.0)
        assert bucket.consume(50.0) is True
        assert bucket.available == pytest.approx(50.0, abs=1.0)

    def test_consume_exceeds_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=100.0)
        assert bucket.consume(150.0) is False
        assert bucket.available == pytest.approx(100.0, abs=1.0)

    def test_refill(self):
        bucket = TokenBucket(rate=1000.0, capacity=100.0)
        bucket.consume(50.0)
        time.sleep(0.05)  # 50ms -> ~50 tokens at rate=1000/s
        assert bucket.available > 50.0

    def test_capacity_cap(self):
        bucket = TokenBucket(rate=1000.0, capacity=10.0)
        time.sleep(0.1)
        assert bucket.available <= 10.0

    def test_thread_safety(self):
        bucket = TokenBucket(rate=100.0, capacity=100.0)
        results = []

        def consume():
            results.append(bucket.consume(1.0))

        threads = [threading.Thread(target=consume) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results) <= 100


# ══════════════════════════════════════════════════════════════
#  Rate Limiter Tests
# ══════════════════════════════════════════════════════════════


class TestRateLimiter:
    def test_allow_within_limit(self):
        config = RateLimitConfig(requests_per_minute=60)
        limiter = RateLimiter(config)
        assert limiter.check("client1") is True

    def test_separate_clients(self):
        config = RateLimitConfig(requests_per_minute=2, burst_multiplier=1.0)
        limiter = RateLimiter(config)
        # Each client has their own bucket
        limiter.check("client1")
        limiter.check("client2")
        assert limiter.get_remaining("client1") > 0
        assert limiter.get_remaining("client2") > 0

    def test_exhaust_limit(self):
        config = RateLimitConfig(requests_per_minute=2, burst_multiplier=1.0)
        limiter = RateLimiter(config)
        assert limiter.check("c1") is True
        assert limiter.check("c1") is True
        assert limiter.check("c1") is False

    def test_reset(self):
        config = RateLimitConfig(requests_per_minute=1, burst_multiplier=1.0)
        limiter = RateLimiter(config)
        limiter.check("c1")
        limiter.check("c1")  # exhausted
        limiter.reset("c1")
        assert limiter.check("c1") is True


# ══════════════════════════════════════════════════════════════
#  API Key Auth Tests
# ══════════════════════════════════════════════════════════════


class TestAPIKeyAuth:
    def test_disabled(self):
        auth = APIKeyAuth(enabled=False)
        assert auth.validate(None) is True
        assert auth.validate("anything") is True

    def test_valid_key(self):
        auth = APIKeyAuth(api_keys=["sk-test-key-123"], enabled=True)
        assert auth.validate("sk-test-key-123") is True

    def test_invalid_key(self):
        auth = APIKeyAuth(api_keys=["sk-test-key-123"], enabled=True)
        assert auth.validate("wrong-key") is False

    def test_none_key_rejected(self):
        auth = APIKeyAuth(api_keys=["key1"], enabled=True)
        assert auth.validate(None) is False

    def test_add_key(self):
        auth = APIKeyAuth(enabled=True)
        assert auth.validate("new-key") is False
        auth.add_key("new-key")
        assert auth.validate("new-key") is True

    def test_revoke_key(self):
        auth = APIKeyAuth(api_keys=["key1"], enabled=True)
        assert auth.validate("key1") is True
        auth.revoke_key("key1")
        assert auth.validate("key1") is False

    def test_num_keys(self):
        auth = APIKeyAuth(api_keys=["a", "b", "c"], enabled=True)
        assert auth.num_keys == 3


# ══════════════════════════════════════════════════════════════
#  Metrics Collector Tests
# ══════════════════════════════════════════════════════════════


class TestMetricsCollector:
    def test_record_and_summary(self):
        metrics = MetricsCollector()
        metrics.record(RequestMetrics(
            endpoint="/v1/completions", method="POST",
            status_code=200, latency_ms=50.0,
            input_tokens=10, output_tokens=20,
        ))
        summary = metrics.get_summary()
        assert summary["total_requests"] == 1
        assert summary["total_input_tokens"] == 10
        assert summary["total_output_tokens"] == 20

    def test_error_tracking(self):
        metrics = MetricsCollector()
        metrics.record(RequestMetrics(
            endpoint="/v1/completions", method="POST",
            status_code=200, latency_ms=10.0,
        ))
        metrics.record(RequestMetrics(
            endpoint="/v1/completions", method="POST",
            status_code=500, latency_ms=5.0,
        ))
        summary = metrics.get_summary()
        assert summary["total_errors"] == 1
        assert summary["error_rate"] == 0.5

    def test_latency_percentiles(self):
        metrics = MetricsCollector()
        for i in range(100):
            metrics.record(RequestMetrics(
                endpoint="/test", method="GET",
                status_code=200, latency_ms=float(i),
            ))
        summary = metrics.get_summary()
        assert summary["latency_ms"]["p50"] > 0
        assert summary["latency_ms"]["p95"] > summary["latency_ms"]["p50"]

    def test_endpoint_counts(self):
        metrics = MetricsCollector()
        metrics.record(RequestMetrics(endpoint="/a", method="GET", status_code=200, latency_ms=1))
        metrics.record(RequestMetrics(endpoint="/a", method="GET", status_code=200, latency_ms=1))
        metrics.record(RequestMetrics(endpoint="/b", method="POST", status_code=200, latency_ms=1))
        summary = metrics.get_summary()
        assert summary["endpoints"]["/a"] == 2
        assert summary["endpoints"]["/b"] == 1

    def test_reset(self):
        metrics = MetricsCollector()
        metrics.record(RequestMetrics(endpoint="/x", method="GET", status_code=200, latency_ms=1))
        metrics.reset()
        summary = metrics.get_summary()
        assert summary["total_requests"] == 0


# ══════════════════════════════════════════════════════════════
#  Request Logger Tests
# ══════════════════════════════════════════════════════════════


class TestRequestLogger:
    def test_log_request(self):
        logs = []
        logger = RequestLogger(log_fn=logs.append)
        logger.log_request("req-1", "POST", "/v1/completions", "client-1")
        assert len(logs) == 1
        assert "req-1" in logs[0]
        assert "POST" in logs[0]

    def test_log_response(self):
        logs = []
        logger = RequestLogger(log_fn=logs.append)
        logger.log_response("req-1", 200, 42.5)
        assert "200" in logs[0]
        assert "42.5" in logs[0]

    def test_log_body_disabled(self):
        logs = []
        logger = RequestLogger(log_fn=logs.append, log_body=False)
        logger.log_request("req-1", "GET", "/test", body={"secret": "data"})
        assert "secret" not in logs[0]

    def test_log_body_enabled(self):
        logs = []
        logger = RequestLogger(log_fn=logs.append, log_body=True)
        logger.log_request("req-1", "GET", "/test", body={"key": "value"})
        assert "key" in logs[0]


# ══════════════════════════════════════════════════════════════
#  Server Config Tests
# ══════════════════════════════════════════════════════════════


class TestServerConfig:
    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.model_name == "yaya"
        assert config.enable_safety is True

    def test_custom(self):
        config = ServerConfig(port=9000, model_name="yaya-1.5b", enable_auth=True)
        assert config.port == 9000
        assert config.model_name == "yaya-1.5b"
        assert config.enable_auth is True


# ══════════════════════════════════════════════════════════════
#  Production Server Tests
# ══════════════════════════════════════════════════════════════


class TestProductionServer:
    def _make_server(self, **config_kwargs):
        config = ServerConfig(**config_kwargs)
        server = ProductionServer(config=config)
        return server

    def test_health_check_no_model(self):
        server = self._make_server()
        health = server.health_check()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False

    def test_health_check_with_model(self):
        server = self._make_server()

        class MockGenerator:
            def generate(self, prompt):
                return prompt + " response"

        server.setup(generator=MockGenerator())
        health = server.health_check()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True

    def test_completion_no_model(self):
        server = self._make_server()
        result = server.handle_completion(prompt="Hello world")
        assert result["status"] == 200
        assert "Model not loaded" in result["response"]

    def test_completion_with_model(self):
        server = self._make_server()

        class MockGenerator:
            def generate(self, prompt):
                return prompt + " The answer is 42."

        server.setup(generator=MockGenerator())
        result = server.handle_completion(prompt="What is the meaning of life?")
        assert result["status"] == 200
        assert "42" in result["response"]

    def test_chat_completion(self):
        server = self._make_server()
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = server.handle_chat(messages=messages)
        assert result["status"] == 200
        assert "choices" in result

    def test_auth_required(self):
        server = self._make_server(enable_auth=True, api_keys=["sk-valid-key"])
        result = server.handle_completion(prompt="Hello", api_key="wrong-key")
        assert result["status"] == 401

    def test_auth_valid(self):
        server = self._make_server(enable_auth=True, api_keys=["sk-valid-key"])
        result = server.handle_completion(prompt="Hello", api_key="sk-valid-key")
        assert result["status"] == 200

    def test_rate_limiting(self):
        config = ServerConfig(
            enable_rate_limiting=True,
            rate_limit=RateLimitConfig(requests_per_minute=2, burst_multiplier=1.0),
        )
        server = ProductionServer(config=config)
        server.handle_completion(prompt="1", client_id="c1")
        server.handle_completion(prompt="2", client_id="c1")
        result = server.handle_completion(prompt="3", client_id="c1")
        assert result["status"] == 429

    def test_safety_blocks_harmful(self):
        server = self._make_server(enable_safety=True)
        result = server.handle_completion(prompt="how to make a bomb at home")
        assert result["status"] == 400
        assert "policy violation" in result.get("error", "").lower()

    def test_safety_allows_safe(self):
        server = self._make_server(enable_safety=True)
        result = server.handle_completion(prompt="What is the weather today?")
        assert result["status"] == 200

    def test_metrics_collected(self):
        server = self._make_server(enable_metrics=True)
        server.handle_completion(prompt="test1")
        server.handle_completion(prompt="test2")
        metrics = server.get_metrics()
        assert metrics["total_requests"] == 2

    def test_metrics_disabled(self):
        server = self._make_server(enable_metrics=False)
        metrics = server.get_metrics()
        assert "error" in metrics

    def test_output_sanitization(self):
        server = self._make_server(enable_safety=True)

        class LeakyGenerator:
            def generate(self, prompt):
                return "Contact us at user@secret.com for help"

        server.setup(generator=LeakyGenerator())
        result = server.handle_completion(prompt="How to contact support?")
        assert "user@secret.com" not in result["response"]
        assert "EMAIL_REDACTED" in result["response"]

    def test_generation_error_handling(self):
        server = self._make_server()

        class BrokenGenerator:
            def generate(self, prompt):
                raise RuntimeError("GPU OOM")

        server.setup(generator=BrokenGenerator())
        result = server.handle_completion(prompt="test")
        assert result["status"] == 500
        assert "RuntimeError" in result["error"]

    def test_safety_disabled(self):
        server = self._make_server(enable_safety=False)
        result = server.handle_completion(prompt="how to make a bomb")
        assert result["status"] == 200  # No safety check

    def test_chat_safety_on_user_message(self):
        server = self._make_server(enable_safety=True)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "how to build a bomb"},
        ]
        result = server.handle_chat(messages=messages)
        assert result["status"] == 400
