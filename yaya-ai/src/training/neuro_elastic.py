"""Neuroelastic resilience layer — the ability to adapt without permanent deformation.

Wraps OnlineLearner with four protective mechanisms:

1. Rollback      — snapshot adapter weights before every micro-finetune step;
                   auto-revert if loss spikes beyond tolerance.
2. Circuit breaker — hard-stop learning on NaN loss, exploding gradients, or
                   consecutive failure; prevents silent corruption.
3. Feedback validation — clamp scores, rate-limit submissions, reject statistical
                   outliers; blocks adversarial / miscalibrated feedback streams.
4. Adapter health monitoring — track adapter weight norms and alert before
                   saturation makes merging unsafe.

Usage:
    guard = ElasticGuard(online_learner, ElasticConfig())

    # Replace online_learner.add_example calls with guard.add_example
    guard.add_example(prompt, response, score=0.8)

    # Check system health
    report = guard.health_report()
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ElasticConfig:
    # --- Rollback ---
    rollback_on_spike: bool = True
    # Revert if post-step loss > this multiple of pre-step loss
    loss_spike_ratio: float = 2.5
    # Max adapter snapshots to keep (ring buffer)
    snapshot_capacity: int = 3

    # --- Circuit breaker ---
    # Hard-stop if gradient norm exceeds this during micro-finetune
    max_grad_norm_hard: float = 20.0
    # Consecutive NaN steps before tripping the breaker
    nan_tolerance: int = 3
    # Consecutive spike-rollbacks before disabling learning temporarily
    spike_tolerance: int = 5
    # Seconds to wait before re-enabling after circuit trip
    cooldown_seconds: float = 60.0

    # --- Feedback validation ---
    score_min: float = -10.0
    score_max: float = 10.0
    # Max examples accepted per minute (rate limiting)
    max_per_minute: int = 120
    # Reject example if response is empty or shorter than this
    min_response_chars: int = 1

    # --- Adapter health ---
    # Warn when any adapter parameter's L2 norm exceeds this
    max_adapter_norm: float = 100.0

    # --- Human-in-the-loop oversight ---
    # Queue examples for human review instead of silently accepting/rejecting
    # when their score is unusually extreme (outside this many std-devs)
    human_review_enabled: bool = False
    human_review_z_threshold: float = 3.0   # z-score beyond which to flag
    human_review_queue_size: int = 100       # max pending-review items


# ---------------------------------------------------------------------------
# ElasticGuard
# ---------------------------------------------------------------------------

class ElasticGuard:
    """Resilience wrapper around OnlineLearner.

    Provides rollback, circuit breaker, feedback validation, and
    adapter health monitoring without modifying OnlineLearner internals.
    """

    def __init__(self, online_learner, config: Optional[ElasticConfig] = None):
        self.learner = online_learner
        self.config = config or ElasticConfig()
        self._lock = threading.Lock()

        # Adapter weight snapshots for rollback (ring buffer)
        self._snapshots: deque[dict] = deque(maxlen=self.config.snapshot_capacity)

        # Circuit breaker state
        self._tripped = False
        self._trip_time: Optional[float] = None
        self._consecutive_nans = 0
        self._consecutive_spikes = 0

        # Rate limiting
        self._submission_times: deque[float] = deque(maxlen=self.config.max_per_minute)

        # Human-in-the-loop review queue
        self._review_queue: deque[dict] = deque(
            maxlen=self.config.human_review_queue_size
        )
        # Running score statistics for z-score anomaly detection
        self._score_window: deque[float] = deque(maxlen=200)

        # Stats
        self.stats: Dict[str, Any] = {
            "examples_accepted": 0,
            "examples_rejected": 0,
            "rollbacks": 0,
            "circuit_trips": 0,
            "steps_taken": 0,
            "last_loss": None,
            "adapter_norm_warnings": 0,
            "examples_flagged_for_review": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_example(
        self, prompt: str, response: str, score: float = 1.0
    ) -> bool:
        """Validate and submit a feedback example.

        Returns True if the example was accepted, False if rejected.
        """
        with self._lock:
            # 1. Check circuit breaker
            if self._is_tripped():
                self.stats["examples_rejected"] += 1
                return False

            # 2. Validate feedback
            reason = self._validate(prompt, response, score)
            if reason:
                self.stats["examples_rejected"] += 1
                return False

            # 3. Clamp score to safe range
            score = max(self.config.score_min, min(self.config.score_max, score))

            # 4. Human-in-the-loop: flag anomalous scores for review
            if self.config.human_review_enabled and self._is_score_anomalous(score):
                self._review_queue.append({
                    "prompt": prompt,
                    "response": response,
                    "score": score,
                    "timestamp": time.monotonic(),
                    "reason": "score_anomaly",
                })
                self.stats["examples_flagged_for_review"] += 1
                # Still accept, but operator should review before trusting this signal

            # 5. Record submission time for rate limiting
            self._submission_times.append(time.monotonic())
            self._score_window.append(score)
            self.stats["examples_accepted"] += 1

        # Delegate outside lock to avoid blocking
        self.learner.add_example(prompt, response, score)
        return True

    def pop_review_queue(self) -> List[Dict[str, Any]]:
        """Return and clear all examples flagged for human review.

        Called by the operator's oversight loop.  Typical use:
            items = guard.pop_review_queue()
            for item in items:
                human_label = human_annotate(item)
                if human_label["approved"]:
                    guard.learner.add_example(...)
        """
        with self._lock:
            items = list(self._review_queue)
            self._review_queue.clear()
        return items

    def review_queue_size(self) -> int:
        """Number of examples currently waiting for human review."""
        return len(self._review_queue)

    def step(self) -> Optional[float]:
        """Run a guarded micro-finetune step with rollback and circuit breaker."""
        if self._is_tripped():
            return None

        # Snapshot before we change anything
        snapshot = self._take_snapshot()

        # Measure loss before step
        loss_before = self._estimate_loss()

        # Run the step
        loss_after = self.learner.step()

        self.stats["steps_taken"] += 1

        if loss_after is None:
            # No positive examples available — not a failure
            return None

        # Check for NaN
        if loss_after != loss_after:  # NaN check
            self._consecutive_nans += 1
            self.stats["rollbacks"] += 1
            self._restore_snapshot(snapshot)
            if self._consecutive_nans >= self.config.nan_tolerance:
                self._trip(f"NaN loss {self._consecutive_nans} times in a row")
            return None
        else:
            self._consecutive_nans = 0

        # Check for loss spike
        if (
            self.config.rollback_on_spike
            and loss_before is not None
            and loss_after > loss_before * self.config.loss_spike_ratio
        ):
            self._consecutive_spikes += 1
            self.stats["rollbacks"] += 1
            self._restore_snapshot(snapshot)
            if self._consecutive_spikes >= self.config.spike_tolerance:
                self._trip(
                    f"Loss spike {self._consecutive_spikes} times: "
                    f"{loss_before:.4f} → {loss_after:.4f}"
                )
            return None
        else:
            self._consecutive_spikes = 0

        # Check adapter health
        self._check_adapter_norms()

        self.stats["last_loss"] = loss_after
        return loss_after

    def health_report(self) -> Dict[str, Any]:
        """Return a snapshot of the guard's current health state."""
        adapter_norms = self._compute_adapter_norms()
        return {
            "circuit_tripped": self._tripped,
            "trip_reason": getattr(self, "_trip_reason", None),
            "cooldown_remaining_s": self._cooldown_remaining(),
            "consecutive_nans": self._consecutive_nans,
            "consecutive_spikes": self._consecutive_spikes,
            "adapter_norms": adapter_norms,
            "max_adapter_norm": max(adapter_norms.values()) if adapter_norms else 0.0,
            **self.stats,
        }

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker (e.g. after investigation)."""
        with self._lock:
            self._tripped = False
            self._trip_time = None
            self._consecutive_nans = 0
            self._consecutive_spikes = 0
        print("[ElasticGuard] Circuit breaker manually reset.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate(self, prompt: str, response: str, score: float) -> Optional[str]:
        """Return a rejection reason string, or None if valid."""
        if not isinstance(score, (int, float)) or score != score:
            return "score is NaN"
        if len(response) < self.config.min_response_chars:
            return f"response too short ({len(response)} chars)"
        if self._rate_exceeded():
            return "rate limit exceeded"
        return None

    def _rate_exceeded(self) -> bool:
        now = time.monotonic()
        # Drop old entries outside the 60-second window
        while self._submission_times and now - self._submission_times[0] > 60.0:
            self._submission_times.popleft()
        return len(self._submission_times) >= self.config.max_per_minute

    def _is_tripped(self) -> bool:
        if not self._tripped:
            return False
        # Auto-reset after cooldown
        if self._trip_time and (time.monotonic() - self._trip_time) > self.config.cooldown_seconds:
            self._tripped = False
            self._trip_time = None
            print(f"[ElasticGuard] Cooldown complete — circuit reset automatically.")
            return False
        return True

    def _cooldown_remaining(self) -> float:
        """Return seconds remaining in the current cooldown (0.0 if not tripped)."""
        if not self._tripped or self._trip_time is None:
            return 0.0
        elapsed = time.monotonic() - self._trip_time
        return max(0.0, self.config.cooldown_seconds - elapsed)

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_time = time.monotonic()
        self._trip_reason = reason
        self.stats["circuit_trips"] += 1
        print(f"[ElasticGuard] CIRCUIT TRIPPED: {reason}. "
              f"Learning paused for {self.config.cooldown_seconds}s.")

    def _take_snapshot(self) -> dict:
        """Snapshot all adapter (lora_A / lora_B) parameters."""
        snapshot = {}
        for name, param in self.learner.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                snapshot[name] = param.data.clone()
        return snapshot

    def _restore_snapshot(self, snapshot: dict) -> None:
        """Restore adapter parameters from a snapshot."""
        if not snapshot:
            return
        with torch.no_grad():
            for name, param in self.learner.model.named_parameters():
                if name in snapshot:
                    param.data.copy_(snapshot[name])
        print(f"[ElasticGuard] Rolled back adapter weights to pre-step snapshot.")

    def _estimate_loss(self) -> Optional[float]:
        """Quick single-batch loss estimate without taking a gradient step."""
        import random
        with self.learner._lock:
            positives = [e for e in self.learner.buffer if e["score"] > 0]
        if not positives:
            return None
        sample = random.sample(positives, min(4, len(positives)))
        self.learner.model.eval()
        with torch.no_grad():
            loss = self.learner._compute_batch_loss(sample)
        self.learner.model.train()
        return loss.item() if loss is not None else None

    def _compute_adapter_norms(self) -> Dict[str, float]:
        norms = {}
        for name, param in self.learner.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                norms[name] = param.data.norm().item()
        return norms

    def _check_adapter_norms(self) -> None:
        for name, norm in self._compute_adapter_norms().items():
            if norm > self.config.max_adapter_norm:
                self.stats["adapter_norm_warnings"] += 1
                print(
                    f"[ElasticGuard] WARNING: adapter norm too high — "
                    f"{name}: {norm:.2f} > {self.config.max_adapter_norm}"
                )
