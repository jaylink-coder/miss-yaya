"""Alignment and capability drift monitoring.

Tracks statistical signatures of model behavior over time to detect when
a continually-adapting model drifts from its intended behavior — a critical
safety layer for systems that self-modify via online learning or MAML.

Three detection signals:

1. **KL divergence drift** — compare current token-probability distributions
   on fixed "probe" prompts against a reference snapshot.  High KL = the
   model's predictions have shifted significantly.

2. **Entropy collapse** — monitor average output entropy.  A sudden drop
   often signals the model has overfit to recent feedback (reward hacking)
   or that adapter saturation has reduced output diversity.

3. **Score regression** — if ForgettingTracker scores are provided, alert
   when any task falls below a registered safety floor.

Usage:
    monitor = AlignmentMonitor(model, tokenizer, AlignmentConfig())

    # Register probe prompts that represent expected behavior
    monitor.add_probe("What is 2 + 2?")
    monitor.add_probe("Translate 'hello' to Swahili:")

    # After initial training, set the reference baseline
    monitor.set_reference()

    # After each online learning step:
    report = monitor.check_drift()
    if report["drift_detected"]:
        print("ALIGNMENT ALERT:", report["alerts"])

Reference: Amodei et al. (2016) "Concrete Problems in AI Safety";
           Leike et al. (2018) "AI Safety Gridworlds"
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AlignmentConfig:
    # Max probe prompts stored
    max_probes: int = 10
    # Max tokens generated per probe for distribution measurement
    probe_tokens: int = 20
    # Alert if KL divergence from reference exceeds this (nats)
    kl_alert_threshold: float = 0.5
    # Alert if mean output entropy drops below this (nats)
    # Low entropy = overconfident / collapsed distribution
    entropy_alert_threshold: float = 0.3
    # Alert if any task score in score_floors drops by more than this fraction
    score_regression_threshold: float = 0.10
    # How many historical snapshots to keep for trend analysis
    snapshot_history: int = 50


# ---------------------------------------------------------------------------
# AlignmentMonitor
# ---------------------------------------------------------------------------

class AlignmentMonitor:
    """Monitors model output distribution for alignment drift.

    Maintains a reference distribution snapshot and compares it against
    current model outputs on fixed probe prompts.  Emits structured alerts
    when any signal exceeds its configured threshold.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[AlignmentConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AlignmentConfig()

        self._probes: List[str] = []
        self._reference: Optional[Dict[str, torch.Tensor]] = None
        self._reference_entropy: Optional[float] = None
        self._history: deque[dict] = deque(maxlen=self.config.snapshot_history)
        # Task score safety floors: {task_id: min_acceptable_score}
        self._score_floors: Dict[str, float] = {}

        self.stats: Dict[str, Any] = {
            "checks_run": 0,
            "alerts_raised": 0,
            "drift_events": 0,
        }

    # ------------------------------------------------------------------
    # Probe management
    # ------------------------------------------------------------------

    def add_probe(self, prompt: str) -> None:
        """Register a prompt whose output distribution is monitored.

        Choose prompts that represent the model's core expected behaviors
        (e.g. basic reasoning, language tasks, safety-critical queries).
        """
        if len(self._probes) < self.config.max_probes:
            self._probes.append(prompt)

    def set_reference(self) -> None:
        """Snapshot current distributions as the alignment reference baseline.

        Call this after initial training, before any online adaptation begins.
        This establishes the "intended behavior" distribution.
        """
        if not self._probes:
            print("[AlignmentMonitor] WARNING: no probes registered — reference not set.")
            return
        dist = self._measure_distributions()
        self._reference = dist
        self._reference_entropy = _mean_entropy(dist)
        print(
            f"[AlignmentMonitor] Reference baseline set on {len(self._probes)} probes "
            f"(mean entropy={self._reference_entropy:.3f} nats)."
        )

    def set_score_floor(self, task_id: str, min_score: float) -> None:
        """Register a minimum acceptable score for a task.

        If ForgettingTracker scores fall below this, an alert is raised.
        Use negative loss (e.g. -2.0) or accuracy (e.g. 0.75) as the score.
        """
        self._score_floors[task_id] = min_score

    # ------------------------------------------------------------------
    # Drift checking
    # ------------------------------------------------------------------

    def check_drift(
        self,
        task_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run a full alignment check and return a structured report.

        Args:
            task_scores: Optional {task_id: current_score} from ForgettingTracker.
                         Used to check score regression against registered floors.

        Returns dict with:
            drift_detected:  bool — True if any signal exceeded threshold
            alerts:          list[str] — human-readable alert messages
            kl_divergence:   float — mean KL vs reference (0.0 if no reference)
            mean_entropy:    float — mean output entropy of current distributions
            score_regressions: dict[task_id, float] — tasks that dropped below floor
        """
        self.stats["checks_run"] += 1
        alerts: List[str] = []
        kl_div = 0.0
        mean_entropy = 0.0
        score_regressions: Dict[str, float] = {}

        if self._probes:
            current_dist = self._measure_distributions()
            mean_entropy = _mean_entropy(current_dist)

            # KL divergence from reference
            if self._reference is not None:
                kl_div = _mean_kl(self._reference, current_dist)
                if kl_div > self.config.kl_alert_threshold:
                    alerts.append(
                        f"KL divergence {kl_div:.3f} nats exceeds threshold "
                        f"{self.config.kl_alert_threshold} — output distribution shifted."
                    )

            # Entropy collapse
            if (
                self._reference_entropy is not None
                and mean_entropy < self._reference_entropy - self.config.entropy_alert_threshold
            ):
                alerts.append(
                    f"Output entropy collapsed: {mean_entropy:.3f} nats "
                    f"(was {self._reference_entropy:.3f}, threshold drop "
                    f"{self.config.entropy_alert_threshold})."
                )

        # Score regression check
        if task_scores and self._score_floors:
            for task_id, floor in self._score_floors.items():
                current = task_scores.get(task_id)
                if current is not None and current < floor:
                    drop = floor - current
                    if drop / (abs(floor) + 1e-8) > self.config.score_regression_threshold:
                        score_regressions[task_id] = current
                        alerts.append(
                            f"Task '{task_id}' score {current:.3f} fell below safety "
                            f"floor {floor:.3f} (drop {drop:.3f})."
                        )

        drift_detected = len(alerts) > 0
        if drift_detected:
            self.stats["alerts_raised"] += len(alerts)
            self.stats["drift_events"] += 1
            for alert in alerts:
                print(f"[AlignmentMonitor] ALERT: {alert}")

        snapshot = {
            "timestamp": time.monotonic(),
            "kl_divergence": kl_div,
            "mean_entropy": mean_entropy,
            "drift_detected": drift_detected,
            "alerts": alerts,
            "score_regressions": score_regressions,
        }
        self._history.append(snapshot)

        return snapshot

    def drift_trend(self) -> List[float]:
        """Return KL divergence values over the last N snapshots."""
        return [s["kl_divergence"] for s in self._history]

    def is_safe(self) -> bool:
        """Return True if the last check passed (or no check has been run)."""
        if not self._history:
            return True
        return not self._history[-1]["drift_detected"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        state = {
            "probes": self._probes,
            "reference": self._reference,
            "reference_entropy": self._reference_entropy,
            "score_floors": self._score_floors,
            "stats": self.stats,
        }
        torch.save(state, path)
        print(f"[AlignmentMonitor] state saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self._probes = state.get("probes", [])
        self._reference = state.get("reference")
        self._reference_entropy = state.get("reference_entropy")
        self._score_floors = state.get("score_floors", {})
        self.stats.update(state.get("stats", {}))
        print(f"[AlignmentMonitor] loaded from {path} ({len(self._probes)} probes)")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _measure_distributions(self) -> Dict[str, torch.Tensor]:
        """Run each probe and capture the first-token probability distribution.

        Returns {prompt: softmax_probs[vocab_size]} — one distribution per probe.
        """
        device = next(self.model.parameters()).device
        result: Dict[str, torch.Tensor] = {}

        self.model.eval()
        with torch.no_grad():
            for prompt in self._probes:
                try:
                    ids = self.tokenizer.encode(prompt, add_bos=True)
                    if not ids:
                        continue
                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    out = self.model(input_ids=inp)
                    logits = out["logits"][:, -1, :]  # last-token logits
                    probs = F.softmax(logits.float(), dim=-1).squeeze(0)  # [vocab]
                    result[prompt] = probs.cpu()
                except Exception:
                    continue

        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mean_entropy(distributions: Dict[str, torch.Tensor]) -> float:
    """Mean entropy (nats) across all probe distributions."""
    if not distributions:
        return 0.0
    entropies = []
    for probs in distributions.values():
        probs = probs.clamp(min=1e-10)
        entropies.append(-(probs * probs.log()).sum().item())
    return sum(entropies) / len(entropies)


def _mean_kl(
    reference: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
) -> float:
    """Mean KL(reference || current) across shared probe keys (nats)."""
    shared = [k for k in reference if k in current]
    if not shared:
        return 0.0
    kls = []
    for k in shared:
        p = reference[k].clamp(min=1e-10)
        q = current[k].clamp(min=1e-10)
        kl = (p * (p.log() - q.log())).sum().item()
        kls.append(kl)
    return sum(kls) / len(kls)
