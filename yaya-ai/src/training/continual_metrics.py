"""Continual learning evaluation metrics.

Tracks per-task performance across training phases to quantify:

1. **Forgetting** — average drop in task accuracy after training on later tasks.
   F_j = max_{t≤T}(R_{j,t}) − R_{j,T}   (higher = worse)

2. **Backward transfer (BT)** — how learning new tasks changed past-task scores.
   Negative BT = catastrophic forgetting. Positive BT = benefited past tasks.

3. **Plasticity** — average final score across all tasks (higher = better).
   Measures the model's ability to learn new tasks.

4. **Intransigence** — proxy for how hard new tasks are to learn (gap between
   ideal single-task performance and continual learning performance).

Reference: Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for
Continual Learning" — https://arxiv.org/abs/1706.08840
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    task_id: str
    phase: int       # Training phase index (0 = first task, 1 = second task, ...)
    score: float     # Performance metric — higher is better (e.g. accuracy, -loss)


# ---------------------------------------------------------------------------
# ForgettingTracker
# ---------------------------------------------------------------------------

class ForgettingTracker:
    """Track per-task performance across training phases.

    Usage pattern:
        tracker = ForgettingTracker()

        # Phase 0: trained on task_A
        tracker.record("task_A", phase=0, score=0.92)
        tracker.record("task_B", phase=0, score=0.45)  # zero-shot on B

        # Phase 1: trained on task_B
        tracker.record("task_A", phase=1, score=0.81)  # dropped 0.11
        tracker.record("task_B", phase=1, score=0.89)

        report = tracker.report()
        # avg_forgetting=0.11, backward_transfer=-0.11, plasticity=0.85

    Tip: use -perplexity or accuracy as the score so "higher is better" holds.
    """

    def __init__(self):
        self._records: List[TaskRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, task_id: str, phase: int, score: float) -> None:
        """Record a task's performance after a training phase.

        Args:
            task_id:  Unique name for the task (e.g. "sentiment", "qa_v1").
            phase:    Training phase index — 0 for first task, 1 for second, etc.
            score:    Performance on this task.  Higher = better (use -loss or accuracy).
        """
        self._records.append(TaskRecord(task_id=task_id, phase=phase, score=score))

    # ------------------------------------------------------------------
    # Per-task helpers
    # ------------------------------------------------------------------

    def task_ids(self) -> List[str]:
        """Return task IDs in order first seen."""
        seen: Dict[str, None] = {}
        for r in self._records:
            seen[r.task_id] = None
        return list(seen)

    def scores_for(self, task_id: str) -> List[Tuple[int, float]]:
        """Return [(phase, score)] sorted by phase for a given task."""
        return sorted(
            [(r.phase, r.score) for r in self._records if r.task_id == task_id],
            key=lambda x: x[0],
        )

    # ------------------------------------------------------------------
    # Forgetting metrics
    # ------------------------------------------------------------------

    def forgetting(self) -> Dict[str, float]:
        """Per-task forgetting: peak_score − final_score.

        Positive values mean the model forgot; negative means it improved
        (backward-transfer benefit).  Only computed for tasks with ≥2 records.
        """
        result: Dict[str, float] = {}
        for task_id in self.task_ids():
            history = self.scores_for(task_id)
            if len(history) < 2:
                continue
            scores = [s for _, s in history]
            result[task_id] = max(scores) - scores[-1]
        return result

    def avg_forgetting(self) -> float:
        """Average forgetting across all tasks with ≥2 measurements.

        Returns 0.0 if no tasks have been evaluated more than once.
        """
        f = self.forgetting()
        return sum(f.values()) / len(f) if f else 0.0

    def backward_transfer(self) -> float:
        """Backward transfer: avg change in past-task scores after new training.

        BT = avg(final_score − first_score) over tasks with ≥2 records.
        Negative = catastrophic forgetting. Positive = beneficial transfer.
        """
        total, count = 0.0, 0
        for task_id in self.task_ids():
            history = self.scores_for(task_id)
            if len(history) < 2:
                continue
            first_score = history[0][1]
            last_score = history[-1][1]
            total += last_score - first_score
            count += 1
        return total / count if count > 0 else 0.0

    def plasticity(self) -> float:
        """Average final-phase score across all tasks.

        Measures the model's ability to learn (higher = better).
        """
        finals = []
        for task_id in self.task_ids():
            history = self.scores_for(task_id)
            if history:
                finals.append(history[-1][1])
        return sum(finals) / len(finals) if finals else 0.0

    def max_forgetting(self) -> Optional[Tuple[str, float]]:
        """Return (task_id, forgetting_score) for the most-forgotten task."""
        f = self.forgetting()
        if not f:
            return None
        worst = max(f, key=f.__getitem__)
        return worst, f[worst]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> Dict:
        """Full continual learning report dict.

        Keys:
            num_tasks:             int
            num_phases:            int
            avg_forgetting:        float — lower is better
            backward_transfer:     float — higher is better
            plasticity:            float — higher is better
            per_task_forgetting:   dict[task_id, float]
            per_task_history:      dict[task_id, list[(phase, score)]]
        """
        phases = [r.phase for r in self._records]
        return {
            "num_tasks": len(self.task_ids()),
            "num_phases": (max(phases) + 1) if phases else 0,
            "avg_forgetting": self.avg_forgetting(),
            "backward_transfer": self.backward_transfer(),
            "plasticity": self.plasticity(),
            "per_task_forgetting": self.forgetting(),
            "per_task_history": {
                tid: self.scores_for(tid) for tid in self.task_ids()
            },
        }

    def summary_line(self) -> str:
        """One-line human-readable summary for logging."""
        r = self.report()
        return (
            f"[ForgettingTracker] tasks={r['num_tasks']} phases={r['num_phases']} "
            f"avg_forgetting={r['avg_forgetting']:.4f} "
            f"BT={r['backward_transfer']:.4f} "
            f"plasticity={r['plasticity']:.4f}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "records": [
                {"task_id": r.task_id, "phase": r.phase, "score": r.score}
                for r in self._records
            ]
        }

    def load_state_dict(self, state: dict) -> None:
        self._records = [TaskRecord(**rec) for rec in state.get("records", [])]

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        print(f"[ForgettingTracker] state saved to {path} ({len(self._records)} records)")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(state)
        print(f"[ForgettingTracker] loaded {len(self._records)} records from {path}")
