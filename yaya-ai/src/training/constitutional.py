"""Constitutional AI — self-critique and revision loop for Yaya.

Implements Anthropic's Constitutional AI (CAI) approach:
  1. Generate an initial response.
  2. Critique it against a set of constitutional principles.
  3. Revise the response based on the critique.
  4. Optionally generate (initial, revised) preference pairs for DPO training.

This gives Yaya a self-improvement loop that does not require human labels —
the model uses its own principles to score and correct itself.

Reference: Bai et al. (2022) "Constitutional AI: Harmlessness from AI Feedback"

Usage:
    cai = ConstitutionalAI(generator.generate)

    # One-shot improvement
    result = cai.revise("How do I pick a lock?")
    print(result["revised"])       # safer / better response
    print(result["critique"])      # what was wrong with the original

    # Collect DPO preference pairs for offline training
    pairs = cai.generate_preference_pairs(prompts, n_revisions=2)
    # pairs: list of {"prompt", "chosen" (revised), "rejected" (original)}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constitutional principles
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """A single principle the model should critique itself against."""
    name: str
    critique_prompt: str     # Instructions for generating the critique
    revision_prompt: str     # Instructions for revising based on the critique

    def format_critique(self, prompt: str, response: str) -> str:
        return (
            f"Human request: {prompt}\n\n"
            f"AI response: {response}\n\n"
            f"{self.critique_prompt}\n\n"
            "Critique:"
        )

    def format_revision(self, prompt: str, response: str, critique: str) -> str:
        return (
            f"Human request: {prompt}\n\n"
            f"Original AI response: {response}\n\n"
            f"Critique: {critique}\n\n"
            f"{self.revision_prompt}\n\n"
            "Revised response:"
        )


# ---------------------------------------------------------------------------
# Built-in principle library
# ---------------------------------------------------------------------------

HELPFULNESS = ConstitutionalPrinciple(
    name="helpfulness",
    critique_prompt=(
        "Identify any ways the above response is unhelpful, incomplete, or unclear. "
        "Point out missing information or vague answers."
    ),
    revision_prompt=(
        "Please rewrite the AI response to be more helpful, complete, and clear, "
        "addressing the critique above."
    ),
)

HARMLESSNESS = ConstitutionalPrinciple(
    name="harmlessness",
    critique_prompt=(
        "Identify any ways the above response could be harmful, dangerous, misleading, "
        "or could facilitate illegal activity. Be specific."
    ),
    revision_prompt=(
        "Please rewrite the AI response to remove any harmful, dangerous, or misleading "
        "content, while still being as helpful as possible."
    ),
)

HONESTY = ConstitutionalPrinciple(
    name="honesty",
    critique_prompt=(
        "Identify any ways the above response contains false information, "
        "unwarranted certainty, or fails to acknowledge limitations."
    ),
    revision_prompt=(
        "Please rewrite the AI response to be more accurate and honest, "
        "correcting any errors and acknowledging uncertainty where appropriate."
    ),
)

CONCISENESS = ConstitutionalPrinciple(
    name="conciseness",
    critique_prompt=(
        "Identify any ways the above response is unnecessarily verbose, repetitive, "
        "or padded with filler content."
    ),
    revision_prompt=(
        "Please rewrite the AI response to be more concise and direct, "
        "removing unnecessary padding while keeping all essential information."
    ),
)

# Default constitution — applied in order during revision
DEFAULT_CONSTITUTION: List[ConstitutionalPrinciple] = [
    HARMLESSNESS,
    HONESTY,
    HELPFULNESS,
]


# ---------------------------------------------------------------------------
# ConstitutionalAI
# ---------------------------------------------------------------------------

@dataclass
class CAIConfig:
    """Configuration for the Constitutional AI loop."""
    n_revisions: int = 1          # How many critique-revise cycles per response
    max_critique_tokens: int = 150
    max_revision_tokens: int = 300
    # If True, use the revised response as input for the next revision cycle
    chain_revisions: bool = True
    # Principles to apply (in order); uses DEFAULT_CONSTITUTION if None
    principles: Optional[List[ConstitutionalPrinciple]] = None

    def get_principles(self) -> List[ConstitutionalPrinciple]:
        return self.principles if self.principles is not None else DEFAULT_CONSTITUTION


class ConstitutionalAI:
    """Self-critique and revision loop.

    Wraps any generate_fn (str → str) and applies constitutional principles
    to improve responses without human labels.

    Args:
        generate_fn: Function that takes a prompt string and returns a response string.
        config:      CAIConfig controlling the revision loop.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        config: Optional[CAIConfig] = None,
    ):
        self.generate_fn = generate_fn
        self.config = config or CAIConfig()

    def _generate(self, prompt: str) -> str:
        """Call generate_fn and strip leading/trailing whitespace."""
        return self.generate_fn(prompt).strip()

    def critique(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> str:
        """Generate a critique of `response` against `principle`."""
        critique_prompt = principle.format_critique(prompt, response)
        raw = self._generate(critique_prompt)
        # Extract only the critique part (after "Critique:" if echoed)
        if "Critique:" in raw:
            raw = raw.split("Critique:", 1)[-1].strip()
        return raw

    def revise(
        self,
        prompt: str,
        initial_response: Optional[str] = None,
    ) -> Dict:
        """Run the full critique-revise loop.

        Args:
            prompt:           The original user prompt.
            initial_response: Pre-generated response. If None, generates one first.

        Returns:
            dict with keys:
                original    — the initial response
                revised     — the final revised response
                critique    — the last critique applied
                history     — list of (principle, critique, revision) tuples
        """
        if initial_response is None:
            initial_response = self._generate(prompt)

        current = initial_response
        history: List[Tuple[str, str, str]] = []

        for principle in self.config.get_principles():
            for _ in range(self.config.n_revisions):
                critique_text = self.critique(prompt, current, principle)

                revision_prompt = principle.format_revision(prompt, current, critique_text)
                revised = self._generate(revision_prompt)
                # Strip "Revised response:" prefix if echoed
                if "Revised response:" in revised:
                    revised = revised.split("Revised response:", 1)[-1].strip()

                history.append((principle.name, critique_text, revised))

                if self.config.chain_revisions:
                    current = revised  # Use revised output as input for next cycle

        return {
            "original":  initial_response,
            "revised":   current,
            "critique":  history[-1][1] if history else "",
            "history":   history,
        }

    def generate_preference_pairs(
        self,
        prompts: List[str],
        n_revisions: int = 1,
    ) -> List[Dict[str, str]]:
        """Generate DPO-ready preference pairs from a list of prompts.

        For each prompt:
          - `rejected` = initial (unrevised) response
          - `chosen`   = revised response after constitutional critique

        Args:
            prompts:     List of user prompts.
            n_revisions: Override config.n_revisions for this call.

        Returns:
            List of dicts: {"prompt", "chosen", "rejected"}
        """
        original_n = self.config.n_revisions
        self.config.n_revisions = n_revisions

        pairs = []
        for prompt in prompts:
            try:
                result = self.revise(prompt)
                if result["original"] != result["revised"]:
                    pairs.append({
                        "prompt":   prompt,
                        "chosen":   result["revised"],
                        "rejected": result["original"],
                    })
            except Exception as e:
                print(f"[CAI] WARNING: failed on prompt {prompt[:40]!r}: {e}")

        self.config.n_revisions = original_n
        return pairs

    def self_score(
        self,
        prompt: str,
        response: str,
        principle: Optional[ConstitutionalPrinciple] = None,
    ) -> float:
        """Ask the model to rate a response on [0, 1] against a principle.

        Useful for auto-scoring OnlineLearner feedback without human labels.

        Returns:
            Float in [0, 1].  Higher = better response.
        """
        p = principle or HELPFULNESS
        score_prompt = (
            f"Human request: {prompt}\n\n"
            f"AI response: {response}\n\n"
            f"On a scale from 0.0 to 1.0, how well does this response satisfy: "
            f"{p.name}? "
            "Reply with a single number only."
        )
        raw = self._generate(score_prompt).strip()
        # Parse first float found
        match = re.search(r"\d+(?:\.\d+)?", raw)
        if match:
            try:
                score = float(match.group())
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        return 0.5  # Default neutral score if parsing fails
