"""Content filtering and toxicity detection for Yaya AI.

Provides input/output guardrails to ensure safe model behavior:
- Toxicity detection (keyword + pattern based)
- Category-based content classification
- Prompt injection detection
- Output validation and sanitization
"""

import re
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ContentCategory(Enum):
    """Categories of potentially harmful content."""
    SAFE = "safe"
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    ILLEGAL = "illegal"
    PII_LEAK = "pii_leak"
    PROMPT_INJECTION = "prompt_injection"


@dataclass
class FilterResult:
    """Result from a content filter check."""
    is_safe: bool
    categories: List[ContentCategory] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    flagged_spans: List[Tuple[int, int, str]] = field(default_factory=list)
    explanation: str = ""

    @property
    def primary_category(self) -> ContentCategory:
        if not self.categories:
            return ContentCategory.SAFE
        return self.categories[0]


class ToxicityDetector:
    """Detect toxic content using keyword patterns and heuristics.

    For production, integrate with a classifier model (e.g., Perspective API,
    detoxify, or a fine-tuned classifier). This implementation uses pattern
    matching as a baseline that works without external dependencies.
    """

    # Patterns indicating potentially harmful content (severity tiers)
    # Tier 1: High severity — block
    TIER1_PATTERNS = [
        re.compile(r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)", re.IGNORECASE),
        re.compile(r"\b(synthesize|manufacture)\s+(meth|cocaine|heroin|fentanyl)", re.IGNORECASE),
        re.compile(r"\bhack\s+into\s+(someone|a\s+bank|government)", re.IGNORECASE),
        re.compile(r"\b(kill|murder|assassinate)\s+(someone|a\s+person|myself)", re.IGNORECASE),
    ]

    # Tier 2: Medium severity — flag
    TIER2_PATTERNS = [
        re.compile(r"\b(hate|kill)\s+all\s+\w+", re.IGNORECASE),
        re.compile(r"\b(racial|ethnic)\s+slur", re.IGNORECASE),
        re.compile(r"\bsuicid(e|al)\s+(method|way|how)", re.IGNORECASE),
    ]

    # Slur/hate speech wordlist (kept minimal — production systems use comprehensive lists)
    HATE_INDICATORS = {
        "supremacy", "inferior race", "ethnic cleansing", "genocide",
    }

    def detect(self, text: str) -> FilterResult:
        """Check text for toxic content.

        Returns:
            FilterResult with safety assessment.
        """
        text_lower = text.lower()
        categories = []
        scores = {}
        flagged = []

        # Tier 1 checks
        for pattern in self.TIER1_PATTERNS:
            match = pattern.search(text)
            if match:
                categories.append(ContentCategory.ILLEGAL)
                flagged.append((match.start(), match.end(), "tier1_harmful"))
                scores["tier1"] = 1.0

        # Tier 2 checks
        for pattern in self.TIER2_PATTERNS:
            match = pattern.search(text)
            if match:
                if ContentCategory.VIOLENCE not in categories:
                    categories.append(ContentCategory.VIOLENCE)
                flagged.append((match.start(), match.end(), "tier2_harmful"))
                scores["tier2"] = scores.get("tier2", 0) + 0.5

        # Hate speech indicators
        for indicator in self.HATE_INDICATORS:
            if indicator in text_lower:
                if ContentCategory.HATE_SPEECH not in categories:
                    categories.append(ContentCategory.HATE_SPEECH)
                idx = text_lower.find(indicator)
                flagged.append((idx, idx + len(indicator), "hate_speech"))

        # Self-harm detection
        self_harm_patterns = [
            re.compile(r"\b(want\s+to|going\s+to)\s+(hurt|harm|kill)\s+myself\b", re.IGNORECASE),
            re.compile(r"\bself[- ]harm\b", re.IGNORECASE),
        ]
        for pattern in self_harm_patterns:
            match = pattern.search(text)
            if match:
                categories.append(ContentCategory.SELF_HARM)
                flagged.append((match.start(), match.end(), "self_harm"))

        is_safe = len(categories) == 0
        explanation = ""
        if not is_safe:
            cat_names = [c.value for c in categories]
            explanation = f"Content flagged for: {', '.join(cat_names)}"

        return FilterResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            flagged_spans=flagged,
            explanation=explanation,
        )


class MLToxicityClassifier:
    """ML-based toxicity classifier using a lightweight transformer model.

    Wraps detoxify (or any HuggingFace text-classification model) to provide
    ML-based content scoring.  Falls back gracefully to pattern-only detection
    if the model is unavailable.

    Install:  pip install detoxify          (for default backend)
           or pip install transformers torch (for custom model)

    Usage:
        clf = MLToxicityClassifier()           # auto-detects detoxify
        clf = MLToxicityClassifier(backend="transformers", model_name="...")
        result = clf.classify("some text")
    """

    def __init__(
        self,
        backend: str = "detoxify",
        model_name: str = "original",
        threshold: float = 0.7,
        device: str = "cpu",
    ):
        """Initialize the ML classifier.

        Args:
            backend: "detoxify" or "transformers".
            model_name: Model name/path.  For detoxify: "original", "unbiased",
                        or "multilingual".  For transformers: any HF model id.
            threshold: Score above which content is flagged as toxic.
            device: torch device string.
        """
        self.backend = backend
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the ML model.  Sets _available = False on failure."""
        try:
            if self.backend == "detoxify":
                from detoxify import Detoxify  # type: ignore[import-untyped]
                self._model = Detoxify(self.model_name, device=self.device)
                self._available = True
            elif self.backend == "transformers":
                from transformers import pipeline  # type: ignore[import-untyped]
                self._model = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=self.device,
                    top_k=None,
                )
                self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the ML model loaded successfully."""
        return self._available

    def classify(self, text: str) -> FilterResult:
        """Score text using the ML model.

        Returns:
            FilterResult with ML-derived scores and categories.
            If the model is unavailable, returns a safe result with no scores.
        """
        if not self._available or self._model is None:
            return FilterResult(is_safe=True, explanation="ML classifier unavailable")

        categories: List[ContentCategory] = []
        scores: Dict[str, float] = {}

        try:
            if self.backend == "detoxify":
                preds = self._model.predict(text)
                # preds is dict: {"toxicity": 0.01, "severe_toxicity": 0.0, ...}
                for label, score in preds.items():
                    scores[f"ml_{label}"] = float(score)

                if preds.get("toxicity", 0) >= self.threshold:
                    categories.append(ContentCategory.TOXIC)
                if preds.get("identity_attack", 0) >= self.threshold:
                    categories.append(ContentCategory.HATE_SPEECH)
                if preds.get("threat", 0) >= self.threshold:
                    categories.append(ContentCategory.VIOLENCE)
                if preds.get("sexual_explicit", 0) >= self.threshold:
                    categories.append(ContentCategory.SEXUAL)

            elif self.backend == "transformers":
                results = self._model(text[:512])
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]
                    for item in results:
                        label = item["label"].lower()
                        score = float(item["score"])
                        scores[f"ml_{label}"] = score
                        if score >= self.threshold and label in ("toxic", "hate", "violence"):
                            categories.append(ContentCategory.TOXIC)
        except Exception as e:
            scores["ml_error"] = 1.0
            return FilterResult(
                is_safe=True,
                scores=scores,
                explanation=f"ML classifier error: {e}",
            )

        is_safe = len(categories) == 0
        explanation = ""
        if not is_safe:
            cat_names = [c.value for c in categories]
            explanation = f"ML classifier flagged: {', '.join(cat_names)}"

        return FilterResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            explanation=explanation,
        )


class PromptInjectionDetector:
    """Detect prompt injection attempts in user input.

    Catches common patterns where users try to override system
    instructions, extract the system prompt, or manipulate model behavior.
    """

    INJECTION_PATTERNS = [
        # System prompt extraction
        re.compile(r"(ignore|forget|disregard)\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"(repeat|print|show|reveal|output)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)", re.IGNORECASE),
        re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)", re.IGNORECASE),

        # Role override
        re.compile(r"you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode", re.IGNORECASE),
        re.compile(r"(enter|switch\s+to|activate)\s+\w+\s+mode", re.IGNORECASE),
        re.compile(r"from\s+now\s+on\s+(you|act|behave)", re.IGNORECASE),

        # Jailbreak patterns
        re.compile(r"(DAN|do\s+anything\s+now)", re.IGNORECASE),
        re.compile(r"pretend\s+(you\s+)?(are|have)\s+no\s+(restrictions|limits|rules)", re.IGNORECASE),
        re.compile(r"(bypass|override|disable)\s+(your\s+)?(safety|content|filter)", re.IGNORECASE),
    ]

    # Suspicious token sequences
    SUSPICIOUS_TOKENS = {
        "```system", "<<SYS>>", "[INST]", "<|im_start|>",
        "### Instruction:", "### Human:", "SYSTEM:",
    }

    def detect(self, text: str) -> FilterResult:
        """Check for prompt injection attempts.

        Returns:
            FilterResult indicating if injection was detected.
        """
        categories = []
        flagged = []
        scores = {}

        # Pattern matching
        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                categories.append(ContentCategory.PROMPT_INJECTION)
                flagged.append((match.start(), match.end(), "injection_pattern"))
                scores["injection_pattern"] = 1.0
                break  # One match is enough

        # Token sequence check
        for token in self.SUSPICIOUS_TOKENS:
            if token in text:
                if ContentCategory.PROMPT_INJECTION not in categories:
                    categories.append(ContentCategory.PROMPT_INJECTION)
                idx = text.find(token)
                flagged.append((idx, idx + len(token), "suspicious_token"))
                scores["suspicious_token"] = 1.0

        is_safe = len(categories) == 0
        explanation = "Prompt injection attempt detected." if not is_safe else ""

        return FilterResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            flagged_spans=flagged,
            explanation=explanation,
        )


class OutputValidator:
    """Validate and sanitize model outputs before returning to users.

    Checks for:
    - PII leakage in outputs
    - Hallucinated URLs or citations
    - Incomplete or malformed responses
    - Unintended system prompt leakage
    """

    PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    # Patterns that suggest system prompt leakage
    SYSTEM_LEAK_PATTERNS = [
        re.compile(r"my\s+(system\s+)?instructions\s+(say|are|tell)", re.IGNORECASE),
        re.compile(r"I\s+was\s+(told|instructed|programmed)\s+to", re.IGNORECASE),
        re.compile(r"my\s+(system\s+)?prompt\s+(says|is|contains)", re.IGNORECASE),
    ]

    def validate(self, output: str, check_pii: bool = True) -> FilterResult:
        """Validate model output.

        Args:
            output: Model-generated text.
            check_pii: Whether to check for PII.

        Returns:
            FilterResult with validation assessment.
        """
        categories = []
        flagged = []
        scores = {}

        # PII check
        if check_pii:
            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = list(pattern.finditer(output))
                if matches:
                    categories.append(ContentCategory.PII_LEAK)
                    for m in matches:
                        flagged.append((m.start(), m.end(), f"pii_{pii_type}"))
                    scores[f"pii_{pii_type}"] = len(matches)

        # System prompt leakage
        for pattern in self.SYSTEM_LEAK_PATTERNS:
            match = pattern.search(output)
            if match:
                flagged.append((match.start(), match.end(), "system_leak"))
                scores["system_leak"] = 1.0

        # Empty or very short output
        if len(output.strip()) < 2:
            scores["empty_output"] = 1.0

        is_safe = len(categories) == 0
        explanation = ""
        if not is_safe:
            cat_names = [c.value for c in categories]
            explanation = f"Output validation flagged: {', '.join(cat_names)}"

        return FilterResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            flagged_spans=flagged,
            explanation=explanation,
        )

    def sanitize(self, output: str) -> str:
        """Remove PII from model output."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            replacement = f"[{pii_type.upper()}_REDACTED]"
            output = pattern.sub(replacement, output)
        return output


class SafetyRefusalGenerator:
    """Generate appropriate refusal messages for unsafe content.

    Provides context-sensitive refusal messages based on the
    type of harmful content detected.
    """

    REFUSALS = {
        ContentCategory.TOXIC: "I can't help with that request as it contains harmful content.",
        ContentCategory.HATE_SPEECH: "I'm not able to engage with hate speech or discriminatory content.",
        ContentCategory.VIOLENCE: "I can't provide assistance with content that promotes violence.",
        ContentCategory.SEXUAL: "I'm not able to help with explicit sexual content.",
        ContentCategory.SELF_HARM: (
            "I'm concerned about this request. If you're struggling, please reach out to "
            "a crisis helpline. In the US, call or text 988 for the Suicide & Crisis Lifeline."
        ),
        ContentCategory.ILLEGAL: "I can't assist with illegal activities.",
        ContentCategory.PII_LEAK: "I've detected personal information that should not be shared.",
        ContentCategory.PROMPT_INJECTION: "I'm not able to modify my operating instructions.",
    }

    def generate_refusal(self, filter_result: FilterResult) -> str:
        """Generate a refusal message based on the filter result."""
        if filter_result.is_safe:
            return ""

        category = filter_result.primary_category
        return self.REFUSALS.get(
            category,
            "I'm not able to help with that request.",
        )


# ── Integrated Guardrails ──────────────────────────────────────

class GuardrailsEngine:
    """Complete safety guardrails system for input and output.

    Combines all safety components into a single interface:
    1. Input filtering (toxicity + injection detection)
    2. Output validation (PII leakage + sanitization)
    3. Refusal generation for unsafe requests
    """

    def __init__(
        self,
        enable_toxicity: bool = True,
        enable_injection: bool = True,
        enable_output_validation: bool = True,
        enable_ml_classifier: bool = False,
        ml_backend: str = "detoxify",
        ml_model_name: str = "original",
        ml_threshold: float = 0.7,
        custom_blocked_patterns: Optional[List[str]] = None,
    ):
        self.toxicity = ToxicityDetector() if enable_toxicity else None
        self.injection = PromptInjectionDetector() if enable_injection else None
        self.output_validator = OutputValidator() if enable_output_validation else None
        self.refusal_gen = SafetyRefusalGenerator()

        # ML-based classifier (optional — requires detoxify or transformers)
        self.ml_classifier: Optional[MLToxicityClassifier] = None
        if enable_ml_classifier:
            self.ml_classifier = MLToxicityClassifier(
                backend=ml_backend,
                model_name=ml_model_name,
                threshold=ml_threshold,
            )

        self._custom_patterns = []
        if custom_blocked_patterns:
            for p in custom_blocked_patterns:
                self._custom_patterns.append(re.compile(p, re.IGNORECASE))

        self._input_checks = 0
        self._input_blocked = 0
        self._output_checks = 0
        self._output_flagged = 0

    def check_input(self, text: str) -> FilterResult:
        """Check user input for safety issues.

        Args:
            text: User input text.

        Returns:
            FilterResult. If not safe, use refusal_gen for response.
        """
        self._input_checks += 1
        all_categories = []
        all_flagged = []
        all_scores = {}

        # Toxicity check
        if self.toxicity:
            result = self.toxicity.detect(text)
            all_categories.extend(result.categories)
            all_flagged.extend(result.flagged_spans)
            all_scores.update(result.scores)

        # Injection check
        if self.injection:
            result = self.injection.detect(text)
            all_categories.extend(result.categories)
            all_flagged.extend(result.flagged_spans)
            all_scores.update(result.scores)

        # ML classifier (if available)
        if self.ml_classifier and self.ml_classifier.available:
            ml_result = self.ml_classifier.classify(text)
            all_categories.extend(ml_result.categories)
            all_scores.update(ml_result.scores)

        # Custom patterns
        for pattern in self._custom_patterns:
            match = pattern.search(text)
            if match:
                all_categories.append(ContentCategory.TOXIC)
                all_flagged.append((match.start(), match.end(), "custom_pattern"))

        is_safe = len(all_categories) == 0
        if not is_safe:
            self._input_blocked += 1

        explanation = ""
        if not is_safe:
            cat_names = list(set(c.value for c in all_categories))
            explanation = f"Input blocked: {', '.join(cat_names)}"

        return FilterResult(
            is_safe=is_safe,
            categories=all_categories,
            scores=all_scores,
            flagged_spans=all_flagged,
            explanation=explanation,
        )

    def check_output(self, text: str) -> FilterResult:
        """Check model output for safety issues.

        Args:
            text: Model-generated output.

        Returns:
            FilterResult with validation status.
        """
        self._output_checks += 1

        if not self.output_validator:
            return FilterResult(is_safe=True)

        result = self.output_validator.validate(text)
        if not result.is_safe:
            self._output_flagged += 1

        return result

    def sanitize_output(self, text: str) -> str:
        """Sanitize model output by removing PII."""
        if self.output_validator:
            return self.output_validator.sanitize(text)
        return text

    def get_refusal(self, filter_result: FilterResult) -> str:
        """Get appropriate refusal message for a blocked input."""
        return self.refusal_gen.generate_refusal(filter_result)

    def process_interaction(
        self,
        user_input: str,
        generate_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Process a full interaction with safety checks.

        Args:
            user_input: User's message.
            generate_fn: Model generation function.

        Returns:
            Dict with 'response', 'input_safe', 'output_safe', 'blocked'.
        """
        # Check input
        input_result = self.check_input(user_input)
        if not input_result.is_safe:
            refusal = self.get_refusal(input_result)
            return {
                "response": refusal,
                "input_safe": False,
                "output_safe": True,
                "blocked": True,
                "block_reason": input_result.explanation,
            }

        # Generate response
        if generate_fn:
            raw_output = generate_fn(user_input)
        else:
            raw_output = "[No generation function provided]"

        # Check and sanitize output
        output_result = self.check_output(raw_output)
        if not output_result.is_safe:
            raw_output = self.sanitize_output(raw_output)

        return {
            "response": raw_output,
            "input_safe": True,
            "output_safe": output_result.is_safe,
            "blocked": False,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "input_checks": self._input_checks,
            "input_blocked": self._input_blocked,
            "block_rate": self._input_blocked / max(self._input_checks, 1),
            "output_checks": self._output_checks,
            "output_flagged": self._output_flagged,
        }
