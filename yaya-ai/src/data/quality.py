"""Data quality scoring, language detection, PII removal, and MinHash dedup.

Production-grade filtering components for the Yaya data engine.
Extends the basic processing.py with more sophisticated methods.
"""

import re
import math
import hashlib
import struct
from typing import List, Set, Optional, Dict, Tuple, Any
from dataclasses import dataclass


# ── Language Detection ─────────────────────────────────────────

class LanguageDetector:
    """Detect document language using character n-gram heuristics.

    Uses a lightweight approach based on common word frequency
    for the most common languages. For production, consider
    fastText's language ID model (lid.176.bin).
    """

    # Top frequent words per language (small but effective for detection)
    LANGUAGE_WORDS = {
        "en": {"the", "is", "and", "of", "to", "in", "a", "that", "it", "for",
               "was", "on", "are", "with", "as", "his", "they", "be", "at", "this",
               "have", "from", "not", "by", "but", "had", "or", "an", "were", "which"},
        "fr": {"le", "la", "de", "et", "les", "des", "en", "un", "une", "est",
               "que", "du", "dans", "qui", "au", "pas", "sur", "pour", "par", "ce"},
        "de": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
               "des", "auf", "ist", "ein", "dem", "nicht", "eine", "als", "auch", "es"},
        "es": {"de", "la", "el", "en", "y", "que", "los", "del", "las", "un",
               "por", "con", "una", "su", "para", "es", "al", "como", "se", "no"},
    }

    def __init__(self, target_language: str = "en"):
        self.target_language = target_language

    def detect(self, text: str) -> str:
        """Detect the language of a text string.

        Returns language code (en, fr, de, es, or 'unknown').
        """
        words = set(text.lower().split()[:200])  # Sample first 200 words
        scores = {}

        for lang, lang_words in self.LANGUAGE_WORDS.items():
            overlap = len(words & lang_words)
            scores[lang] = overlap

        if not scores or max(scores.values()) < 3:
            return "unknown"

        return max(scores, key=scores.get)

    def is_target_language(self, text: str) -> bool:
        """Check if text is in the target language."""
        return self.detect(text) == self.target_language

    def detect_with_fasttext(self, text: str, model_path: str) -> str:
        """Detect language using fastText model (higher accuracy).

        Requires: pip install fasttext
                  Download lid.176.bin from fastText

        Args:
            text: Text to classify.
            model_path: Path to fastText lid model.

        Returns:
            ISO 639-1 language code.
        """
        try:
            import fasttext
            model = fasttext.load_model(model_path)
            # fastText expects single line
            clean_text = text.replace("\n", " ")[:1000]
            predictions = model.predict(clean_text, k=1)
            # Output: (('__label__en',), array([0.99]))
            label = predictions[0][0].replace("__label__", "")
            return label
        except ImportError:
            return self.detect(text)


# ── MinHash LSH Near-Deduplication ─────────────────────────────

class MinHasher:
    """MinHash signatures for near-duplicate detection.

    Computes MinHash signatures from document n-grams, enabling
    approximate Jaccard similarity estimation.
    """

    def __init__(self, num_perm: int = 128, ngram_size: int = 5):
        """Initialize MinHasher.

        Args:
            num_perm: Number of hash permutations (higher = more accurate).
            ngram_size: Character n-gram size for shingling.
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        # Generate random hash parameters (a, b) for each permutation
        import random
        rng = random.Random(42)
        self._max_hash = (1 << 32) - 1
        self._mersenne_prime = (1 << 61) - 1
        self._a = [rng.randint(1, self._mersenne_prime - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, self._mersenne_prime - 1) for _ in range(num_perm)]

    def _get_shingles(self, text: str) -> Set[int]:
        """Extract character n-gram shingles as hashed integers."""
        text = text.lower().strip()
        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngram = text[i:i + self.ngram_size]
            h = struct.unpack("<I", hashlib.md5(ngram.encode("utf-8")).digest()[:4])[0]
            shingles.add(h)
        return shingles

    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature for a document.

        Args:
            text: Document text.

        Returns:
            List of `num_perm` hash values forming the signature.
        """
        shingles = self._get_shingles(text)
        if not shingles:
            return [self._max_hash] * self.num_perm

        signature = []
        for i in range(self.num_perm):
            min_hash = float("inf")
            a, b = self._a[i], self._b[i]
            for s in shingles:
                h = ((a * s + b) % self._mersenne_prime) & self._max_hash
                if h < min_hash:
                    min_hash = h
            signature.append(min_hash)

        return signature

    @staticmethod
    def estimate_similarity(sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have equal length")
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class LSHIndex:
    """Locality-Sensitive Hashing index for fast near-neighbor lookup.

    Divides MinHash signatures into bands; documents sharing a band
    are candidate duplicates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        """Initialize LSH index.

        Args:
            num_perm: Number of MinHash permutations.
            threshold: Jaccard similarity threshold for duplicate detection.
        """
        self.num_perm = num_perm
        self.threshold = threshold

        # Compute optimal bands and rows
        # b * r = num_perm, threshold ≈ (1/b)^(1/r)
        best_b, best_r = 1, num_perm
        best_err = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b != 0:
                continue
            r = num_perm // b
            est_threshold = (1.0 / b) ** (1.0 / r)
            err = abs(est_threshold - threshold)
            if err < best_err:
                best_err = err
                best_b, best_r = b, r

        self.bands = best_b
        self.rows = best_r
        self._buckets: List[Dict[int, List[str]]] = [
            {} for _ in range(self.bands)
        ]

    def insert(self, doc_id: str, signature: List[int]) -> bool:
        """Insert a document signature. Returns True if it's a near-duplicate.

        Args:
            doc_id: Unique document identifier.
            signature: MinHash signature.

        Returns:
            True if this document is a near-duplicate of an existing one.
        """
        is_dup = False

        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hash(tuple(signature[start:end]))

            bucket = self._buckets[band_idx]
            if band_hash in bucket:
                is_dup = True
            else:
                bucket[band_hash] = []
            bucket.setdefault(band_hash, []).append(doc_id)

        return is_dup

    @property
    def num_buckets(self) -> int:
        return sum(len(b) for b in self._buckets)


class NearDeduplicator:
    """Near-duplicate document removal using MinHash + LSH.

    More robust than exact hashing — catches paraphrases,
    reformatted text, and minor edits.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.8):
        self.hasher = MinHasher(num_perm=num_perm)
        self.lsh = LSHIndex(num_perm=num_perm, threshold=threshold)
        self._doc_count = 0
        self._dup_count = 0

    def is_duplicate(self, text: str) -> bool:
        """Check if a document is a near-duplicate."""
        self._doc_count += 1
        sig = self.hasher.compute_signature(text)
        is_dup = self.lsh.insert(f"doc_{self._doc_count}", sig)
        if is_dup:
            self._dup_count += 1
        return is_dup

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_docs": self._doc_count,
            "duplicates": self._dup_count,
            "unique": self._doc_count - self._dup_count,
            "dup_rate": self._dup_count / max(self._doc_count, 1),
        }


# ── Quality Scoring ────────────────────────────────────────────

class QualityScorer:
    """Score document quality using heuristic features.

    Assigns a 0-1 quality score based on:
    - Text length and structure
    - Vocabulary diversity
    - Sentence structure
    - Presence of boilerplate patterns
    """

    # Common boilerplate patterns
    BOILERPLATE_PATTERNS = [
        re.compile(r"cookie", re.IGNORECASE),
        re.compile(r"subscribe to our newsletter", re.IGNORECASE),
        re.compile(r"click here", re.IGNORECASE),
        re.compile(r"terms of service", re.IGNORECASE),
        re.compile(r"privacy policy", re.IGNORECASE),
        re.compile(r"all rights reserved", re.IGNORECASE),
        re.compile(r"©\s*\d{4}", re.IGNORECASE),
        re.compile(r"share on (facebook|twitter|linkedin)", re.IGNORECASE),
        re.compile(r"log ?in|sign ?up|register", re.IGNORECASE),
    ]

    def score(self, text: str) -> float:
        """Compute quality score for a document.

        Returns:
            Float between 0.0 (low quality) and 1.0 (high quality).
        """
        if not text or len(text) < 50:
            return 0.0

        scores = []

        # 1. Length score (prefer 200-10000 chars)
        length = len(text)
        if length < 200:
            scores.append(length / 200)
        elif length > 100000:
            scores.append(max(0.3, 1.0 - (length - 100000) / 900000))
        else:
            scores.append(1.0)

        # 2. Vocabulary diversity (type-token ratio on first 500 words)
        words = text.lower().split()[:500]
        if words:
            ttr = len(set(words)) / len(words)
            scores.append(min(1.0, ttr * 2))  # TTR of 0.5+ is good
        else:
            scores.append(0.0)

        # 3. Sentence structure (avg words per sentence)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 8 <= avg_sent_len <= 40:
                scores.append(1.0)
            elif avg_sent_len < 5 or avg_sent_len > 80:
                scores.append(0.3)
            else:
                scores.append(0.7)
        else:
            scores.append(0.2)

        # 4. Boilerplate penalty
        boilerplate_hits = sum(
            1 for p in self.BOILERPLATE_PATTERNS if p.search(text[:2000])
        )
        boilerplate_score = max(0.0, 1.0 - boilerplate_hits * 0.15)
        scores.append(boilerplate_score)

        # 5. Alphabetic ratio
        alpha = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha / len(text)
        scores.append(min(1.0, alpha_ratio * 1.5))

        return sum(scores) / len(scores)


# ── PII Removal ────────────────────────────────────────────────

class PIIRemover:
    """Remove personally identifiable information from text.

    Detects and masks:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    """

    PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "phone_us": re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        ),
        "ssn": re.compile(
            r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"
        ),
        "credit_card": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
        "ip_address": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    }

    REPLACEMENTS = {
        "email": "[EMAIL]",
        "phone_us": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CREDIT_CARD]",
        "ip_address": "[IP_ADDRESS]",
    }

    def __init__(self, categories: Optional[List[str]] = None):
        """Initialize PII remover.

        Args:
            categories: PII categories to remove. None = all.
        """
        self.categories = categories or list(self.PATTERNS.keys())
        self._counts: Dict[str, int] = {c: 0 for c in self.categories}

    def remove(self, text: str) -> str:
        """Remove PII from text, replacing with placeholder tokens."""
        for category in self.categories:
            pattern = self.PATTERNS[category]
            replacement = self.REPLACEMENTS[category]
            matches = pattern.findall(text)
            self._counts[category] += len(matches)
            text = pattern.sub(replacement, text)
        return text

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._counts)


# ── Integrated Pipeline ────────────────────────────────────────

@dataclass
class FilterStats:
    """Track filtering statistics across the pipeline."""
    total_input: int = 0
    passed_language: int = 0
    passed_quality: int = 0
    passed_dedup: int = 0
    passed_length: int = 0
    final_output: int = 0

    def summary(self) -> Dict[str, Any]:
        total = max(self.total_input, 1)
        return {
            "total_input": self.total_input,
            "final_output": self.final_output,
            "pass_rate": self.final_output / total,
            "rejected_language": self.total_input - self.passed_language,
            "rejected_quality": self.passed_language - self.passed_quality,
            "rejected_dedup": self.passed_quality - self.passed_dedup,
            "rejected_length": self.passed_dedup - self.passed_length,
        }


class DataQualityPipeline:
    """Complete data quality pipeline combining all filtering stages.

    Pipeline order:
    1. Language detection
    2. PII removal
    3. Quality scoring
    4. Near-deduplication
    5. Length filtering
    """

    def __init__(
        self,
        language: str = "en",
        quality_threshold: float = 0.5,
        dedup_threshold: float = 0.8,
        min_doc_length: int = 100,
        max_doc_length: int = 1000000,
        remove_pii: bool = True,
        minhash_num_perm: int = 128,
    ):
        self.lang_detector = LanguageDetector(target_language=language)
        self.quality_scorer = QualityScorer()
        self.near_dedup = NearDeduplicator(
            num_perm=minhash_num_perm, threshold=dedup_threshold
        )
        self.pii_remover = PIIRemover() if remove_pii else None
        self.quality_threshold = quality_threshold
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.stats = FilterStats()

    def process_document(self, text: str) -> Optional[str]:
        """Run a single document through the full quality pipeline.

        Returns:
            Cleaned text if it passes all filters, None otherwise.
        """
        self.stats.total_input += 1

        # 1. Language check
        if not self.lang_detector.is_target_language(text):
            return None
        self.stats.passed_language += 1

        # 2. PII removal
        if self.pii_remover:
            text = self.pii_remover.remove(text)

        # 3. Quality scoring
        score = self.quality_scorer.score(text)
        if score < self.quality_threshold:
            return None
        self.stats.passed_quality += 1

        # 4. Near-deduplication
        if self.near_dedup.is_duplicate(text):
            return None
        self.stats.passed_dedup += 1

        # 5. Length filtering
        if len(text) < self.min_doc_length or len(text) > self.max_doc_length:
            return None
        self.stats.passed_length += 1

        self.stats.final_output += 1
        return text

    def get_stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        result = self.stats.summary()
        result["dedup_stats"] = self.near_dedup.stats
        if self.pii_remover:
            result["pii_stats"] = self.pii_remover.stats
        return result
