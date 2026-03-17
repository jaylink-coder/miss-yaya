"""Embedding-based retriever with similarity search.

Computes embeddings for document chunks and queries, then performs
nearest-neighbor search to find relevant context for generation.
Supports both dense (embedding) and sparse (BM25) retrieval.
"""

import math
import re
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from collections import Counter

from src.rag.document_store import DocumentStore, DocumentChunk


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance score."""
    chunk: DocumentChunk
    score: float
    method: str = "dense"  # dense, sparse, hybrid


# ── Dense Retrieval (Embedding-based) ──────────────────────────

class EmbeddingModel:
    """Interface for computing text embeddings.

    Default implementation uses a simple bag-of-words TF-IDF approach.
    For production, swap with a real embedding model (e.g., sentence-transformers).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts: List[str]):
        """Build vocabulary and IDF from a corpus."""
        self._doc_count = len(texts)
        df = Counter()

        for text in texts:
            words = set(self._tokenize(text))
            for w in words:
                df[w] += 1
                if w not in self._vocab:
                    self._vocab[w] = len(self._vocab) % self.dim

        # Compute IDF
        for word, count in df.items():
            self._idf[word] = math.log((self._doc_count + 1) / (count + 1)) + 1

    def embed(self, text: str) -> List[float]:
        """Compute embedding vector for text.

        Uses TF-IDF weighted bag-of-words projected to fixed dimension.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        tf = Counter(tokens)
        vector = [0.0] * self.dim

        for word, count in tf.items():
            if word in self._vocab:
                idx = self._vocab[word]
                tfidf = (count / len(tokens)) * self._idf.get(word, 1.0)
                vector[idx] += tfidf

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        return [self.embed(text) for text in texts]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseRetriever:
    """Dense retrieval using embedding similarity.

    Computes query and document embeddings, then finds nearest
    neighbors using cosine similarity.
    """

    def __init__(
        self,
        store: DocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        """Initialize dense retriever.

        Args:
            store: Document store with chunks.
            embedding_model: Embedding model (uses default if None).
            embed_fn: Custom embedding function (overrides model).
        """
        self.store = store
        self.model = embedding_model or EmbeddingModel()
        self._embed_fn = embed_fn

    def _embed(self, text: str) -> List[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        return self.model.embed(text)

    def index(self):
        """Compute and store embeddings for all chunks."""
        chunks = self.store.get_all_chunks()
        if not chunks:
            return

        # Build vocab from all chunk texts
        texts = [c.text for c in chunks]
        if isinstance(self.model, EmbeddingModel):
            self.model.build_vocab(texts)

        # Compute embeddings
        for chunk in chunks:
            embedding = self._embed(chunk.text)
            self.store.set_embedding(chunk.chunk_id, embedding)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve most relevant chunks for a query.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of RetrievalResult sorted by relevance (descending).
        """
        query_embedding = self._embed(query)

        results = []
        for chunk in self.store.get_all_chunks():
            if chunk.embedding is None:
                continue

            score = cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                results.append(RetrievalResult(
                    chunk=chunk, score=score, method="dense"
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ── Sparse Retrieval (BM25) ───────────────────────────────────

class BM25Retriever:
    """BM25 sparse retrieval for keyword-based search.

    Implements the Okapi BM25 ranking function for term-based
    document retrieval. Complements dense retrieval.
    """

    def __init__(
        self,
        store: DocumentStore,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.store = store
        self.k1 = k1
        self.b = b
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lens: Dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._doc_term_freqs: Dict[str, Counter] = {}
        self._n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index(self):
        """Build BM25 index from all chunks in the store."""
        chunks = self.store.get_all_chunks()
        self._n_docs = len(chunks)

        total_len = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._doc_lens[chunk.chunk_id] = len(tokens)
            self._doc_term_freqs[chunk.chunk_id] = Counter(tokens)
            total_len += len(tokens)

            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avg_dl = total_len / max(self._n_docs, 1)

    def _score_document(self, query_terms: List[str], chunk_id: str) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        dl = self._doc_lens.get(chunk_id, 0)
        tf_map = self._doc_term_freqs.get(chunk_id, Counter())

        for term in query_terms:
            if term not in self._doc_freqs:
                continue

            df = self._doc_freqs[term]
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))

            score += idf * (numerator / max(denominator, 1e-8))

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve chunks using BM25 scoring."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        results = []
        for chunk in self.store.get_all_chunks():
            score = self._score_document(query_terms, chunk.chunk_id)
            if score > min_score:
                results.append(RetrievalResult(
                    chunk=chunk, score=score, method="sparse"
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ── Hybrid Retrieval ───────────────────────────────────────────

class HybridRetriever:
    """Combine dense and sparse retrieval with score fusion.

    Uses Reciprocal Rank Fusion (RRF) to merge results from
    both dense (embedding) and sparse (BM25) retrievers.
    """

    def __init__(
        self,
        store: DocumentStore,
        embedding_model: Optional[EmbeddingModel] = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.dense = DenseRetriever(store, embedding_model)
        self.sparse = BM25Retriever(store)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def index(self):
        """Build both dense and sparse indices."""
        self.dense.index()
        self.sparse.index()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid dense+sparse with RRF fusion.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            Fused and re-ranked results.
        """
        # Get results from both retrievers (fetch more than top_k for fusion)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        for rank, result in enumerate(dense_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.dense_weight / (self.rrf_k + rank + 1)
            chunk_map[cid] = result.chunk

        for rank, result in enumerate(sparse_results):
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.sparse_weight / (self.rrf_k + rank + 1)
            chunk_map[cid] = result.chunk

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for cid, score in ranked[:top_k]:
            results.append(RetrievalResult(
                chunk=chunk_map[cid], score=score, method="hybrid"
            ))

        return results
