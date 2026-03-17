"""Document store with chunking, indexing, and metadata management.

Handles ingestion of documents (text, files, web pages), splits them
into chunks suitable for retrieval, and stores them with embeddings.
"""

import os
import json
import hashlib
import math
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class DocumentChunk:
    """A single chunk of a document with metadata."""
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_index: int = 0
    total_chunks: int = 1

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate (words * 1.3)."""
        return int(len(self.text.split()) * 1.3)


@dataclass
class Document:
    """A full document before chunking."""
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


class TextChunker:
    """Split documents into overlapping chunks for retrieval.

    Supports multiple chunking strategies:
    - Fixed size with overlap
    - Sentence-aware (splits on sentence boundaries)
    - Paragraph-aware (splits on paragraph boundaries)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "sentence",
        min_chunk_size: int = 50,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Target chunk size in tokens (estimated from words).
            chunk_overlap: Number of overlapping tokens between chunks.
            strategy: Chunking strategy: 'fixed', 'sentence', 'paragraph'.
            min_chunk_size: Minimum chunk size (discard smaller chunks).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def chunk_fixed(self, text: str) -> List[str]:
        """Split text into fixed-size word chunks with overlap."""
        words = text.split()
        # Convert token sizes to word sizes (approx)
        words_per_chunk = int(self.chunk_size / 1.3)
        overlap_words = int(self.chunk_overlap / 1.3)

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.split()) >= int(self.min_chunk_size / 1.3):
                chunks.append(chunk)
            start += words_per_chunk - overlap_words

        return chunks

    def chunk_sentence(self, text: str) -> List[str]:
        """Split text on sentence boundaries, respecting chunk size."""
        import re
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        words_per_chunk = int(self.chunk_size / 1.3)
        overlap_words = int(self.chunk_overlap / 1.3)

        for sentence in sentences:
            sent_words = len(sentence.split())

            if current_size + sent_words > words_per_chunk and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= int(self.min_chunk_size / 1.3):
                    chunks.append(chunk_text)

                # Keep overlap sentences
                overlap_size = 0
                overlap_start = len(current_chunk)
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_size += len(current_chunk[i].split())
                    if overlap_size >= overlap_words:
                        overlap_start = i
                        break
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sent_words

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= int(self.min_chunk_size / 1.3):
                chunks.append(chunk_text)

        return chunks

    def chunk_paragraph(self, text: str) -> List[str]:
        """Split text on paragraph boundaries."""
        paragraphs = text.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        words_per_chunk = int(self.chunk_size / 1.3)

        for para in paragraphs:
            para_words = len(para.split())

            if current_size + para_words > words_per_chunk and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_words

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text.split()) >= int(self.min_chunk_size / 1.3):
                chunks.append(chunk_text)

        return chunks

    def chunk(self, text: str) -> List[str]:
        """Split text using the configured strategy."""
        if self.strategy == "fixed":
            return self.chunk_fixed(text)
        elif self.strategy == "sentence":
            return self.chunk_sentence(text)
        elif self.strategy == "paragraph":
            return self.chunk_paragraph(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")


class DocumentStore:
    """In-memory document store with persistence.

    Stores document chunks with their embeddings for retrieval.
    Supports adding, removing, and searching documents.
    """

    def __init__(
        self,
        chunker: Optional[TextChunker] = None,
        persist_dir: Optional[str] = None,
    ):
        self.chunker = chunker or TextChunker()
        self.persist_dir = persist_dir
        self._chunks: Dict[str, DocumentChunk] = {}
        self._documents: Dict[str, Document] = {}

        if persist_dir and os.path.exists(os.path.join(persist_dir, "store.json")):
            self._load()

    def _generate_doc_id(self, text: str, source: str = "") -> str:
        """Generate a deterministic document ID."""
        content = f"{source}:{text[:500]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _generate_chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}_chunk_{index:04d}"

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "",
        document_id: Optional[str] = None,
    ) -> List[str]:
        """Add a document, chunk it, and store the chunks.

        Args:
            text: Document text.
            metadata: Optional metadata dict.
            source: Document source identifier.
            document_id: Optional custom document ID.

        Returns:
            List of chunk IDs created.
        """
        if not text or not text.strip():
            return []

        doc_id = document_id or self._generate_doc_id(text, source)
        doc = Document(
            document_id=doc_id,
            text=text,
            metadata=metadata or {},
            source=source,
        )
        self._documents[doc_id] = doc

        # Chunk the document
        chunk_texts = self.chunker.chunk(text)
        chunk_ids = []

        for i, chunk_text in enumerate(chunk_texts):
            chunk_id = self._generate_chunk_id(doc_id, i)
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                metadata={**(metadata or {}), "source": source},
                chunk_index=i,
                total_chunks=len(chunk_texts),
            )
            self._chunks[chunk_id] = chunk
            chunk_ids.append(chunk_id)

        return chunk_ids

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """Add multiple documents.

        Args:
            documents: List of dicts with 'text' and optional 'metadata', 'source'.

        Returns:
            Total number of chunks created.
        """
        total = 0
        for doc in documents:
            chunk_ids = self.add_document(
                text=doc.get("text", ""),
                metadata=doc.get("metadata"),
                source=doc.get("source", ""),
            )
            total += len(chunk_ids)
        return total

    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks.get(chunk_id)

    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        return sorted(
            [c for c in self._chunks.values() if c.document_id == document_id],
            key=lambda c: c.chunk_index,
        )

    def remove_document(self, document_id: str):
        """Remove a document and all its chunks."""
        self._documents.pop(document_id, None)
        to_remove = [
            cid for cid, c in self._chunks.items()
            if c.document_id == document_id
        ]
        for cid in to_remove:
            del self._chunks[cid]

    def set_embedding(self, chunk_id: str, embedding: List[float]):
        """Set the embedding vector for a chunk."""
        if chunk_id in self._chunks:
            self._chunks[chunk_id].embedding = embedding

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks in the store."""
        return list(self._chunks.values())

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_documents(self) -> int:
        return len(self._documents)

    def save(self):
        """Persist the store to disk."""
        if not self.persist_dir:
            return

        os.makedirs(self.persist_dir, exist_ok=True)
        data = {
            "documents": {
                doc_id: {"text": doc.text, "metadata": doc.metadata, "source": doc.source}
                for doc_id, doc in self._documents.items()
            },
            "chunks": {
                chunk_id: {
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                }
                for chunk_id, chunk in self._chunks.items()
            },
        }
        with open(os.path.join(self.persist_dir, "store.json"), "w") as f:
            json.dump(data, f)

    def _load(self):
        """Load persisted store from disk."""
        path = os.path.join(self.persist_dir, "store.json")
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            data = json.load(f)

        for doc_id, doc_data in data.get("documents", {}).items():
            self._documents[doc_id] = Document(
                document_id=doc_id, **doc_data
            )

        for chunk_id, chunk_data in data.get("chunks", {}).items():
            self._chunks[chunk_id] = DocumentChunk(
                chunk_id=chunk_id, **chunk_data
            )

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        embedded = sum(1 for c in self._chunks.values() if c.embedding is not None)
        return {
            "documents": self.num_documents,
            "chunks": self.num_chunks,
            "chunks_with_embeddings": embedded,
            "sources": list(set(d.source for d in self._documents.values() if d.source)),
        }
