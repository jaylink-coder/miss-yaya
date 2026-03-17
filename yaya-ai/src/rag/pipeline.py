"""RAG Pipeline — retrieve context, augment prompt, generate response.

Ties together the document store, retriever, and generation model
into a complete retrieval-augmented generation pipeline.
"""

import json
from typing import List, Dict, Optional, Any, Callable

from src.rag.document_store import DocumentStore, DocumentChunk, TextChunker
from src.rag.retriever import (
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    RetrievalResult,
    EmbeddingModel,
)


class ContextBuilder:
    """Build augmented prompts from retrieved context.

    Formats retrieved chunks into a context block that gets
    prepended to the user query before generation.
    """

    def __init__(
        self,
        max_context_tokens: int = 2048,
        include_metadata: bool = True,
        context_header: str = "The following information may be relevant to answering the question:\n",
        context_footer: str = "\nUsing the above context, please answer the following question:\n",
        source_format: str = "[Source: {source}]",
    ):
        self.max_context_tokens = max_context_tokens
        self.include_metadata = include_metadata
        self.context_header = context_header
        self.context_footer = context_footer
        self.source_format = source_format

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def build_context(
        self,
        results: List[RetrievalResult],
        query: str,
    ) -> str:
        """Build a context-augmented prompt from retrieval results.

        Args:
            results: Ranked retrieval results.
            query: Original user query.

        Returns:
            Augmented prompt string.
        """
        if not results:
            return query

        context_parts = [self.context_header]
        token_budget = self.max_context_tokens
        token_budget -= self._estimate_tokens(self.context_header)
        token_budget -= self._estimate_tokens(self.context_footer)
        token_budget -= self._estimate_tokens(query)

        for i, result in enumerate(results):
            chunk_text = result.chunk.text
            chunk_tokens = self._estimate_tokens(chunk_text)

            if chunk_tokens > token_budget:
                # Truncate to fit
                words = chunk_text.split()
                max_words = int(token_budget / 1.3)
                if max_words > 20:
                    chunk_text = " ".join(words[:max_words]) + "..."
                else:
                    break

            source = result.chunk.metadata.get("source", "")
            if self.include_metadata and source:
                source_line = self.source_format.format(source=source)
                context_parts.append(f"\n--- Document {i+1} {source_line} ---")
            else:
                context_parts.append(f"\n--- Document {i+1} ---")

            context_parts.append(chunk_text)
            token_budget -= self._estimate_tokens(chunk_text) + 10  # overhead

            if token_budget <= 0:
                break

        context_parts.append(self.context_footer)
        context_parts.append(query)

        return "\n".join(context_parts)

    def build_messages(
        self,
        results: List[RetrievalResult],
        query: str,
        system_prompt: str = "You are a helpful assistant. Use the provided context to answer questions accurately.",
    ) -> List[Dict[str, str]]:
        """Build chat messages with context for the chat template.

        Args:
            results: Retrieval results.
            query: User query.
            system_prompt: System message.

        Returns:
            List of message dicts.
        """
        context = self.build_context(results, "")
        return [
            {"role": "system", "content": system_prompt + "\n\n" + context},
            {"role": "user", "content": query},
        ]


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline.

    Usage:
        store = DocumentStore()
        store.add_document("Einstein developed the theory of relativity...")
        store.add_document("Quantum mechanics describes nature at atomic scales...")

        rag = RAGPipeline(store=store, generate_fn=my_model.generate)
        rag.index()

        answer = rag.query("What did Einstein develop?")
    """

    def __init__(
        self,
        store: DocumentStore,
        generate_fn: Optional[Callable[[str], str]] = None,
        retriever_type: str = "hybrid",
        embedding_model: Optional[EmbeddingModel] = None,
        top_k: int = 5,
        max_context_tokens: int = 2048,
    ):
        """Initialize RAG pipeline.

        Args:
            store: Document store with ingested documents.
            generate_fn: Text generation function (model wrapper).
            retriever_type: 'dense', 'sparse', or 'hybrid'.
            embedding_model: Custom embedding model.
            top_k: Number of chunks to retrieve.
            max_context_tokens: Max tokens for context window.
        """
        self.store = store
        self.generate_fn = generate_fn
        self.top_k = top_k

        # Initialize retriever
        if retriever_type == "dense":
            self.retriever = DenseRetriever(store, embedding_model)
        elif retriever_type == "sparse":
            self.retriever = BM25Retriever(store)
        elif retriever_type == "hybrid":
            self.retriever = HybridRetriever(store, embedding_model)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        self.context_builder = ContextBuilder(
            max_context_tokens=max_context_tokens,
        )

        self._indexed = False

    def index(self):
        """Build the retrieval index. Must be called after adding documents."""
        self.retriever.index()
        self._indexed = True

    def add_document(self, text: str, metadata: Optional[Dict] = None, source: str = ""):
        """Add a document and re-index."""
        self.store.add_document(text, metadata, source)
        self._indexed = False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query.
            top_k: Override default top_k.

        Returns:
            List of RetrievalResult.
        """
        if not self._indexed:
            self.index()

        k = top_k or self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """Full RAG query: retrieve context, build prompt, generate answer.

        Args:
            question: User question.
            top_k: Number of chunks to retrieve.
            return_context: Include retrieved context in result.

        Returns:
            Dict with 'answer', 'sources', and optionally 'context'.
        """
        # Retrieve
        results = self.retrieve(question, top_k)

        # Build augmented prompt
        augmented_prompt = self.context_builder.build_context(results, question)

        # Generate
        if self.generate_fn:
            answer = self.generate_fn(augmented_prompt)
        else:
            answer = f"[No generation model configured. Retrieved {len(results)} chunks.]"

        # Collect sources
        sources = []
        for r in results:
            sources.append({
                "chunk_id": r.chunk.chunk_id,
                "score": round(r.score, 4),
                "text_preview": r.chunk.text[:200],
                "source": r.chunk.metadata.get("source", ""),
                "method": r.method,
            })

        response = {
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": len(results),
        }

        if return_context:
            response["context"] = augmented_prompt

        return response

    def query_with_messages(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Build chat messages with RAG context for the chat template.

        Returns messages suitable for ChatTemplate.from_messages().
        """
        results = self.retrieve(question, top_k)
        prompt = system_prompt or "You are a helpful assistant. Answer based on the provided context."
        return self.context_builder.build_messages(results, question, prompt)

    def stats(self) -> Dict[str, Any]:
        """Return pipeline statistics."""
        return {
            "store": self.store.stats(),
            "indexed": self._indexed,
            "retriever_type": type(self.retriever).__name__,
            "top_k": self.top_k,
        }
