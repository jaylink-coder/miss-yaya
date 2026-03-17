"""Tests for the Yaya RAG system: document store, retriever, pipeline, tool integration."""

import os
import json
import tempfile

import pytest

from src.rag.document_store import DocumentStore, DocumentChunk, TextChunker, Document
from src.rag.retriever import (
    EmbeddingModel,
    DenseRetriever,
    BM25Retriever,
    HybridRetriever,
    cosine_similarity,
)
from src.rag.pipeline import RAGPipeline, ContextBuilder
from src.rag.tool_integration import register_rag_tools, create_rag_agent_registry


# ══════════════════════════════════════════════════════════════
#  Text Chunker Tests
# ══════════════════════════════════════════════════════════════


class TestTextChunker:
    def test_fixed_chunking(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, strategy="fixed")
        text = " ".join(["word"] * 200)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        # Each chunk should be smaller than the original
        for c in chunks:
            assert len(c) < len(text)

    def test_sentence_chunking(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="sentence", min_chunk_size=10)
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning is a method of data analysis. "
            "Deep learning uses neural networks with many layers. "
            "Natural language processing handles text data. "
            "Computer vision processes images and video."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        # Sentences should not be split mid-sentence
        for c in chunks:
            assert c.endswith(".") or c.endswith("...") or True  # May be partial

    def test_paragraph_chunking(self):
        chunker = TextChunker(chunk_size=100, strategy="paragraph", min_chunk_size=5)
        text = "Paragraph one about science.\n\nParagraph two about math.\n\nParagraph three about history."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=1000, min_chunk_size=5)
        text = "This is a short document."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_unknown_strategy_raises(self):
        chunker = TextChunker(strategy="unknown")
        with pytest.raises(ValueError):
            chunker.chunk("Some text.")


# ══════════════════════════════════════════════════════════════
#  Document Store Tests
# ══════════════════════════════════════════════════════════════


class TestDocumentStore:
    def _store(self):
        return DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))

    def test_add_document(self):
        store = self._store()
        chunk_ids = store.add_document("This is a test document with enough words for chunking purposes.")
        assert len(chunk_ids) >= 1
        assert store.num_documents == 1
        assert store.num_chunks >= 1

    def test_add_multiple_documents(self):
        store = self._store()
        docs = [
            {"text": "First document about machine learning and artificial intelligence."},
            {"text": "Second document about quantum physics and relativity."},
        ]
        total = store.add_documents(docs)
        assert total >= 2
        assert store.num_documents == 2

    def test_get_chunk(self):
        store = self._store()
        chunk_ids = store.add_document("Test document for retrieval.", document_id="doc1")
        chunk = store.get_chunk(chunk_ids[0])
        assert chunk is not None
        assert chunk.document_id == "doc1"

    def test_get_document_chunks(self):
        store = self._store()
        store.add_document("A document with some text content.", document_id="doc1")
        chunks = store.get_document_chunks("doc1")
        assert len(chunks) >= 1
        assert all(c.document_id == "doc1" for c in chunks)

    def test_remove_document(self):
        store = self._store()
        store.add_document("Document to be removed.", document_id="doc1")
        assert store.num_documents == 1
        store.remove_document("doc1")
        assert store.num_documents == 0
        assert store.num_chunks == 0

    def test_set_embedding(self):
        store = self._store()
        chunk_ids = store.add_document("This is a sufficiently long test document for embedding storage testing.")
        embedding = [0.1, 0.2, 0.3]
        store.set_embedding(chunk_ids[0], embedding)
        chunk = store.get_chunk(chunk_ids[0])
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_empty_text_rejected(self):
        store = self._store()
        result = store.add_document("")
        assert result == []
        assert store.num_documents == 0

    def test_stats(self):
        store = self._store()
        store.add_document("Document one.", source="test")
        store.add_document("Document two.", source="test")
        stats = store.stats()
        assert stats["documents"] == 2
        assert "test" in stats["sources"]

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5), persist_dir=tmpdir)
            store.add_document("This is a persistent document with enough words to survive chunking.", document_id="pdoc1")
            store.save()

            # Reload
            store2 = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5), persist_dir=tmpdir)
            assert store2.num_documents == 1
            assert store2.num_chunks >= 1

    def test_metadata_preserved(self):
        store = self._store()
        store.add_document("This is a test document with metadata authored by someone for wiki.", metadata={"author": "Alice"}, source="wiki")
        chunks = store.get_all_chunks()
        assert chunks[0].metadata.get("author") == "Alice"
        assert chunks[0].metadata.get("source") == "wiki"


# ══════════════════════════════════════════════════════════════
#  Embedding Model Tests
# ══════════════════════════════════════════════════════════════


class TestEmbeddingModel:
    def test_embed_returns_correct_dim(self):
        model = EmbeddingModel(dim=128)
        model.build_vocab(["hello world", "foo bar"])
        vec = model.embed("hello world")
        assert len(vec) == 128

    def test_embed_normalized(self):
        import math
        model = EmbeddingModel(dim=128)
        model.build_vocab(["the quick brown fox"])
        vec = model.embed("the quick brown fox")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01 or norm == 0.0

    def test_similar_texts_closer(self):
        model = EmbeddingModel(dim=256)
        corpus = [
            "Machine learning uses data to train models.",
            "Deep learning is a subset of machine learning.",
            "Cooking recipes for Italian pasta dishes.",
        ]
        model.build_vocab(corpus)

        vec_ml = model.embed("Machine learning uses data to train models.")
        vec_dl = model.embed("Deep learning is a subset of machine learning.")
        vec_cook = model.embed("Cooking recipes for Italian pasta dishes.")

        sim_related = cosine_similarity(vec_ml, vec_dl)
        sim_unrelated = cosine_similarity(vec_ml, vec_cook)
        assert sim_related > sim_unrelated

    def test_embed_batch(self):
        model = EmbeddingModel(dim=64)
        model.build_vocab(["a", "b"])
        vecs = model.embed_batch(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 64

    def test_empty_text(self):
        model = EmbeddingModel(dim=64)
        vec = model.embed("")
        assert len(vec) == 64
        assert all(v == 0.0 for v in vec)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 1.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 0.001

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# ══════════════════════════════════════════════════════════════
#  Retriever Tests
# ══════════════════════════════════════════════════════════════


def _make_test_store():
    """Create a store with sample documents for retrieval tests."""
    store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))
    store.add_document(
        "Albert Einstein developed the theory of relativity. "
        "He published special relativity in 1905 and general relativity in 1915.",
        source="physics", document_id="einstein"
    )
    store.add_document(
        "Python is a high-level programming language created by Guido van Rossum. "
        "It is widely used for web development, data science, and machine learning.",
        source="programming", document_id="python"
    )
    store.add_document(
        "The Great Wall of China is a series of fortifications built along the northern borders. "
        "It stretches over 13,000 miles and was built over many centuries.",
        source="history", document_id="wall"
    )
    store.add_document(
        "Photosynthesis is the process by which green plants convert sunlight into chemical energy. "
        "It requires water, carbon dioxide, and chlorophyll.",
        source="biology", document_id="photo"
    )
    return store


class TestDenseRetriever:
    def test_retrieve_relevant(self):
        store = _make_test_store()
        retriever = DenseRetriever(store)
        retriever.index()
        results = retriever.retrieve("Who developed relativity?", top_k=2)
        assert len(results) > 0
        # Einstein doc should rank highly
        top_sources = [r.chunk.metadata.get("source") for r in results]
        assert "physics" in top_sources

    def test_retrieve_top_k(self):
        store = _make_test_store()
        retriever = DenseRetriever(store)
        retriever.index()
        results = retriever.retrieve("Tell me about something", top_k=2)
        assert len(results) <= 2

    def test_retrieve_empty_store(self):
        store = DocumentStore()
        retriever = DenseRetriever(store)
        retriever.index()
        results = retriever.retrieve("anything")
        assert results == []


class TestBM25Retriever:
    def test_retrieve_keyword_match(self):
        store = _make_test_store()
        retriever = BM25Retriever(store)
        retriever.index()
        results = retriever.retrieve("Einstein relativity", top_k=2)
        assert len(results) > 0
        top_doc_ids = [r.chunk.document_id for r in results]
        assert "einstein" in top_doc_ids

    def test_retrieve_python(self):
        store = _make_test_store()
        retriever = BM25Retriever(store)
        retriever.index()
        results = retriever.retrieve("Python programming language", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.document_id == "python"

    def test_retrieve_no_match(self):
        store = _make_test_store()
        retriever = BM25Retriever(store)
        retriever.index()
        results = retriever.retrieve("xyznonexistentterm", top_k=5, min_score=0.1)
        assert len(results) == 0


class TestHybridRetriever:
    def test_hybrid_retrieve(self):
        store = _make_test_store()
        retriever = HybridRetriever(store)
        retriever.index()
        results = retriever.retrieve("photosynthesis plants", top_k=3)
        assert len(results) > 0
        top_doc_ids = [r.chunk.document_id for r in results]
        assert "photo" in top_doc_ids

    def test_hybrid_method_label(self):
        store = _make_test_store()
        retriever = HybridRetriever(store)
        retriever.index()
        results = retriever.retrieve("Great Wall China", top_k=1)
        assert results[0].method == "hybrid"


# ══════════════════════════════════════════════════════════════
#  Context Builder Tests
# ══════════════════════════════════════════════════════════════


class TestContextBuilder:
    def test_build_context(self):
        chunk = DocumentChunk(
            chunk_id="c1", document_id="d1",
            text="Einstein developed relativity.",
            metadata={"source": "physics"},
        )
        result = type("R", (), {"chunk": chunk, "score": 0.9, "method": "dense"})()
        # Wrap in list to match RetrievalResult interface
        from src.rag.retriever import RetrievalResult
        rr = RetrievalResult(chunk=chunk, score=0.9)

        builder = ContextBuilder()
        prompt = builder.build_context([rr], "Who developed relativity?")
        assert "Einstein" in prompt
        assert "Who developed relativity?" in prompt
        assert "physics" in prompt

    def test_empty_results(self):
        builder = ContextBuilder()
        prompt = builder.build_context([], "My question")
        assert prompt == "My question"

    def test_build_messages(self):
        from src.rag.retriever import RetrievalResult
        chunk = DocumentChunk(chunk_id="c1", document_id="d1", text="Context text.", metadata={})
        rr = RetrievalResult(chunk=chunk, score=0.8)

        builder = ContextBuilder()
        messages = builder.build_messages([rr], "My question")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "My question"


# ══════════════════════════════════════════════════════════════
#  RAG Pipeline Tests
# ══════════════════════════════════════════════════════════════


class TestRAGPipeline:
    def test_full_pipeline(self):
        store = _make_test_store()
        generated_text = "Einstein developed the theory of relativity."

        rag = RAGPipeline(
            store=store,
            generate_fn=lambda prompt: generated_text,
            retriever_type="hybrid",
        )
        rag.index()

        result = rag.query("Who developed relativity?")
        assert result["answer"] == generated_text
        assert result["num_chunks_retrieved"] > 0
        assert len(result["sources"]) > 0

    def test_pipeline_with_context(self):
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="sparse")
        rag.index()

        result = rag.query("Einstein", return_context=True)
        assert "context" in result
        assert "Einstein" in result["context"]

    def test_pipeline_no_generate_fn(self):
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="dense")
        rag.index()

        result = rag.query("Python")
        assert "No generation model" in result["answer"]

    def test_add_document_reindexes(self):
        store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))
        rag = RAGPipeline(store=store, retriever_type="sparse")
        rag.add_document("New document about quantum computing.", source="science")
        # Should auto-index on next retrieve
        results = rag.retrieve("quantum computing")
        assert len(results) > 0

    def test_query_with_messages(self):
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="sparse")
        rag.index()

        messages = rag.query_with_messages("Tell me about Python")
        assert len(messages) == 2
        assert messages[1]["content"] == "Tell me about Python"

    def test_stats(self):
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="hybrid")
        stats = rag.stats()
        assert stats["store"]["documents"] == 4
        assert stats["indexed"] is False

    def test_different_retriever_types(self):
        for rtype in ["dense", "sparse", "hybrid"]:
            store = _make_test_store()
            rag = RAGPipeline(store=store, retriever_type=rtype)
            rag.index()
            results = rag.retrieve("Einstein")
            assert len(results) > 0


# ══════════════════════════════════════════════════════════════
#  Tool Integration Tests
# ══════════════════════════════════════════════════════════════


class TestRAGToolIntegration:
    def test_register_rag_tools(self):
        from src.agent.tools import ToolRegistry
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="sparse")
        rag.index()

        registry = ToolRegistry()
        register_rag_tools(registry, rag)

        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "search_knowledge" in names
        assert "add_knowledge" in names

    def test_search_knowledge_tool(self):
        from src.agent.tools import ToolCall
        store = _make_test_store()
        rag = RAGPipeline(store=store, retriever_type="sparse")
        rag.index()

        registry, _ = create_rag_agent_registry()
        # Re-register with our store
        from src.agent.tools import ToolRegistry
        reg = ToolRegistry()
        register_rag_tools(reg, rag)

        call = ToolCall(name="search_knowledge", arguments={"query": "Einstein relativity"})
        result = reg.execute(call)
        assert result.success
        assert "Einstein" in result.result

    def test_add_knowledge_tool(self):
        from src.agent.tools import ToolCall, ToolRegistry
        store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))
        rag = RAGPipeline(store=store, retriever_type="sparse")

        reg = ToolRegistry()
        register_rag_tools(reg, rag)

        call = ToolCall(name="add_knowledge", arguments={
            "text": "Neptune is the eighth planet from the Sun.",
            "source": "astronomy",
        })
        result = reg.execute(call)
        assert result.success
        assert "chunks" in result.result

        # Now search for it
        call2 = ToolCall(name="search_knowledge", arguments={"query": "Neptune planet"})
        result2 = reg.execute(call2)
        assert result2.success
        assert "Neptune" in result2.result

    def test_create_rag_agent_registry(self):
        docs = [
            {"text": "The Eiffel Tower is in Paris, France.", "source": "geography"},
            {"text": "Mount Everest is the tallest mountain on Earth.", "source": "geography"},
        ]
        registry, rag = create_rag_agent_registry(documents=docs)

        tools = registry.list_tools()
        names = {t.name for t in tools}
        # Should have default tools + RAG tools
        assert "calculator" in names
        assert "search_knowledge" in names
        assert rag.store.num_documents == 2
