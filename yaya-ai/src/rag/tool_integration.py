"""RAG integration with the Yaya Agent Framework.

Registers retrieval as a tool that the agent can invoke during
multi-step reasoning, enabling knowledge-grounded responses.
"""

from typing import Optional, Dict, Any

from src.agent.tools import ToolRegistry, ParameterSchema
from src.rag.pipeline import RAGPipeline
from src.rag.document_store import DocumentStore


def _make_search_fn(rag: RAGPipeline):
    """Create a search function closure over a RAG pipeline."""
    def search_knowledge(query: str, num_results: int = 3) -> str:
        results = rag.retrieve(query, top_k=num_results)
        if not results:
            return "No relevant documents found."

        parts = []
        for i, r in enumerate(results):
            source = r.chunk.metadata.get("source", "unknown")
            parts.append(
                f"[{i+1}] (score: {r.score:.3f}, source: {source})\n{r.chunk.text[:500]}"
            )
        return "\n\n".join(parts)
    return search_knowledge


def _make_add_fn(rag: RAGPipeline):
    """Create a document addition function closure."""
    def add_knowledge(text: str, source: str = "user") -> str:
        chunk_ids = rag.store.add_document(text, source=source)
        rag._indexed = False  # Force re-index on next retrieval
        return f"Added document with {len(chunk_ids)} chunks."
    return add_knowledge


def register_rag_tools(registry: ToolRegistry, rag: RAGPipeline):
    """Register RAG-related tools in the agent's tool registry.

    Adds:
    - search_knowledge: Search the knowledge base for relevant information
    - add_knowledge: Add new information to the knowledge base

    Args:
        registry: Agent tool registry.
        rag: Configured RAG pipeline.
    """
    registry.register_function(
        name="search_knowledge",
        description=(
            "Search the knowledge base for information relevant to a query. "
            "Returns the most relevant document excerpts with relevance scores."
        ),
        parameters=[
            ParameterSchema(
                name="query",
                type="string",
                description="Search query describing what information you need",
            ),
            ParameterSchema(
                name="num_results",
                type="integer",
                description="Number of results to return (default: 3)",
                required=False,
                default=3,
            ),
        ],
        implementation=_make_search_fn(rag),
        category="knowledge",
    )

    registry.register_function(
        name="add_knowledge",
        description="Add new information to the knowledge base for future retrieval.",
        parameters=[
            ParameterSchema(
                name="text",
                type="string",
                description="Text content to add to the knowledge base",
            ),
            ParameterSchema(
                name="source",
                type="string",
                description="Source identifier for the content",
                required=False,
                default="user",
            ),
        ],
        implementation=_make_add_fn(rag),
        category="knowledge",
    )


def create_rag_agent_registry(
    documents: Optional[list] = None,
    retriever_type: str = "hybrid",
    top_k: int = 5,
) -> tuple:
    """Create a ToolRegistry pre-loaded with default tools + RAG tools.

    Args:
        documents: Optional list of dicts with 'text' and 'source' to pre-load.
        retriever_type: Type of retriever ('dense', 'sparse', 'hybrid').
        top_k: Default number of results per query.

    Returns:
        Tuple of (ToolRegistry, RAGPipeline).
    """
    from src.agent.tools import create_default_registry
    from src.rag.document_store import TextChunker

    store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))
    if documents:
        store.add_documents(documents)

    rag = RAGPipeline(
        store=store,
        retriever_type=retriever_type,
        top_k=top_k,
    )

    if documents:
        rag.index()

    registry = create_default_registry()
    register_rag_tools(registry, rag)

    return registry, rag
