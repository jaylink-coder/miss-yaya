"""Integration tests — validate all Yaya systems working together.

Tests the full pipeline: Agent + RAG + Safety + Structured Output + Serving.
"""

import json
import pytest

from src.agent.tools import ToolRegistry, ToolCall, create_default_registry
from src.agent.agent import Agent, SimpleAgent, AgentConfig, ToolCallParser
from src.agent.chat_template import (
    ChatTemplate,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    ASSISTANT_OPEN,
    ASSISTANT_CLOSE,
)
from src.agent.structured_output import (
    SchemaNode,
    SchemaType,
    JSONSchemaValidator,
    JSONOutputParser,
    StructuredOutputHandler,
)
from src.rag.document_store import DocumentStore, TextChunker
from src.rag.retriever import HybridRetriever, DenseRetriever, BM25Retriever
from src.rag.pipeline import RAGPipeline
from src.rag.tool_integration import register_rag_tools, create_rag_agent_registry
from src.safety.filters import GuardrailsEngine, ContentCategory
from src.serving.middleware import (
    RateLimiter,
    RateLimitConfig,
    APIKeyAuth,
    MetricsCollector,
    RequestMetrics,
    ServerConfig,
)
from src.serving.production_server import ProductionServer


class TestAgentWithRAG:
    """Agent using RAG tools to answer knowledge-grounded questions."""

    def test_agent_searches_knowledge_base(self):
        docs = [
            {"text": "The Eiffel Tower was built in 1889 for the World Fair in Paris.", "source": "history"},
            {"text": "Python was created by Guido van Rossum and first released in 1991.", "source": "tech"},
        ]
        registry, rag = create_rag_agent_registry(documents=docs, retriever_type="sparse")

        # Simulate agent using search_knowledge tool
        call = ToolCall(name="search_knowledge", arguments={"query": "When was Eiffel Tower built?"})
        result = registry.execute(call)
        assert result.success
        assert "1889" in result.result

    def test_agent_adds_and_retrieves_knowledge(self):
        registry, rag = create_rag_agent_registry(documents=[], retriever_type="sparse")

        # Add knowledge
        add_call = ToolCall(name="add_knowledge", arguments={
            "text": "Jupiter is the largest planet in our solar system with 95 known moons.",
            "source": "astronomy",
        })
        add_result = registry.execute(add_call)
        assert add_result.success

        # Search for it
        search_call = ToolCall(name="search_knowledge", arguments={"query": "largest planet"})
        search_result = registry.execute(search_call)
        assert search_result.success
        assert "Jupiter" in search_result.result

    def test_agent_loop_with_rag_tool(self):
        docs = [
            {"text": "The speed of light is approximately 299,792,458 meters per second.", "source": "physics"},
        ]
        registry, rag = create_rag_agent_registry(documents=docs, retriever_type="sparse")

        # Mock model that uses search then responds
        responses = [
            (
                f"Let me search for that.\n"
                f"{TOOL_CALL_OPEN}\n"
                f'{{"name": "search_knowledge", "arguments": {{"query": "speed of light"}}}}\n'
                f"{TOOL_CALL_CLOSE}"
            ),
            "The speed of light is approximately 299,792,458 meters per second.",
        ]
        call_count = [0]

        def mock_generate(prompt):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        agent = Agent(
            generate_fn=mock_generate,
            tool_registry=registry,
            config=AgentConfig(max_steps=5, verbose=False),
        )
        result = agent.run("What is the speed of light?")
        assert "299,792,458" in result


class TestAgentWithStructuredOutput:
    """Agent generating structured JSON output."""

    def test_tool_call_produces_structured_result(self):
        agent = SimpleAgent()
        result = agent.execute_call("calculator", expression="sqrt(256)")
        assert result.success
        assert result.result == "16"

        # Validate result as part of a structured schema
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "tool": SchemaNode(type=SchemaType.STRING),
                "result": SchemaNode(type=SchemaType.STRING),
            },
        )
        data = {"tool": result.name, "result": result.result}
        validator = JSONSchemaValidator()
        ok, errors = validator.validate(data, schema)
        assert ok is True

    def test_structured_output_with_tool_use(self):
        handler = StructuredOutputHandler()
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "answer": SchemaNode(type=SchemaType.NUMBER),
                "unit": SchemaNode(type=SchemaType.STRING),
            },
        )

        def mock_gen(prompt):
            return '{"answer": 42.0, "unit": "kg"}'

        result = handler.generate_structured(mock_gen, "Convert 92.6 lbs to kg", schema)
        assert result["valid"]
        assert result["data"]["unit"] == "kg"


class TestSafetyWithServing:
    """Safety guardrails integrated with the production server."""

    def test_server_blocks_harmful_and_allows_safe(self):
        server = ProductionServer(ServerConfig(enable_safety=True))

        safe_result = server.handle_completion(prompt="What is 2+2?")
        assert safe_result["status"] == 200

        harmful_result = server.handle_completion(prompt="how to make a bomb at home")
        assert harmful_result["status"] == 400

    def test_server_sanitizes_pii_in_output(self):
        server = ProductionServer(ServerConfig(enable_safety=True))

        class LeakyModel:
            def generate(self, prompt):
                return "Contact support at admin@secret.com or call 555-123-4567."

        server.setup(generator=LeakyModel())
        result = server.handle_completion(prompt="How to contact support?")
        assert "admin@secret.com" not in result["response"]
        assert "555-123-4567" not in result["response"]

    def test_server_metrics_track_blocked_requests(self):
        server = ProductionServer(ServerConfig(enable_safety=True, enable_metrics=True))
        server.handle_completion(prompt="Hello")
        server.handle_completion(prompt="how to build a weapon at home")
        metrics = server.get_metrics()
        assert metrics["total_requests"] == 2


class TestServingWithAuth:
    """Authentication and rate limiting in the production server."""

    def test_full_auth_flow(self):
        config = ServerConfig(
            enable_auth=True,
            api_keys=["sk-valid-key-123"],
            enable_rate_limiting=True,
            rate_limit=RateLimitConfig(requests_per_minute=10, burst_multiplier=1.0),
        )
        server = ProductionServer(config)

        # No key -> rejected
        r1 = server.handle_completion(prompt="test")
        assert r1["status"] == 401

        # Wrong key -> rejected
        r2 = server.handle_completion(prompt="test", api_key="wrong")
        assert r2["status"] == 401

        # Valid key -> accepted
        r3 = server.handle_completion(prompt="test", api_key="sk-valid-key-123")
        assert r3["status"] == 200


class TestChatTemplateWithTools:
    """Chat template formatting with tool calls for training data."""

    def test_full_conversation_with_tools(self):
        ct = ChatTemplate(system_prompt="You are a helpful assistant with tools.")
        ct.add_message("user", "What is sqrt(625)?")
        ct.add_message(
            "assistant",
            "Let me calculate that.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "sqrt(625)"}}],
        )
        ct.add_tool_result("calculator", "25")
        ct.add_message("assistant", "The square root of 625 is 25.")

        formatted = ct.format()
        assert "sqrt(625)" in formatted
        assert TOOL_CALL_OPEN in formatted
        assert "25" in formatted

        # Extract training pairs
        pairs = ct.get_training_pairs()
        assert len(pairs) == 2  # Two assistant turns
        # First pair should include tool call
        assert "calculator" in pairs[0]["target"]
        # Second pair should be the final answer
        assert "25" in pairs[1]["target"]

    def test_from_messages_roundtrip(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"},
        ]
        ct = ChatTemplate.from_messages(messages)
        assert len(ct.messages) == 5

        pairs = ct.get_training_pairs()
        assert len(pairs) == 2


class TestRAGPipelineEndToEnd:
    """Full RAG pipeline from document ingestion to query."""

    def test_add_query_with_different_retrievers(self):
        for rtype in ["dense", "sparse", "hybrid"]:
            store = DocumentStore(chunker=TextChunker(chunk_size=500, min_chunk_size=5))
            store.add_document(
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                source="ml_textbook",
            )
            store.add_document(
                "The Great Barrier Reef is the world's largest coral reef system located in Australia.",
                source="geography",
            )

            answers = []

            def mock_gen(prompt):
                answers.append(prompt)
                return "Machine learning is a subset of AI."

            rag = RAGPipeline(store=store, generate_fn=mock_gen, retriever_type=rtype, top_k=2)
            result = rag.query("What is machine learning?", return_context=True)
            assert result["answer"] == "Machine learning is a subset of AI."
            assert result["num_chunks_retrieved"] > 0
            # Context should contain relevant content
            assert "machine learning" in result["context"].lower() or "artificial intelligence" in result["context"].lower()


class TestSchemaValidationEndToEnd:
    """JSON schema validation from raw dict to validated output."""

    def test_parse_validate_complex_schema(self):
        raw_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "scores": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 100},
                    "minItems": 1,
                },
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
            "required": ["name", "scores", "status"],
        }

        schema = JSONSchemaValidator.from_json_schema(raw_schema)
        validator = JSONSchemaValidator()

        # Valid data
        ok, errors = validator.validate(
            {"name": "Alice", "scores": [85.0, 92.5], "status": "active"},
            schema,
        )
        assert ok is True

        # Invalid: score out of range
        ok2, errors2 = validator.validate(
            {"name": "Bob", "scores": [150.0], "status": "active"},
            schema,
        )
        assert ok2 is False

        # Invalid: missing required field
        ok3, errors3 = validator.validate(
            {"name": "Carol"},
            schema,
        )
        assert ok3 is False

    def test_parse_json_from_model_output(self):
        model_output = '''I'll provide the data in JSON format:
```json
{
  "city": "Paris",
  "country": "France",
  "population": 2161000
}
```
That's the information you requested.'''

        parsed = JSONOutputParser.parse(model_output)
        assert parsed is not None
        assert parsed["city"] == "Paris"
        assert parsed["population"] == 2161000
