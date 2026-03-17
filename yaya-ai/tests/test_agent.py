"""Tests for the Yaya Agent Framework: tools, chat template, agent loop, data generator."""

import json
import pytest

from src.agent.tools import (
    ParameterSchema,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ToolRegistry,
    create_default_registry,
    _calculator,
    _json_extract,
    _string_transform,
    _unit_convert,
)
from src.agent.chat_template import (
    ChatTemplate,
    SYSTEM_OPEN,
    SYSTEM_CLOSE,
    USER_OPEN,
    USER_CLOSE,
    ASSISTANT_OPEN,
    ASSISTANT_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_RESULT_OPEN,
    TOOL_RESULT_CLOSE,
    format_tool_call,
    format_tool_result,
)
from src.agent.agent import Agent, SimpleAgent, ToolCallParser, AgentConfig


# ══════════════════════════════════════════════════════════════
#  Tool Schema Tests
# ══════════════════════════════════════════════════════════════


class TestParameterSchema:
    def test_to_json_schema(self):
        param = ParameterSchema(name="x", type="number", description="A number")
        schema = param.to_json_schema()
        assert schema["type"] == "number"
        assert schema["description"] == "A number"

    def test_enum_in_schema(self):
        param = ParameterSchema(name="op", type="string", enum=["add", "sub"])
        schema = param.to_json_schema()
        assert schema["enum"] == ["add", "sub"]


class TestToolDefinition:
    def test_to_openai_format(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters=[
                ParameterSchema(name="input", type="string", description="Input text"),
            ],
        )
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "test_tool"
        assert "input" in fmt["function"]["parameters"]["properties"]
        assert "input" in fmt["function"]["parameters"]["required"]

    def test_to_prompt_format(self):
        tool = ToolDefinition(
            name="calc",
            description="Calculate math",
            parameters=[ParameterSchema(name="expr", type="string", description="Expression")],
        )
        prompt = tool.to_prompt_format()
        assert "calc" in prompt
        assert "Calculate math" in prompt
        assert "expr" in prompt

    def test_optional_params_not_required(self):
        tool = ToolDefinition(
            name="t",
            description="d",
            parameters=[
                ParameterSchema(name="a", type="string", required=True),
                ParameterSchema(name="b", type="string", required=False),
            ],
        )
        fmt = tool.to_openai_format()
        assert "a" in fmt["function"]["parameters"]["required"]
        assert "b" not in fmt["function"]["parameters"]["required"]


class TestToolCall:
    def test_to_json(self):
        tc = ToolCall(name="calc", arguments={"expression": "2+2"})
        j = json.loads(tc.to_json())
        assert j["name"] == "calc"
        assert j["arguments"]["expression"] == "2+2"

    def test_from_json(self):
        raw = '{"name": "calc", "arguments": {"expression": "3*4"}}'
        tc = ToolCall.from_json(raw)
        assert tc.name == "calc"
        assert tc.arguments["expression"] == "3*4"


# ══════════════════════════════════════════════════════════════
#  Built-in Tool Implementation Tests
# ══════════════════════════════════════════════════════════════


class TestCalculator:
    def test_basic_arithmetic(self):
        assert _calculator("2 + 3") == "5"
        assert _calculator("10 * 5") == "50"
        assert _calculator("100 / 4") == "25"

    def test_power(self):
        assert _calculator("2**10") == "1024"

    def test_sqrt(self):
        assert _calculator("sqrt(144)") == "12"

    def test_trig(self):
        result = float(_calculator("sin(pi/2)"))
        assert abs(result - 1.0) < 0.001

    def test_log(self):
        assert _calculator("log2(256)") == "8"

    def test_syntax_error(self):
        result = _calculator("2 +* 3")
        assert "error" in result.lower() or "Error" in result

    def test_unsafe_blocked(self):
        result = _calculator("__import__('os').system('ls')")
        assert "Unsupported" in result or "Unknown" in result


class TestJsonExtract:
    def test_simple_path(self):
        data = '{"name": "Alice", "age": 30}'
        assert _json_extract(data, "name") == "Alice"

    def test_nested_path(self):
        data = '{"data": {"items": [{"id": 1}, {"id": 2}]}}'
        assert _json_extract(data, "data.items.1.id") == "2"

    def test_invalid_json(self):
        assert "Invalid JSON" in _json_extract("not json", "key")

    def test_missing_key(self):
        assert "not found" in _json_extract('{"a": 1}', "b")


class TestStringTransform:
    def test_upper(self):
        assert _string_transform("hello", "upper") == "HELLO"

    def test_lower(self):
        assert _string_transform("HELLO", "lower") == "hello"

    def test_reverse(self):
        assert _string_transform("abc", "reverse") == "cba"

    def test_word_count(self):
        assert _string_transform("one two three", "word_count") == "3"

    def test_length(self):
        assert _string_transform("hello", "length") == "5"

    def test_unknown_op(self):
        result = _string_transform("x", "unknown_op")
        assert "Unknown operation" in result


class TestUnitConvert:
    def test_fahrenheit_to_celsius(self):
        result = _unit_convert(212, "F", "C")
        assert "100" in result

    def test_miles_to_km(self):
        result = _unit_convert(1, "mi", "km")
        assert "1.609" in result

    def test_pounds_to_kg(self):
        result = _unit_convert(100, "lb", "kg")
        assert "45" in result

    def test_invalid_units(self):
        result = _unit_convert(1, "xyz", "abc")
        assert "Cannot convert" in result


# ══════════════════════════════════════════════════════════════
#  Tool Registry Tests
# ══════════════════════════════════════════════════════════════


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.registry.register_function(
            name="add",
            description="Add two numbers",
            parameters=[
                ParameterSchema(name="a", type="number", description="First number"),
                ParameterSchema(name="b", type="number", description="Second number"),
            ],
            implementation=lambda a, b: str(a + b),
        )

    def test_register_and_get(self):
        tool = self.registry.get("add")
        assert tool is not None
        assert tool.name == "add"

    def test_list_tools(self):
        tools = self.registry.list_tools()
        assert len(tools) == 1

    def test_validate_valid_call(self):
        call = ToolCall(name="add", arguments={"a": 1, "b": 2})
        assert self.registry.validate_call(call) is None

    def test_validate_missing_param(self):
        call = ToolCall(name="add", arguments={"a": 1})
        error = self.registry.validate_call(call)
        assert "Missing required parameter" in error

    def test_validate_unknown_tool(self):
        call = ToolCall(name="nonexistent", arguments={})
        error = self.registry.validate_call(call)
        assert "Unknown tool" in error

    def test_validate_unknown_param(self):
        call = ToolCall(name="add", arguments={"a": 1, "b": 2, "c": 3})
        error = self.registry.validate_call(call)
        assert "Unknown parameter" in error

    def test_execute_success(self):
        call = ToolCall(name="add", arguments={"a": 3, "b": 4})
        result = self.registry.execute(call)
        assert result.success is True
        assert result.result == "7"

    def test_execute_validation_error(self):
        call = ToolCall(name="add", arguments={"a": 1})
        result = self.registry.execute(call)
        assert result.success is False

    def test_execute_runtime_error(self):
        self.registry.register_function(
            name="fail",
            description="Always fails",
            parameters=[],
            implementation=lambda: 1 / 0,
        )
        call = ToolCall(name="fail", arguments={})
        result = self.registry.execute(call)
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_system_prompt_generation(self):
        prompt = self.registry.get_system_prompt()
        assert "add" in prompt
        assert "tool_call" in prompt

    def test_openai_schema(self):
        schema = self.registry.get_openai_schema()
        assert len(schema) == 1
        assert schema[0]["function"]["name"] == "add"


class TestDefaultRegistry:
    def test_has_builtin_tools(self):
        registry = create_default_registry()
        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert "calculator" in names
        assert "json_extract" in names
        assert "string_transform" in names
        assert "unit_convert" in names
        assert "datetime_info" in names

    def test_calculator_execution(self):
        registry = create_default_registry()
        call = ToolCall(name="calculator", arguments={"expression": "2**8"})
        result = registry.execute(call)
        assert result.success
        assert result.result == "256"


# ══════════════════════════════════════════════════════════════
#  Chat Template Tests
# ══════════════════════════════════════════════════════════════


class TestChatTags:
    def test_closing_tags_are_valid(self):
        assert SYSTEM_CLOSE == "<" + chr(47) + "|system|>"
        assert USER_CLOSE == "<" + chr(47) + "|user|>"
        assert ASSISTANT_CLOSE == "<" + chr(47) + "|assistant|>"
        assert TOOL_CALL_CLOSE == "<" + chr(47) + "tool_call>"
        assert TOOL_RESULT_CLOSE == "<" + chr(47) + "tool_result>"


class TestChatTemplate:
    def test_simple_conversation(self):
        ct = ChatTemplate(system_prompt="You are helpful.")
        ct.add_message("user", "Hello")
        ct.add_message("assistant", "Hi there!")
        formatted = ct.format()
        assert SYSTEM_OPEN in formatted
        assert USER_OPEN in formatted
        assert ASSISTANT_OPEN in formatted
        assert "Hello" in formatted
        assert "Hi there!" in formatted

    def test_format_for_generation(self):
        ct = ChatTemplate()
        ct.add_message("user", "What is 2+2?")
        gen = ct.format_for_generation()
        assert gen.endswith(ASSISTANT_OPEN + "\n")

    def test_tool_call_in_conversation(self):
        ct = ChatTemplate()
        ct.add_message("user", "Calculate 5*3")
        ct.add_message(
            "assistant",
            "Let me calculate.",
            tool_calls=[{"name": "calculator", "arguments": {"expression": "5*3"}}],
        )
        ct.add_tool_result("calculator", "15")
        ct.add_message("assistant", "5 times 3 is 15.")

        formatted = ct.format()
        assert TOOL_CALL_OPEN in formatted
        assert TOOL_CALL_CLOSE in formatted
        assert TOOL_RESULT_OPEN in formatted
        assert '"calculator"' in formatted

    def test_training_pairs(self):
        ct = ChatTemplate(system_prompt="Be helpful.")
        ct.add_message("user", "Hi")
        ct.add_message("assistant", "Hello!")
        ct.add_message("user", "How are you?")
        ct.add_message("assistant", "I'm good!")

        pairs = ct.get_training_pairs()
        assert len(pairs) == 2
        assert "Hello!" in pairs[0]["target"]
        assert "I'm good!" in pairs[1]["target"]

    def test_from_messages(self):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        ct = ChatTemplate.from_messages(messages)
        assert len(ct.messages) == 3
        formatted = ct.format()
        assert "System prompt" in formatted
        assert "Question" in formatted
        assert "Answer" in formatted


class TestFormatFunctions:
    def test_format_tool_call(self):
        result = format_tool_call("calc", {"expr": "2+2"})
        assert TOOL_CALL_OPEN in result
        assert TOOL_CALL_CLOSE in result
        assert '"calc"' in result

    def test_format_tool_result(self):
        result = format_tool_result("calc", "4", success=True)
        assert TOOL_RESULT_OPEN in result
        assert TOOL_RESULT_CLOSE in result
        assert '"success"' in result


# ══════════════════════════════════════════════════════════════
#  Tool Call Parser Tests
# ══════════════════════════════════════════════════════════════


class TestToolCallParser:
    def test_parse_single_call(self):
        text = (
            f"Let me calculate that.\n"
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calculator", "arguments": {{"expression": "2+2"}}}}\n'
            f"{TOOL_CALL_CLOSE}"
        )
        calls = ToolCallParser.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments["expression"] == "2+2"

    def test_parse_multiple_calls(self):
        text = (
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calc", "arguments": {{"expression": "1+1"}}}}\n'
            f"{TOOL_CALL_CLOSE}\n"
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calc", "arguments": {{"expression": "2+2"}}}}\n'
            f"{TOOL_CALL_CLOSE}"
        )
        calls = ToolCallParser.parse(text)
        assert len(calls) == 2

    def test_has_tool_call(self):
        assert ToolCallParser.has_tool_call(f"text {TOOL_CALL_OPEN} more")
        assert not ToolCallParser.has_tool_call("plain text")

    def test_extract_thought(self):
        text = f"I need to calculate this.\n{TOOL_CALL_OPEN}\nstuff\n{TOOL_CALL_CLOSE}"
        thought = ToolCallParser.extract_thought(text)
        assert thought == "I need to calculate this."

    def test_no_tool_call_returns_full_text(self):
        text = "Just a normal response."
        thought = ToolCallParser.extract_thought(text)
        assert thought == text

    def test_malformed_json_skipped(self):
        text = f"{TOOL_CALL_OPEN}\nnot valid json\n{TOOL_CALL_CLOSE}"
        calls = ToolCallParser.parse(text)
        assert len(calls) == 0


# ══════════════════════════════════════════════════════════════
#  Agent Tests
# ══════════════════════════════════════════════════════════════


class TestSimpleAgent:
    def test_execute_call(self):
        agent = SimpleAgent()
        result = agent.execute_call("calculator", expression="7*8")
        assert result.success
        assert result.result == "56"

    def test_execute_text_with_tool_call(self):
        text = (
            f"Let me help.\n"
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calculator", "arguments": {{"expression": "10+5"}}}}\n'
            f"{TOOL_CALL_CLOSE}"
        )
        agent = SimpleAgent()
        results = agent.execute_text(text)
        assert len(results) == 1
        assert results[0].result == "15"

    def test_execute_text_no_tool_call(self):
        agent = SimpleAgent()
        results = agent.execute_text("Plain text response.")
        assert len(results) == 0


class TestAgent:
    def _make_agent(self, responses):
        """Create an agent with a mock generate function."""
        call_count = [0]

        def mock_generate(prompt: str) -> str:
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        config = AgentConfig(max_steps=5, verbose=False)
        return Agent(generate_fn=mock_generate, config=config)

    def test_direct_response(self):
        agent = self._make_agent(["The answer is 42."])
        result = agent.run("What is the answer?")
        assert result == "The answer is 42."
        assert len(agent.history) == 1

    def test_tool_use_then_respond(self):
        # First response uses a tool, second gives final answer
        tool_call_text = (
            f"Let me calculate.\n"
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calculator", "arguments": {{"expression": "6*7"}}}}\n'
            f"{TOOL_CALL_CLOSE}"
        )
        agent = self._make_agent([tool_call_text, "6 times 7 is 42."])
        result = agent.run("What is 6 times 7?")
        assert "42" in result
        assert len(agent.history) == 2

    def test_max_steps_limit(self):
        # Always returns tool calls — should stop at max_steps
        tool_call_text = (
            f"{TOOL_CALL_OPEN}\n"
            f'{{"name": "calculator", "arguments": {{"expression": "1+1"}}}}\n'
            f"{TOOL_CALL_CLOSE}"
        )
        agent = self._make_agent([tool_call_text] * 10 + ["done"])
        config = AgentConfig(max_steps=3, verbose=False)
        agent.config = config
        result = agent.run("Loop forever")
        # Should have 3 steps in history (not 10)
        assert len(agent.history) <= 3

    def test_get_trace(self):
        agent = self._make_agent(["Simple answer."])
        agent.run("Question")
        trace = agent.get_trace()
        assert len(trace) == 1
        assert trace[0]["response"] == "Simple answer."


# ══════════════════════════════════════════════════════════════
#  Data Generator Tests
# ══════════════════════════════════════════════════════════════


class TestToolUseDataGenerator:
    def test_generate_calculator_examples(self):
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        examples = gen.generate_calculator_examples()
        assert len(examples) > 0
        assert "conversation" in examples[0]
        assert "training_pairs" in examples[0]
        assert "calculator" in examples[0]["tools_used"]

    def test_generate_unit_examples(self):
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        examples = gen.generate_unit_examples()
        assert len(examples) > 0

    def test_generate_string_examples(self):
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        examples = gen.generate_string_examples()
        assert len(examples) > 0

    def test_generate_all(self):
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        all_examples = gen.generate_all()
        assert len(all_examples) > 10  # Should have a good number of examples

    def test_training_pairs_have_content(self):
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        examples = gen.generate_calculator_examples()
        for ex in examples:
            pairs = ex["training_pairs"]
            assert len(pairs) > 0
            for pair in pairs:
                assert "input" in pair
                assert "target" in pair
                assert len(pair["input"]) > 0
                assert len(pair["target"]) > 0

    def test_save_jsonl(self):
        import tempfile
        import os
        from src.agent.data_generator import ToolUseDataGenerator
        gen = ToolUseDataGenerator()
        examples = gen.generate_calculator_examples()[:3]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tool_use.jsonl")
            gen.save_jsonl(examples, path)
            assert os.path.exists(path)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3
            record = json.loads(lines[0])
            assert "messages" in record
