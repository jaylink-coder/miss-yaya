"""Tests for Yaya Structured Output: JSON schema validation, parsing, and constrained generation."""

import json
import pytest

from src.agent.structured_output import (
    SchemaType,
    SchemaNode,
    JSONSchemaValidator,
    JSONOutputParser,
    StructuredOutputHandler,
)


# ══════════════════════════════════════════════════════════════
#  Schema Validation Tests
# ══════════════════════════════════════════════════════════════


class TestJSONSchemaValidator:
    def setup_method(self):
        self.validator = JSONSchemaValidator()

    def test_string_valid(self):
        schema = SchemaNode(type=SchemaType.STRING)
        ok, errors = self.validator.validate("hello", schema)
        assert ok is True

    def test_string_invalid(self):
        schema = SchemaNode(type=SchemaType.STRING)
        ok, errors = self.validator.validate(42, schema)
        assert ok is False

    def test_integer_valid(self):
        schema = SchemaNode(type=SchemaType.INTEGER)
        ok, errors = self.validator.validate(42, schema)
        assert ok is True

    def test_integer_rejects_bool(self):
        schema = SchemaNode(type=SchemaType.INTEGER)
        ok, errors = self.validator.validate(True, schema)
        assert ok is False

    def test_number_valid(self):
        schema = SchemaNode(type=SchemaType.NUMBER)
        ok, errors = self.validator.validate(3.14, schema)
        assert ok is True

    def test_number_accepts_int(self):
        schema = SchemaNode(type=SchemaType.NUMBER)
        ok, errors = self.validator.validate(42, schema)
        assert ok is True

    def test_boolean_valid(self):
        schema = SchemaNode(type=SchemaType.BOOLEAN)
        ok, errors = self.validator.validate(True, schema)
        assert ok is True

    def test_array_valid(self):
        schema = SchemaNode(
            type=SchemaType.ARRAY,
            items=SchemaNode(type=SchemaType.INTEGER),
        )
        ok, errors = self.validator.validate([1, 2, 3], schema)
        assert ok is True

    def test_array_invalid_items(self):
        schema = SchemaNode(
            type=SchemaType.ARRAY,
            items=SchemaNode(type=SchemaType.INTEGER),
        )
        ok, errors = self.validator.validate([1, "two", 3], schema)
        assert ok is False

    def test_array_min_items(self):
        schema = SchemaNode(type=SchemaType.ARRAY, min_items=2)
        ok, errors = self.validator.validate([1], schema)
        assert ok is False

    def test_array_max_items(self):
        schema = SchemaNode(type=SchemaType.ARRAY, max_items=2)
        ok, errors = self.validator.validate([1, 2, 3], schema)
        assert ok is False

    def test_object_valid(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "name": SchemaNode(type=SchemaType.STRING),
                "age": SchemaNode(type=SchemaType.INTEGER),
            },
        )
        ok, errors = self.validator.validate({"name": "Alice", "age": 30}, schema)
        assert ok is True

    def test_object_missing_required(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "name": SchemaNode(type=SchemaType.STRING, required=True),
                "age": SchemaNode(type=SchemaType.INTEGER, required=True),
            },
        )
        ok, errors = self.validator.validate({"name": "Alice"}, schema)
        assert ok is False
        assert any("required" in e or "age" in e for e in errors)

    def test_object_optional_field(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "name": SchemaNode(type=SchemaType.STRING, required=True),
                "nickname": SchemaNode(type=SchemaType.STRING, required=False),
            },
        )
        ok, errors = self.validator.validate({"name": "Alice"}, schema)
        assert ok is True

    def test_nested_object(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "user": SchemaNode(
                    type=SchemaType.OBJECT,
                    properties={
                        "name": SchemaNode(type=SchemaType.STRING),
                        "score": SchemaNode(type=SchemaType.NUMBER),
                    },
                ),
            },
        )
        ok, errors = self.validator.validate(
            {"user": {"name": "Bob", "score": 95.5}}, schema
        )
        assert ok is True

    def test_enum_valid(self):
        schema = SchemaNode(type=SchemaType.STRING, enum=["red", "green", "blue"])
        ok, errors = self.validator.validate("red", schema)
        assert ok is True

    def test_enum_invalid(self):
        schema = SchemaNode(type=SchemaType.STRING, enum=["red", "green", "blue"])
        ok, errors = self.validator.validate("yellow", schema)
        assert ok is False

    def test_string_min_length(self):
        schema = SchemaNode(type=SchemaType.STRING, min_length=3)
        ok, errors = self.validator.validate("ab", schema)
        assert ok is False

    def test_string_max_length(self):
        schema = SchemaNode(type=SchemaType.STRING, max_length=5)
        ok, errors = self.validator.validate("toolong", schema)
        assert ok is False

    def test_string_pattern(self):
        schema = SchemaNode(type=SchemaType.STRING, pattern=r"^\d{3}-\d{4}$")
        ok, errors = self.validator.validate("123-4567", schema)
        assert ok is True
        ok2, errors2 = self.validator.validate("abc", schema)
        assert ok2 is False

    def test_number_minimum(self):
        schema = SchemaNode(type=SchemaType.NUMBER, minimum=0.0)
        ok, errors = self.validator.validate(-1.0, schema)
        assert ok is False

    def test_number_maximum(self):
        schema = SchemaNode(type=SchemaType.NUMBER, maximum=100.0)
        ok, errors = self.validator.validate(101.0, schema)
        assert ok is False

    def test_null_type(self):
        schema = SchemaNode(type=SchemaType.NULL)
        ok, errors = self.validator.validate(None, schema)
        assert ok is True

    def test_required_null_fails(self):
        schema = SchemaNode(type=SchemaType.STRING, required=True)
        ok, errors = self.validator.validate(None, schema)
        assert ok is False


class TestFromJSONSchema:
    def test_parse_simple_object(self):
        raw = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        node = JSONSchemaValidator.from_json_schema(raw)
        assert node.type == SchemaType.OBJECT
        assert "name" in node.properties
        assert node.properties["name"].required is True
        assert node.properties["age"].required is False

    def test_parse_array(self):
        raw = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
        node = JSONSchemaValidator.from_json_schema(raw)
        assert node.type == SchemaType.ARRAY
        assert node.items.type == SchemaType.STRING
        assert node.min_items == 1

    def test_parse_with_enum(self):
        raw = {
            "type": "string",
            "enum": ["low", "medium", "high"],
        }
        node = JSONSchemaValidator.from_json_schema(raw)
        assert node.enum == ["low", "medium", "high"]


# ══════════════════════════════════════════════════════════════
#  JSON Output Parser Tests
# ══════════════════════════════════════════════════════════════


class TestJSONOutputParser:
    def test_pure_json(self):
        text = '{"name": "Alice", "age": 30}'
        result = JSONOutputParser.parse(text)
        assert result == {"name": "Alice", "age": 30}

    def test_json_in_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = JSONOutputParser.parse(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        text = 'The answer is: {"result": 42} as computed.'
        result = JSONOutputParser.parse(text)
        assert result == {"result": 42}

    def test_json_array(self):
        text = '[1, 2, 3]'
        result = JSONOutputParser.parse(text)
        assert result == [1, 2, 3]

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2]}}'
        result = JSONOutputParser.parse(text)
        assert result["outer"]["inner"] == [1, 2]

    def test_no_json(self):
        text = "This is just plain text with no JSON."
        result = JSONOutputParser.parse(text)
        assert result is None

    def test_extract_json_string(self):
        text = 'prefix {"a": 1} suffix'
        extracted = JSONOutputParser.extract_json(text)
        assert extracted is not None
        assert json.loads(extracted) == {"a": 1}

    def test_json_with_escaped_quotes(self):
        text = '{"message": "He said \\"hello\\""}'
        result = JSONOutputParser.parse(text)
        assert result is not None
        assert "hello" in result["message"]

    def test_code_block_without_lang(self):
        text = '```\n{"x": 1}\n```'
        result = JSONOutputParser.parse(text)
        assert result == {"x": 1}


# ══════════════════════════════════════════════════════════════
#  Structured Output Handler Tests
# ══════════════════════════════════════════════════════════════


class TestStructuredOutputHandler:
    def setup_method(self):
        self.handler = StructuredOutputHandler()

    def test_process_valid_output(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "answer": SchemaNode(type=SchemaType.STRING),
            },
        )
        result = self.handler.process_output('{"answer": "42"}', schema)
        assert result["valid"] is True
        assert result["data"]["answer"] == "42"

    def test_process_invalid_output(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "answer": SchemaNode(type=SchemaType.INTEGER),
            },
        )
        result = self.handler.process_output('{"answer": "not a number"}', schema)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_process_no_json(self):
        schema = SchemaNode(type=SchemaType.OBJECT)
        result = self.handler.process_output("Just some text", schema)
        assert result["valid"] is False
        assert result["data"] is None

    def test_create_prompt_suffix(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "name": SchemaNode(type=SchemaType.STRING, description="User name"),
                "age": SchemaNode(type=SchemaType.INTEGER, description="User age"),
            },
        )
        suffix = self.handler.create_prompt_suffix(schema)
        assert "JSON" in suffix
        assert "name" in suffix
        assert "age" in suffix

    def test_generate_structured_success(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "result": SchemaNode(type=SchemaType.INTEGER),
            },
        )

        def mock_gen(prompt):
            return '{"result": 42}'

        result = self.handler.generate_structured(mock_gen, "What is 6*7?", schema)
        assert result["valid"] is True
        assert result["data"]["result"] == 42
        assert result["attempts"] == 1

    def test_generate_structured_retry(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "value": SchemaNode(type=SchemaType.INTEGER),
            },
        )

        call_count = [0]

        def mock_gen(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "Not JSON"
            return '{"value": 10}'

        result = self.handler.generate_structured(mock_gen, "test", schema)
        assert result["valid"] is True
        assert result["attempts"] == 2

    def test_generate_structured_all_retries_fail(self):
        schema = SchemaNode(
            type=SchemaType.OBJECT,
            properties={
                "x": SchemaNode(type=SchemaType.INTEGER),
            },
        )

        def mock_gen(prompt):
            return "never json"

        handler = StructuredOutputHandler(max_retries=2)
        result = handler.generate_structured(mock_gen, "test", schema)
        assert result["valid"] is False
        assert result["attempts"] == 2
