"""Structured output and JSON mode for Yaya AI.

Provides constrained decoding support to ensure model outputs conform
to a specified JSON schema. Essential for reliable function calling
and structured data extraction.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SchemaType(Enum):
    """Supported JSON Schema types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


@dataclass
class SchemaNode:
    """A node in a JSON schema tree."""
    type: SchemaType
    description: str = ""
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Any = None
    # For objects
    properties: Optional[Dict[str, "SchemaNode"]] = None
    # For arrays
    items: Optional["SchemaNode"] = None
    min_items: int = 0
    max_items: Optional[int] = None
    # For strings
    min_length: int = 0
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    # For numbers
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class JSONSchemaValidator:
    """Validate JSON data against a schema.

    Supports a subset of JSON Schema for practical use:
    - Type checking (string, integer, number, boolean, array, object, null)
    - Required properties
    - Enum constraints
    - String patterns and length constraints
    - Numeric range constraints
    - Array item validation
    - Nested object validation
    """

    def validate(self, data: Any, schema: SchemaNode) -> Tuple[bool, List[str]]:
        """Validate data against a schema.

        Args:
            data: Parsed JSON data.
            schema: Schema to validate against.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []
        self._validate_node(data, schema, "", errors)
        return len(errors) == 0, errors

    def _validate_node(
        self,
        data: Any,
        schema: SchemaNode,
        path: str,
        errors: List[str],
    ):
        # Null check
        if data is None:
            if schema.type != SchemaType.NULL and schema.required:
                errors.append(f"{path}: expected {schema.type.value}, got null")
            return

        # Type check
        type_map = {
            SchemaType.STRING: str,
            SchemaType.INTEGER: int,
            SchemaType.NUMBER: (int, float),
            SchemaType.BOOLEAN: bool,
            SchemaType.ARRAY: list,
            SchemaType.OBJECT: dict,
        }

        expected = type_map.get(schema.type)
        if expected:
            # Special case: bool is subclass of int in Python
            if schema.type == SchemaType.INTEGER and isinstance(data, bool):
                errors.append(f"{path}: expected integer, got boolean")
                return
            if schema.type == SchemaType.NUMBER and isinstance(data, bool):
                errors.append(f"{path}: expected number, got boolean")
                return
            if not isinstance(data, expected):
                errors.append(f"{path}: expected {schema.type.value}, got {type(data).__name__}")
                return

        # Enum check
        if schema.enum is not None and data not in schema.enum:
            errors.append(f"{path}: value {data!r} not in enum {schema.enum}")

        # String validations
        if schema.type == SchemaType.STRING and isinstance(data, str):
            if schema.min_length and len(data) < schema.min_length:
                errors.append(f"{path}: string length {len(data)} < min_length {schema.min_length}")
            if schema.max_length and len(data) > schema.max_length:
                errors.append(f"{path}: string length {len(data)} > max_length {schema.max_length}")
            if schema.pattern and not re.match(schema.pattern, data):
                errors.append(f"{path}: string does not match pattern {schema.pattern}")

        # Number validations
        if schema.type in (SchemaType.INTEGER, SchemaType.NUMBER) and isinstance(data, (int, float)):
            if schema.minimum is not None and data < schema.minimum:
                errors.append(f"{path}: value {data} < minimum {schema.minimum}")
            if schema.maximum is not None and data > schema.maximum:
                errors.append(f"{path}: value {data} > maximum {schema.maximum}")

        # Array validations
        if schema.type == SchemaType.ARRAY and isinstance(data, list):
            if len(data) < schema.min_items:
                errors.append(f"{path}: array length {len(data)} < min_items {schema.min_items}")
            if schema.max_items is not None and len(data) > schema.max_items:
                errors.append(f"{path}: array length {len(data)} > max_items {schema.max_items}")
            if schema.items:
                for i, item in enumerate(data):
                    self._validate_node(item, schema.items, f"{path}[{i}]", errors)

        # Object validations
        if schema.type == SchemaType.OBJECT and isinstance(data, dict):
            if schema.properties:
                for prop_name, prop_schema in schema.properties.items():
                    if prop_name in data:
                        self._validate_node(
                            data[prop_name], prop_schema,
                            f"{path}.{prop_name}" if path else prop_name, errors
                        )
                    elif prop_schema.required:
                        errors.append(
                            f"{path}.{prop_name}" if path else f"{prop_name}: required property missing"
                        )

    @classmethod
    def from_json_schema(cls, schema_dict: Dict[str, Any]) -> SchemaNode:
        """Parse a standard JSON Schema dict into a SchemaNode tree.

        Args:
            schema_dict: JSON Schema as a Python dict.

        Returns:
            Parsed SchemaNode.
        """
        return cls._parse_schema(schema_dict)

    @classmethod
    def _parse_schema(
        cls,
        schema: Dict[str, Any],
        required_fields: Optional[set] = None,
        field_name: str = "",
    ) -> SchemaNode:
        type_str = schema.get("type", "object")
        type_enum = SchemaType(type_str) if type_str in [t.value for t in SchemaType] else SchemaType.STRING

        node = SchemaNode(
            type=type_enum,
            description=schema.get("description", ""),
            enum=schema.get("enum"),
            default=schema.get("default"),
            min_length=schema.get("minLength", 0),
            max_length=schema.get("maxLength"),
            pattern=schema.get("pattern"),
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            min_items=schema.get("minItems", 0),
            max_items=schema.get("maxItems"),
        )

        # Required field check
        if required_fields is not None:
            node.required = field_name in required_fields

        # Object properties
        if "properties" in schema:
            req = set(schema.get("required", []))
            node.properties = {}
            for prop_name, prop_schema in schema["properties"].items():
                node.properties[prop_name] = cls._parse_schema(prop_schema, req, prop_name)

        # Array items
        if "items" in schema:
            node.items = cls._parse_schema(schema["items"])

        return node


class JSONOutputParser:
    """Parse and extract JSON from model output text.

    Handles various formats models might use:
    - Pure JSON
    - JSON in code blocks
    - JSON with surrounding text
    """

    @staticmethod
    def extract_json(text: str) -> Optional[str]:
        """Extract JSON string from model output.

        Tries multiple strategies:
        1. Direct JSON parse
        2. JSON in ```json code blocks
        3. First { ... } or [ ... ] block

        Returns:
            Extracted JSON string, or None if not found.
        """
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Strategy 2: Code block extraction
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match.strip())
                return match.strip()
            except json.JSONDecodeError:
                continue

        # Strategy 3: Find first complete JSON object or array
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue

            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break

        return None

    @staticmethod
    def parse(text: str) -> Optional[Any]:
        """Extract and parse JSON from text.

        Returns:
            Parsed JSON object, or None if extraction fails.
        """
        extracted = JSONOutputParser.extract_json(text)
        if extracted is None:
            return None
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            return None


class StructuredOutputHandler:
    """Handle structured output generation with schema validation.

    Wraps model generation to ensure outputs conform to a schema.
    Supports retry logic for malformed outputs.
    """

    def __init__(
        self,
        validator: Optional[JSONSchemaValidator] = None,
        max_retries: int = 3,
    ):
        self.validator = validator or JSONSchemaValidator()
        self.parser = JSONOutputParser()
        self.max_retries = max_retries

    def create_prompt_suffix(self, schema: SchemaNode) -> str:
        """Generate a prompt suffix that instructs the model to output JSON.

        Args:
            schema: Expected output schema.

        Returns:
            Prompt instruction string.
        """
        schema_desc = self._describe_schema(schema)
        return (
            "\n\nRespond with a JSON object matching this schema:\n"
            f"```json\n{schema_desc}\n```\n"
            "Output ONLY valid JSON, no additional text."
        )

    def _describe_schema(self, schema: SchemaNode, indent: int = 0) -> str:
        """Create a human-readable schema description."""
        prefix = "  " * indent

        if schema.type == SchemaType.OBJECT and schema.properties:
            lines = ["{"]
            for name, prop in schema.properties.items():
                req_marker = " (required)" if prop.required else " (optional)"
                if prop.type in (SchemaType.OBJECT, SchemaType.ARRAY):
                    inner = self._describe_schema(prop, indent + 1)
                    lines.append(f"{prefix}  \"{name}\": {inner},  // {prop.description}{req_marker}")
                else:
                    type_hint = prop.type.value
                    if prop.enum:
                        type_hint = " | ".join(f'"{e}"' for e in prop.enum)
                    lines.append(f"{prefix}  \"{name}\": <{type_hint}>,  // {prop.description}{req_marker}")
            lines.append(f"{prefix}" + "}")
            return "\n".join(lines)

        elif schema.type == SchemaType.ARRAY and schema.items:
            inner = self._describe_schema(schema.items, indent)
            return f"[{inner}, ...]"

        return f"<{schema.type.value}>"

    def process_output(
        self,
        text: str,
        schema: SchemaNode,
    ) -> Dict[str, Any]:
        """Parse and validate model output against schema.

        Args:
            text: Raw model output.
            schema: Expected output schema.

        Returns:
            Dict with 'data', 'valid', 'errors', 'raw'.
        """
        parsed = self.parser.parse(text)

        if parsed is None:
            return {
                "data": None,
                "valid": False,
                "errors": ["Failed to extract JSON from output"],
                "raw": text,
            }

        is_valid, errors = self.validator.validate(parsed, schema)

        return {
            "data": parsed,
            "valid": is_valid,
            "errors": errors,
            "raw": text,
        }

    def generate_structured(
        self,
        generate_fn,
        prompt: str,
        schema: SchemaNode,
    ) -> Dict[str, Any]:
        """Generate structured output with retries.

        Args:
            generate_fn: Model generation function.
            prompt: User prompt.
            schema: Expected output schema.

        Returns:
            Dict with 'data', 'valid', 'errors', 'attempts'.
        """
        full_prompt = prompt + self.create_prompt_suffix(schema)

        for attempt in range(self.max_retries):
            output = generate_fn(full_prompt)
            result = self.process_output(output, schema)

            if result["valid"]:
                result["attempts"] = attempt + 1
                return result

            # On retry, add error feedback to prompt
            if attempt < self.max_retries - 1:
                error_msg = "; ".join(result["errors"])
                full_prompt = (
                    f"{prompt}\n\n"
                    f"Previous attempt had errors: {error_msg}\n"
                    "Please fix and output valid JSON only."
                    + self.create_prompt_suffix(schema)
                )

        result["attempts"] = self.max_retries
        return result
