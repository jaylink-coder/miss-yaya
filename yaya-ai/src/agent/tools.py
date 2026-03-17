"""Tool definitions and registry for Yaya function calling.

Defines the schema for tools/functions that the model can invoke,
a registry for managing available tools, and built-in tool implementations.
"""

import json
import math
import re
import ast
import operator
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict


# ── Tool Schema ────────────────────────────────────────────────

@dataclass
class ParameterSchema:
    """Schema for a single function parameter."""
    name: str
    type: str               # string, integer, number, boolean, array, object
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {"type": self.type, "description": self.description}
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a callable tool/function.

    Compatible with OpenAI function calling format for interoperability.
    """
    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    category: str = "general"
    returns: str = "string"

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_prompt_format(self) -> str:
        """Convert to a human-readable prompt description."""
        params_str = []
        for p in self.parameters:
            req = " (required)" if p.required else " (optional)"
            params_str.append(f"    - {p.name} ({p.type}){req}: {p.description}")

        params_block = "\n".join(params_str) if params_str else "    (no parameters)"

        return (
            f"Function: {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Parameters:\n{params_block}\n"
            f"  Returns: {self.returns}"
        )


@dataclass
class ToolCall:
    """A parsed tool call from the model's output."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments})

    @classmethod
    def from_json(cls, text: str) -> "ToolCall":
        data = json.loads(text)
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
            call_id=data.get("call_id"),
        )


@dataclass
class ToolResult:
    """Result from executing a tool call."""
    call_id: Optional[str]
    name: str
    result: str
    success: bool = True
    error: Optional[str] = None


# ── Tool Registry ──────────────────────────────────────────────

class ToolRegistry:
    """Registry for managing available tools.

    Stores tool definitions and their implementations.
    Handles validation, execution, and schema generation.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._implementations: Dict[str, Callable] = {}

    def register(
        self,
        definition: ToolDefinition,
        implementation: Callable,
    ):
        """Register a tool with its implementation.

        Args:
            definition: Tool schema definition.
            implementation: Callable that executes the tool.
        """
        self._tools[definition.name] = definition
        self._implementations[definition.name] = implementation

    def register_function(
        self,
        name: str,
        description: str,
        parameters: List[ParameterSchema],
        implementation: Callable,
        category: str = "general",
    ):
        """Convenience method to register a tool from parts."""
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
        )
        self.register(definition, implementation)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """List all registered tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def validate_call(self, call: ToolCall) -> Optional[str]:
        """Validate a tool call against its schema.

        Returns:
            None if valid, error message string if invalid.
        """
        definition = self._tools.get(call.name)
        if definition is None:
            return f"Unknown tool: {call.name}"

        # Check required parameters
        param_map = {p.name: p for p in definition.parameters}
        for param in definition.parameters:
            if param.required and param.name not in call.arguments:
                return f"Missing required parameter: {param.name}"

        # Check for unknown parameters
        for arg_name in call.arguments:
            if arg_name not in param_map:
                return f"Unknown parameter: {arg_name}"

        # Basic type checking
        for arg_name, arg_value in call.arguments.items():
            param = param_map.get(arg_name)
            if param and param.enum and arg_value not in param.enum:
                return f"Invalid value for {arg_name}: {arg_value}. Must be one of {param.enum}"

        return None

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a validated tool call.

        Args:
            call: The tool call to execute.

        Returns:
            ToolResult with the execution outcome.
        """
        # Validate first
        error = self.validate_call(call)
        if error:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result="",
                success=False,
                error=error,
            )

        impl = self._implementations.get(call.name)
        if impl is None:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result="",
                success=False,
                error=f"No implementation for tool: {call.name}",
            )

        try:
            result = impl(**call.arguments)
            result_str = str(result) if not isinstance(result, str) else result
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result=result_str,
                success=True,
            )
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                result="",
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

    def get_system_prompt(self) -> str:
        """Generate a system prompt describing all available tools."""
        if not self._tools:
            return ""

        lines = ["You have access to the following tools:\n"]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_format())
            lines.append("")

        lines.append(
            "To use a tool, output a JSON object between <tool_call> and </tool_call> tags:\n"
            '<tool_call>\n{"name": "tool_name", "arguments": {"arg1": "value1"}}\n</tool_call>\n\n'
            "The tool result will be provided between <tool_result> and </tool_result> tags."
        )
        return "\n".join(lines)

    def get_openai_schema(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in OpenAI function calling format."""
        return [t.to_openai_format() for t in self._tools.values()]


# ── Built-in Tools ─────────────────────────────────────────────

def _calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, **, %, sqrt, abs, sin, cos, tan, log, pi, e
    """
    # Allowed names for safe evaluation
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "pi": math.pi, "e": math.e, "inf": math.inf,
        "pow": pow, "int": int, "float": float,
    }

    # Allowed AST node types
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd, ast.Load, ast.Tuple, ast.List,
    }

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Syntax error: {e}"

    # Validate all nodes
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            return f"Unsupported operation: {type(node).__name__}"
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return f"Unknown name: {node.id}"

    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, allowed_names)
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _json_extract(json_text: str, path: str) -> str:
    """Extract a value from JSON using dot-notation path.

    Args:
        json_text: JSON string.
        path: Dot-notation path (e.g., "data.items.0.name").

    Returns:
        Extracted value as string.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    keys = path.split(".")
    current = data

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return f"Key not found: {key}"
            current = current[key]
        elif isinstance(current, list):
            try:
                idx = int(key)
                current = current[idx]
            except (ValueError, IndexError):
                return f"Invalid list index: {key}"
        else:
            return f"Cannot traverse into {type(current).__name__}"

    return json.dumps(current) if isinstance(current, (dict, list)) else str(current)


def _string_transform(text: str, operation: str) -> str:
    """Apply a string transformation.

    Args:
        text: Input string.
        operation: One of: upper, lower, title, strip, reverse, length, word_count
    """
    ops = {
        "upper": lambda t: t.upper(),
        "lower": lambda t: t.lower(),
        "title": lambda t: t.title(),
        "strip": lambda t: t.strip(),
        "reverse": lambda t: t[::-1],
        "length": lambda t: str(len(t)),
        "word_count": lambda t: str(len(t.split())),
    }
    if operation not in ops:
        return f"Unknown operation: {operation}. Available: {', '.join(ops.keys())}"
    return ops[operation](text)


def _datetime_info(query: str = "now") -> str:
    """Get date/time information.

    Args:
        query: One of: now, date, time, weekday, timestamp
    """
    import datetime
    now = datetime.datetime.now()

    queries = {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timestamp": str(int(now.timestamp())),
        "iso": now.isoformat(),
    }
    return queries.get(query, f"Unknown query: {query}. Available: {', '.join(queries.keys())}")


def _unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.

    Supports: length (m/km/mi/ft/in), weight (kg/lb/oz/g), temperature (C/F/K)
    """
    # Length in meters
    length_to_m = {
        "m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048,
        "in": 0.0254, "cm": 0.01, "mm": 0.001, "yd": 0.9144,
    }
    # Weight in kg
    weight_to_kg = {
        "kg": 1.0, "g": 0.001, "mg": 0.000001, "lb": 0.453592,
        "oz": 0.0283495, "ton": 1000.0,
    }

    # Temperature
    if from_unit in ("C", "F", "K") and to_unit in ("C", "F", "K"):
        # Convert to Celsius first
        if from_unit == "F":
            c = (value - 32) * 5 / 9
        elif from_unit == "K":
            c = value - 273.15
        else:
            c = value

        # Convert from Celsius to target
        if to_unit == "F":
            result = c * 9 / 5 + 32
        elif to_unit == "K":
            result = c + 273.15
        else:
            result = c

        return f"{round(result, 4)} {to_unit}"

    # Length
    if from_unit in length_to_m and to_unit in length_to_m:
        meters = value * length_to_m[from_unit]
        result = meters / length_to_m[to_unit]
        return f"{round(result, 6)} {to_unit}"

    # Weight
    if from_unit in weight_to_kg and to_unit in weight_to_kg:
        kg = value * weight_to_kg[from_unit]
        result = kg / weight_to_kg[to_unit]
        return f"{round(result, 6)} {to_unit}"

    return f"Cannot convert from {from_unit} to {to_unit}"


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools."""
    registry = ToolRegistry()

    # Calculator
    registry.register_function(
        name="calculator",
        description="Evaluate a mathematical expression. Supports arithmetic, trigonometry, logarithms, and constants like pi and e.",
        parameters=[
            ParameterSchema(name="expression", type="string",
                          description="Mathematical expression to evaluate, e.g., 'sqrt(144) + 2**3'"),
        ],
        implementation=_calculator,
        category="math",
    )

    # JSON extractor
    registry.register_function(
        name="json_extract",
        description="Extract a value from a JSON string using dot-notation path.",
        parameters=[
            ParameterSchema(name="json_text", type="string",
                          description="JSON string to extract from"),
            ParameterSchema(name="path", type="string",
                          description="Dot-notation path, e.g., 'data.items.0.name'"),
        ],
        implementation=_json_extract,
        category="data",
    )

    # String transform
    registry.register_function(
        name="string_transform",
        description="Apply a transformation to a string: upper, lower, title, strip, reverse, length, word_count.",
        parameters=[
            ParameterSchema(name="text", type="string", description="Input text"),
            ParameterSchema(name="operation", type="string",
                          description="Transformation to apply",
                          enum=["upper", "lower", "title", "strip", "reverse", "length", "word_count"]),
        ],
        implementation=_string_transform,
        category="text",
    )

    # Date/time
    registry.register_function(
        name="datetime_info",
        description="Get current date/time information.",
        parameters=[
            ParameterSchema(name="query", type="string",
                          description="What to get: now, date, time, weekday, timestamp, iso",
                          required=False, default="now"),
        ],
        implementation=_datetime_info,
        category="utility",
    )

    # Unit converter
    registry.register_function(
        name="unit_convert",
        description="Convert between units. Supports length (m/km/mi/ft/in/cm), weight (kg/lb/oz/g), temperature (C/F/K).",
        parameters=[
            ParameterSchema(name="value", type="number", description="Numeric value to convert"),
            ParameterSchema(name="from_unit", type="string", description="Source unit"),
            ParameterSchema(name="to_unit", type="string", description="Target unit"),
        ],
        implementation=_unit_convert,
        category="utility",
    )

    return registry
