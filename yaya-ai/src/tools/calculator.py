"""Calculator tool — safely evaluates math expressions."""

import math
import re
from .base import BaseTool, ToolResult


# Allowed names for safe eval
_SAFE_NAMES = {
    'abs': abs, 'round': round, 'min': min, 'max': max,
    'sum': sum, 'pow': pow, 'int': int, 'float': float,
    'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'pi': math.pi, 'e': math.e,
    'floor': math.floor, 'ceil': math.ceil,
}


def _safe_eval(expr: str) -> str:
    """Evaluate a math expression safely."""
    # Remove anything that isn't math
    expr = expr.strip()
    expr = re.sub(r'[^0-9\s\+\-\*\/\(\)\.\,\%\^a-zA-Z_]', '', expr)
    expr = expr.replace('^', '**')  # support ^ for exponents
    result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return str(result)


class CalculatorTool(BaseTool):
    name = 'calculator'
    description = 'Evaluates mathematical expressions. Input: a math expression like "2 + 2" or "sqrt(144)".'

    def run(self, input_text: str) -> ToolResult:
        try:
            result = _safe_eval(input_text)
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=f'{input_text.strip()} = {result}',
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output='',
                error=f'Could not evaluate: {input_text}. Error: {e}',
            )
