from .calculator import CalculatorTool
from .code_runner import CodeRunnerTool
from .search import SearchTool
from .base import BaseTool, ToolResult

ALL_TOOLS = [CalculatorTool(), CodeRunnerTool(), SearchTool()]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}

__all__ = ['CalculatorTool', 'CodeRunnerTool', 'SearchTool', 'BaseTool', 'ToolResult', 'ALL_TOOLS', 'TOOL_MAP']
