"""Base class for all Yaya tools."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str = ''


class BaseTool:
    name: str = ''
    description: str = ''

    def run(self, input_text: str) -> ToolResult:
        raise NotImplementedError

    def __repr__(self):
        return f'Tool({self.name})'
