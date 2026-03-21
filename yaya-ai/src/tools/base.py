"""Base class for all Yaya tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: str = ''


class BaseTool(ABC):
    name: str = ''
    description: str = ''

    @abstractmethod
    def run(self, input_text: str) -> ToolResult:
        """Execute the tool with the given input. Must be overridden."""

    def __repr__(self):
        return f'Tool({self.name})'
