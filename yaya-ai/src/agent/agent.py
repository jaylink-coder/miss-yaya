"""Yaya Agent — multi-step reasoning with tool use.

Implements the plan-execute-observe-respond cycle that enables
the model to use tools iteratively to solve complex tasks.
"""

import json
import re
import time
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field

from src.agent.tools import ToolRegistry, ToolCall, ToolResult, create_default_registry
from src.agent.chat_template import (
    ChatTemplate,
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_RESULT_OPEN,
    TOOL_RESULT_CLOSE,
)


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    response: str = ""
    timestamp: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_steps: int = 10
    max_retries: int = 2
    timeout_seconds: float = 30.0
    verbose: bool = True
    stop_on_error: bool = False
    parallel_tools: bool = False


class ToolCallParser:
    """Parse tool calls from model-generated text.

    Extracts structured tool calls from text containing
    <tool_call>...</tool_call> tags.
    """

    @staticmethod
    def parse(text: str) -> List[ToolCall]:
        """Extract all tool calls from text.

        Args:
            text: Model output text potentially containing tool calls.

        Returns:
            List of parsed ToolCall objects.
        """
        calls = []
        # Find all tool call blocks
        pattern = re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE)
        matches = re.finditer(pattern, text, re.DOTALL)

        for i, match in enumerate(matches):
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
                call = ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    call_id=f"call_{i}",
                )
                calls.append(call)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return calls

    @staticmethod
    def has_tool_call(text: str) -> bool:
        """Check if text contains any tool call tags."""
        return TOOL_CALL_OPEN in text

    @staticmethod
    def extract_thought(text: str) -> str:
        """Extract the thinking/reasoning text before any tool calls."""
        if TOOL_CALL_OPEN in text:
            return text[:text.index(TOOL_CALL_OPEN)].strip()
        return text.strip()


class Agent:
    """Multi-step reasoning agent with tool use.

    Manages the plan-execute-observe-respond loop:
    1. Model generates text (possibly with tool calls)
    2. Tool calls are parsed and executed
    3. Results are fed back to the model
    4. Repeat until model produces a final response (no tool calls)
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[AgentConfig] = None,
    ):
        """Initialize agent.

        Args:
            generate_fn: Function that takes a prompt string and returns
                        generated text. This wraps the model's generation.
            tool_registry: Registry of available tools. Uses defaults if None.
            config: Agent behavior configuration.
        """
        self.generate_fn = generate_fn
        self.registry = tool_registry or create_default_registry()
        self.config = config or AgentConfig()
        self.parser = ToolCallParser()
        self.history: List[AgentStep] = []

    def _log(self, msg: str):
        if self.config.verbose:
            print(f"  [Agent] {msg}")

    def run(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation: Optional[ChatTemplate] = None,
    ) -> str:
        """Run the agent on a user message.

        Args:
            user_message: The user's input message.
            system_prompt: Optional system prompt (auto-generated if None).
            conversation: Optional existing conversation context.

        Returns:
            The agent's final response text.
        """
        self.history = []

        # Build conversation
        if conversation is None:
            if system_prompt is None:
                system_prompt = self.registry.get_system_prompt()
            conversation = ChatTemplate(system_prompt=system_prompt)

        conversation.add_message("user", user_message)

        for step_num in range(1, self.config.max_steps + 1):
            step = AgentStep(step_number=step_num, timestamp=time.time())
            self._log(f"Step {step_num}/{self.config.max_steps}")

            # Generate model response
            prompt = conversation.format_for_generation()
            generated = self.generate_fn(prompt)

            # Check for tool calls
            if not self.parser.has_tool_call(generated):
                # Final response — no tool calls
                step.response = generated.strip()
                self.history.append(step)
                conversation.add_message("assistant", step.response)
                self._log(f"Final response: {step.response[:100]}...")
                return step.response

            # Parse tool calls
            step.thought = self.parser.extract_thought(generated)
            tool_calls = self.parser.parse(generated)

            if not tool_calls:
                # Malformed tool call — treat as final response
                step.response = generated.strip()
                self.history.append(step)
                conversation.add_message("assistant", step.response)
                return step.response

            self._log(f"Thought: {step.thought[:80]}..." if step.thought else "No thought")
            self._log(f"Tool calls: {[tc.name for tc in tool_calls]}")

            # Execute tool calls
            tc_dicts = [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
            conversation.add_message("assistant", step.thought, tool_calls=tc_dicts)

            for tc in tool_calls:
                step.tool_call = tc
                result = self.registry.execute(tc)
                step.tool_result = result

                self._log(f"  {tc.name}({tc.arguments}) -> {result.result[:80]}...")

                conversation.add_tool_result(
                    name=result.name,
                    result=result.result if result.success else f"Error: {result.error}",
                    success=result.success,
                )

                if not result.success and self.config.stop_on_error:
                    step.response = f"Tool error: {result.error}"
                    self.history.append(step)
                    return step.response

            self.history.append(step)

        # Max steps reached
        self._log("Max steps reached, generating final response")
        prompt = conversation.format_for_generation()
        final = self.generate_fn(prompt)
        return final.strip()

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the full execution trace for debugging."""
        trace = []
        for step in self.history:
            entry = {
                "step": step.step_number,
                "thought": step.thought,
                "response": step.response,
            }
            if step.tool_call:
                entry["tool_call"] = {
                    "name": step.tool_call.name,
                    "arguments": step.tool_call.arguments,
                }
            if step.tool_result:
                entry["tool_result"] = {
                    "name": step.tool_result.name,
                    "result": step.tool_result.result,
                    "success": step.tool_result.success,
                }
            trace.append(entry)
        return trace


class SimpleAgent:
    """Simplified agent for single-turn tool use without a model.

    Useful for testing and for building tool-use training data.
    Executes tool calls directly without model generation.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or create_default_registry()
        self.parser = ToolCallParser()

    def execute_text(self, text: str) -> List[ToolResult]:
        """Parse and execute any tool calls found in text.

        Args:
            text: Text potentially containing tool call tags.

        Returns:
            List of ToolResult objects.
        """
        calls = self.parser.parse(text)
        results = []
        for call in calls:
            result = self.registry.execute(call)
            results.append(result)
        return results

    def execute_call(self, name: str, **kwargs) -> ToolResult:
        """Execute a single tool call by name.

        Args:
            name: Tool name.
            **kwargs: Tool arguments.

        Returns:
            ToolResult.
        """
        call = ToolCall(name=name, arguments=kwargs)
        return self.registry.execute(call)
