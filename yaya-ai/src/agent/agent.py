"""Yaya Agent Loop — Phase 4: Planning and Tool Use."""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.tools import TOOL_MAP, ToolResult
from src.agent.tools import ToolCall, ToolRegistry, create_default_registry
from src.agent.chat_template import TOOL_CALL_OPEN, TOOL_CALL_CLOSE
from src.tokenizer.tokenizer import ASSISTANT_TOKEN, USER_TOKEN, SYSTEM_TOKEN

TOOL_CALL_PATTERN = re.compile(r'<\|tool\|>(.*?):(.*?)<\|/tool\|>', re.DOTALL)

# Pattern for ChatML-style tool calls: <tool_call>\n{json}\n</tool_call>
_CHATML_TOOL_PATTERN = re.compile(
    re.escape(TOOL_CALL_OPEN) + r'\s*(.*?)\s*' + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)


class ToolCallParser:
    """Parse tool calls from model output text."""

    @staticmethod
    def parse(text: str) -> List[ToolCall]:
        calls = []
        for match in _CHATML_TOOL_PATTERN.finditer(text):
            raw = match.group(1).strip()
            try:
                data = json.loads(raw)
                calls.append(ToolCall(
                    name=data["name"],
                    arguments=data.get("arguments", {}),
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls

    @staticmethod
    def has_tool_call(text: str) -> bool:
        return TOOL_CALL_OPEN in text

    @staticmethod
    def extract_thought(text: str) -> str:
        idx = text.find(TOOL_CALL_OPEN)
        if idx == -1:
            return text
        return text[:idx].strip()


@dataclass
class AgentConfig:
    max_steps: int = 5
    verbose: bool = False


@dataclass
class _AgentStep:
    response: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)


class SimpleAgent:
    """Lightweight agent that executes tool calls without a generate loop."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or create_default_registry()

    def execute_call(self, tool_name: str, **kwargs) -> Any:
        call = ToolCall(name=tool_name, arguments=kwargs)
        return self.registry.execute(call)

    def execute_text(self, text: str) -> List[Any]:
        calls = ToolCallParser.parse(text)
        return [self.registry.execute(c) for c in calls]


class Agent:
    """Full agent with generate loop, tool use, and history tracking."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.generate_fn = generate_fn
        self.config = config or AgentConfig()
        self.registry = tool_registry or create_default_registry()
        self.history: List[_AgentStep] = []

    def run(self, query: str) -> str:
        self.history = []
        prompt = query

        for _ in range(self.config.max_steps):
            response = self.generate_fn(prompt)
            calls = ToolCallParser.parse(response)

            step = _AgentStep(response=response, tool_calls=calls)

            if not calls:
                # No tool calls — final response
                thought = ToolCallParser.extract_thought(response)
                step.response = thought if thought else response
                self.history.append(step)
                return step.response

            # Execute tool calls and build context for next iteration
            results = []
            for call in calls:
                result = self.registry.execute(call)
                results.append(result)
            step.tool_results = results
            self.history.append(step)

            # Build next prompt with tool results
            result_texts = []
            for call, result in zip(calls, results):
                r_text = result.result if result.success else f"Error: {result.error}"
                result_texts.append(f"Tool {call.name} returned: {r_text}")
            prompt = query + "\n" + "\n".join(result_texts)

        # Max steps reached — return last response
        if self.history:
            return ToolCallParser.extract_thought(self.history[-1].response)
        return ""

    def get_trace(self) -> List[Dict[str, Any]]:
        trace = []
        for step in self.history:
            entry = {"response": step.response}
            if step.tool_calls:
                entry["tool_calls"] = [
                    {"name": c.name, "arguments": c.arguments}
                    for c in step.tool_calls
                ]
                entry["tool_results"] = [
                    str(r) for r in step.tool_results
                ]
            trace.append(entry)
        return trace

AGENT_SYSTEM_PROMPT = """You are Yaya, a helpful and intelligent AI assistant with access to tools.

You can use the following tools:
- <|tool|>calculator: EXPRESSION<|/tool|>
- <|tool|>search: QUERY<|/tool|>
- <|tool|>code_runner: CODE<|/tool|>

Think step by step. Use tools when needed. Be honest, helpful, and friendly."""


class YayaAgent:
    def __init__(self, generator, tokenizer, memory=None, max_iterations=5):
        self.generator = generator
        self.tokenizer = tokenizer
        self.memory = memory
        self.max_iter = max_iterations

    def _extract_tool_calls(self, text):
        return [(m.group(1).strip(), m.group(2).strip()) for m in TOOL_CALL_PATTERN.finditer(text)]

    def _execute_tool(self, tool_name, tool_input):
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            from src.tools.base import ToolResult as TR
            return TR(tool_name=tool_name, success=False, output='', error=f'Unknown tool: {tool_name}')
        return tool.run(tool_input)

    def _inject_tool_result(self, text, result):
        pattern = re.compile(r'<\|tool\|>.*?<\|/tool\|>', re.DOTALL)
        result_text = (f'<|tool_result|>{result.output}<|/tool_result|>' if result.success
                       else f'<|tool_result|>Error: {result.error}<|/tool_result|>')
        return pattern.sub(result_text, text, count=1)

    def run(self, user_message, conversation_history, max_new_tokens=300, temperature=0.7):
        system = AGENT_SYSTEM_PROMPT
        if self.memory:
            mem_context = self.memory.format_for_prompt(user_message)
            if mem_context:
                system += '\n\n' + mem_context

        history = ([{'role': 'system', 'content': system}]
                   + conversation_history
                   + [{'role': 'user', 'content': user_message}])
        accumulated = ''

        for _ in range(self.max_iter):
            prompt = self.tokenizer.format_chat(history) + '<|assistant|>\n' + accumulated
            generated = self.generator.generate(prompt, max_new_tokens=max_new_tokens,
                                                temperature=temperature, top_p=0.9)
            if '<|assistant|>' in generated:
                new_text = generated.split('<|assistant|>')[-1]
            elif prompt in generated:
                new_text = generated[len(prompt):]
            else:
                new_text = generated
            for stop in ['<|user|>', '<|system|>', '</s>']:
                new_text = new_text.split(stop)[0]
            accumulated += new_text
            tool_calls = self._extract_tool_calls(accumulated)
            if not tool_calls:
                break
            tool_name, tool_input = tool_calls[0]
            result = self._execute_tool(tool_name, tool_input)
            accumulated = self._inject_tool_result(accumulated, result)

        return accumulated.strip()
