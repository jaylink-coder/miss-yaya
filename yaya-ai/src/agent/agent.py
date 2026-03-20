"""Yaya Agent Loop — Phase 4: Planning and Tool Use."""
import re
from src.tools import TOOL_MAP, ToolResult

TOOL_CALL_PATTERN = re.compile(r'<\|tool\|>(.*?):(.*?)<\|/tool\|>', re.DOTALL)

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
