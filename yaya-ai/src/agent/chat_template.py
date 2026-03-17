"""Chat template for Yaya with tool-use support.

Handles formatting of multi-turn conversations with system messages,
user and assistant turns, tool calls, and tool results.
"""

import json
from typing import List, Dict, Optional, Any


def _close(tag: str) -> str:
    """Build a closing XML-style tag."""
    return "<" + chr(47) + tag + ">"


# Chat role tags (constructed to avoid literal closing-tag sequences)
SYSTEM_OPEN = "<|system|>"
SYSTEM_CLOSE = _close("|system|")
USER_OPEN = "<|user|>"
USER_CLOSE = _close("|user|")
ASSISTANT_OPEN = "<|assistant|>"
ASSISTANT_CLOSE = _close("|assistant|")
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = _close("tool_call")
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = _close("tool_result")


def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Format a tool call for inclusion in model output."""
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"{TOOL_CALL_OPEN}\n{payload}\n{TOOL_CALL_CLOSE}"


def format_tool_result(name: str, result: str, success: bool = True) -> str:
    """Format a tool result for inclusion in the conversation."""
    status = "success" if success else "error"
    payload = json.dumps({"name": name, "status": status, "result": result})
    return f"{TOOL_RESULT_OPEN}\n{payload}\n{TOOL_RESULT_CLOSE}"


class ChatTemplate:
    """Format multi-turn conversations for the Yaya model.

    Supports roles: system, user, assistant
    Supports tool calling and tool results
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add a message to the conversation."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, name: str, result: str, success: bool = True):
        """Add a tool result to the conversation."""
        self.messages.append({
            "role": "tool",
            "name": name,
            "content": result,
            "success": success,
        })

    def format(self) -> str:
        """Format the full conversation into a string for tokenization."""
        parts = []

        for msg in self.messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"{SYSTEM_OPEN}\n{content}\n{SYSTEM_CLOSE}")

            elif role == "user":
                parts.append(f"{USER_OPEN}\n{content}\n{USER_CLOSE}")

            elif role == "assistant":
                # Check for tool calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    if content:
                        tc_parts.append(content)
                    for tc in tool_calls:
                        tc_json = json.dumps({"name": tc["name"], "arguments": tc.get("arguments", {})})
                        tc_parts.append(f"{TOOL_CALL_OPEN}\n{tc_json}\n{TOOL_CALL_CLOSE}")
                    parts.append(f"{ASSISTANT_OPEN}\n" + "\n".join(tc_parts) + f"\n{ASSISTANT_CLOSE}")
                else:
                    parts.append(f"{ASSISTANT_OPEN}\n{content}\n{ASSISTANT_CLOSE}")

            elif role == "tool":
                name = msg.get("name", "unknown")
                success = msg.get("success", True)
                status = "success" if success else "error"
                payload = json.dumps({"name": name, "status": status, "result": content})
                parts.append(f"{TOOL_RESULT_OPEN}\n{payload}\n{TOOL_RESULT_CLOSE}")

        return "\n".join(parts)

    def format_for_generation(self) -> str:
        """Format conversation with an open assistant turn for generation."""
        base = self.format()
        return base + f"\n{ASSISTANT_OPEN}\n"

    def get_training_pairs(self) -> List[Dict[str, str]]:
        """Extract (input, target) pairs for SFT training.

        Each assistant message becomes a target, with all prior
        messages as the input context.
        """
        pairs = []
        for i, msg in enumerate(self.messages):
            if msg["role"] == "assistant":
                # Build context from all messages up to this point
                context_template = ChatTemplate()
                context_template.messages = self.messages[:i]
                context = context_template.format_for_generation()

                # Target is the assistant response
                target = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_strs = []
                    if target:
                        tc_strs.append(target)
                    for tc in tool_calls:
                        tc_json = json.dumps({"name": tc["name"], "arguments": tc.get("arguments", {})})
                        tc_strs.append(f"{TOOL_CALL_OPEN}\n{tc_json}\n{TOOL_CALL_CLOSE}")
                    target = "\n".join(tc_strs)

                pairs.append({"input": context, "target": target + f"\n{ASSISTANT_CLOSE}"})

        return pairs

    @classmethod
    def from_messages(cls, messages: List[Dict[str, Any]]) -> "ChatTemplate":
        """Create a ChatTemplate from a list of message dicts."""
        system = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break
        template = cls(system_prompt=system)
        for msg in messages:
            if msg.get("role") == "system":
                continue
            role = msg["role"]
            if role == "tool":
                template.add_tool_result(msg.get("name", ""), msg.get("content", ""), msg.get("success", True))
            else:
                template.add_message(role, msg.get("content", ""), msg.get("tool_calls"))
        return template
