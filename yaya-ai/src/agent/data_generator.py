"""Generate synthetic tool-use training data for SFT.

Creates conversation examples where the assistant uses tools to answer
questions, providing (input, target) pairs for supervised fine-tuning
on tool-use behavior.
"""

import json
import random
from typing import List, Dict, Any, Optional

from src.agent.tools import ToolRegistry, ToolCall, create_default_registry
from src.agent.chat_template import ChatTemplate


# ── Example Templates ──────────────────────────────────────────

CALCULATOR_EXAMPLES = [
    {"q": "What is 15% of 230?", "expr": "230 * 0.15", "follow_up": "15% of 230 is {result}."},
    {"q": "Calculate the square root of 144.", "expr": "sqrt(144)", "follow_up": "The square root of 144 is {result}."},
    {"q": "What is 2 to the power of 10?", "expr": "2**10", "follow_up": "2 to the power of 10 is {result}."},
    {"q": "How much is 3.14 * 5 squared?", "expr": "3.14 * 5**2", "follow_up": "3.14 times 5 squared is {result}."},
    {"q": "What is log base 2 of 256?", "expr": "log2(256)", "follow_up": "Log base 2 of 256 is {result}."},
    {"q": "Calculate sin(pi/4).", "expr": "sin(pi/4)", "follow_up": "sin(pi/4) is approximately {result}."},
    {"q": "What is 17 mod 5?", "expr": "17 % 5", "follow_up": "17 mod 5 is {result}."},
    {"q": "Compute 1000 divided by 7, rounded.", "expr": "round(1000/7, 2)", "follow_up": "1000 divided by 7 is approximately {result}."},
]

UNIT_CONVERT_EXAMPLES = [
    {"q": "Convert 100 degrees Fahrenheit to Celsius.", "v": 100, "from": "F", "to": "C",
     "follow_up": "100 degrees Fahrenheit is {result}."},
    {"q": "How many kilometers is 26.2 miles?", "v": 26.2, "from": "mi", "to": "km",
     "follow_up": "26.2 miles is {result}."},
    {"q": "Convert 180 pounds to kilograms.", "v": 180, "from": "lb", "to": "kg",
     "follow_up": "180 pounds is {result}."},
    {"q": "How many feet is 10 meters?", "v": 10, "from": "m", "to": "ft",
     "follow_up": "10 meters is {result}."},
    {"q": "Convert 0 Celsius to Kelvin.", "v": 0, "from": "C", "to": "K",
     "follow_up": "0 degrees Celsius is {result}."},
]

STRING_EXAMPLES = [
    {"q": "How many words are in 'The quick brown fox jumps over the lazy dog'?",
     "text": "The quick brown fox jumps over the lazy dog", "op": "word_count",
     "follow_up": "The sentence contains {result} words."},
    {"q": "Reverse the string 'hello world'.", "text": "hello world", "op": "reverse",
     "follow_up": "The reversed string is '{result}'."},
    {"q": "Convert 'machine learning' to uppercase.", "text": "machine learning", "op": "upper",
     "follow_up": "In uppercase: {result}."},
]

MULTI_STEP_EXAMPLES = [
    {
        "q": "If a room is 12 feet by 15 feet, what is its area in square meters?",
        "steps": [
            {"tool": "calculator", "args": {"expression": "12 * 15"}, "thought": "First, calculate the area in square feet."},
            {"tool": "unit_convert", "args": {"value": 180, "from_unit": "ft", "to_unit": "m"},
             "thought": "Now I need to convert. Actually, I should convert each dimension first."},
        ],
    },
    {
        "q": "What is 20% tip on a $85.50 meal, and what is the total?",
        "steps": [
            {"tool": "calculator", "args": {"expression": "85.50 * 0.20"}, "thought": "Calculate the tip amount."},
            {"tool": "calculator", "args": {"expression": "85.50 + 17.1"}, "thought": "Now add the tip to the meal cost."},
        ],
    },
]


class ToolUseDataGenerator:
    """Generate synthetic tool-use conversations for SFT training.

    Creates diverse examples of tool usage patterns including:
    - Single tool calls
    - Multi-step reasoning with multiple tool calls
    - Error handling
    - Tool selection from multiple options
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        seed: int = 42,
        system_prompt: Optional[str] = None,
    ):
        self.registry = registry or create_default_registry()
        self.rng = random.Random(seed)
        self.system_prompt = system_prompt or self.registry.get_system_prompt()

    def _make_conversation(
        self,
        user_msg: str,
        assistant_thought: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        follow_up_template: str,
    ) -> Optional[Dict[str, Any]]:
        """Create a single-tool conversation example."""
        # Execute the tool to get real results
        from src.agent.tools import ToolCall as TC
        call = TC(name=tool_name, arguments=tool_args)
        result = self.registry.execute(call)

        if not result.success:
            return None

        follow_up = follow_up_template.format(result=result.result)

        template = ChatTemplate(system_prompt=self.system_prompt)
        template.add_message("user", user_msg)
        template.add_message(
            "assistant",
            assistant_thought,
            tool_calls=[{"name": tool_name, "arguments": tool_args}],
        )
        template.add_tool_result(tool_name, result.result)
        template.add_message("assistant", follow_up)

        pairs = template.get_training_pairs()
        formatted = template.format()

        return {
            "conversation": formatted,
            "messages": template.messages,
            "training_pairs": pairs,
            "tools_used": [tool_name],
        }

    def generate_calculator_examples(self) -> List[Dict[str, Any]]:
        """Generate calculator tool-use examples."""
        examples = []
        for ex in CALCULATOR_EXAMPLES:
            result = self._make_conversation(
                user_msg=ex["q"],
                assistant_thought="Let me calculate that.",
                tool_name="calculator",
                tool_args={"expression": ex["expr"]},
                follow_up_template=ex["follow_up"],
            )
            if result:
                examples.append(result)
        return examples

    def generate_unit_examples(self) -> List[Dict[str, Any]]:
        """Generate unit conversion tool-use examples."""
        examples = []
        for ex in UNIT_CONVERT_EXAMPLES:
            result = self._make_conversation(
                user_msg=ex["q"],
                assistant_thought="I'll convert that for you.",
                tool_name="unit_convert",
                tool_args={"value": ex["v"], "from_unit": ex["from"], "to_unit": ex["to"]},
                follow_up_template=ex["follow_up"],
            )
            if result:
                examples.append(result)
        return examples

    def generate_string_examples(self) -> List[Dict[str, Any]]:
        """Generate string transformation tool-use examples."""
        examples = []
        for ex in STRING_EXAMPLES:
            result = self._make_conversation(
                user_msg=ex["q"],
                assistant_thought="Let me process that string.",
                tool_name="string_transform",
                tool_args={"text": ex["text"], "operation": ex["op"]},
                follow_up_template=ex["follow_up"],
            )
            if result:
                examples.append(result)
        return examples

    def generate_all(self) -> List[Dict[str, Any]]:
        """Generate all types of tool-use examples."""
        all_examples = []
        all_examples.extend(self.generate_calculator_examples())
        all_examples.extend(self.generate_unit_examples())
        all_examples.extend(self.generate_string_examples())
        self.rng.shuffle(all_examples)
        return all_examples

    def save_jsonl(self, examples: List[Dict[str, Any]], output_path: str):
        """Save examples to JSONL format for training."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                # Save just messages and training pairs (not full formatted text)
                record = {
                    "messages": ex["messages"],
                    "tools_used": ex["tools_used"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved {len(examples)} tool-use examples to {output_path}")
