"""Tool-augmented text generator.

Wraps TextGenerator and intercepts <|calc|>EXPR<|/calc|> tokens in the
generated output, runs the CalculatorTool, and injects the result back
into the generation context before continuing.

This gives Yaya exact arithmetic without relying on the model's weights.
"""

import re
from typing import Optional

from src.inference.generator import TextGenerator, GenerationConfig
from src.tools.calculator import CalculatorTool

# Special tokens for tool calls
CALC_OPEN  = "<|calc|>"
CALC_CLOSE = "<|/calc|>"


class ToolAugmentedGenerator:
    """Wraps TextGenerator with calculator tool support.

    During generation, if the model emits:
        <|calc|>EXPR<|/calc|>
    the expression is evaluated and the result is appended as:
        =RESULT
    Then generation resumes from that point.

    The model can make up to `max_tool_calls` calculator calls per response.
    """

    def __init__(
        self,
        generator: TextGenerator,
        max_tool_calls: int = 8,
        verbose: bool = False,
    ):
        self.generator = generator
        self.calc = CalculatorTool()
        self.max_tool_calls = max_tool_calls
        self.verbose = verbose

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate a response, intercepting and executing calculator calls."""
        if config is None:
            config = GenerationConfig()

        full_text = ""
        current_prompt = prompt
        calls_made = 0

        while calls_made <= self.max_tool_calls:
            # Generate from current prompt
            raw = self.generator.generate(current_prompt, config)

            # Extract only the new text (after the prompt)
            tokenizer = self.generator.tokenizer
            prompt_ids = tokenizer.encode(current_prompt, add_bos=True)
            prompt_decoded = tokenizer.decode(prompt_ids)
            new_text = raw[len(prompt_decoded):]

            # Clean any trailing stop tokens
            for stop in ["</s>", "<|endoftext|>"]:
                new_text = new_text.split(stop)[0]

            full_text += new_text

            # Look for a calc call in what was just generated
            calc_match = re.search(
                re.escape(CALC_OPEN) + r"(.+?)" + re.escape(CALC_CLOSE),
                new_text,
            )
            if calc_match is None or calls_made >= self.max_tool_calls:
                break

            expr = calc_match.group(1).strip()
            result = self.calc.run(expr)

            if self.verbose:
                if result.success:
                    print(f"  [calc] {expr} = {result.output}")
                else:
                    print(f"  [calc] ERROR: {result.error}")

            # Inject result and continue generating from there
            injected = f"={result.output}\n" if result.success else f"=ERROR\n"

            # Rebuild prompt: original prompt + everything generated so far + injected result
            current_prompt = current_prompt + new_text + injected
            full_text += injected
            calls_made += 1

        return full_text

    def format_help(self) -> str:
        """Return a description of available tools for the system prompt."""
        return (
            "You have access to a calculator. To use it, write:\n"
            "<|calc|>EXPRESSION<|/calc|>\n"
            "The result will be shown as =RESULT. "
            "Example: <|calc|>47 * 83<|/calc|>=3901\n"
            "Use this for any arithmetic to ensure accuracy."
        )
