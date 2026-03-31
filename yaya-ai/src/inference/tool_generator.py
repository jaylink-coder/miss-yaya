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

    Training data format:  <|calc|>47*83<|/calc|>=3901
    This class injects:                              =3901  (just the value)

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

        # Disable memory during the loop so that prompt boundary slicing works
        # cleanly (memory context prepended inside generate() would shift offsets).
        # We manually update memory once the full response is assembled.
        saved_memory = self.generator.memory
        self.generator.memory = None

        full_text = ""
        current_prompt = prompt
        calls_made = 0

        try:
            while calls_made <= self.max_tool_calls:
                # Generate from current prompt (no memory — we control the context)
                raw = self.generator.generate(current_prompt, config)

                # Extract only the new text.
                # With memory disabled, raw = decode(encode(current_prompt) + new_tokens).
                # tokenizer.decode(tokenizer.encode(s, add_bos=True)) should equal raw[:len(prompt_decoded)].
                tokenizer = self.generator.tokenizer
                prompt_ids = tokenizer.encode(current_prompt, add_bos=True)
                prompt_decoded = tokenizer.decode(prompt_ids)
                new_text = raw[len(prompt_decoded):]

                # Clean trailing stop tokens
                for stop in ["</s>", "<|endoftext|>"]:
                    new_text = new_text.split(stop)[0]

                full_text += new_text

                # Look for the FIRST calc call in the new output
                calc_match = re.search(
                    re.escape(CALC_OPEN) + r"(.+?)" + re.escape(CALC_CLOSE),
                    new_text,
                )
                if calc_match is None or calls_made >= self.max_tool_calls:
                    break

                expr = calc_match.group(1).strip()
                result = self.calc.run(expr)

                # result.output is "EXPR = VALUE" — extract just the numeric value
                # Training format: <|calc|>EXPR<|/calc|>=VALUE
                if result.success:
                    value = result.output.rsplit(" = ", 1)[-1]
                else:
                    value = "ERROR"

                if self.verbose:
                    if result.success:
                        print(f"  [calc] {expr} = {value}")
                    else:
                        print(f"  [calc] ERROR: {result.error}")

                # Inject result and continue generating
                injected = f"={value}\n"
                current_prompt = current_prompt + new_text + injected
                full_text += injected
                calls_made += 1

        finally:
            # Restore memory
            self.generator.memory = saved_memory

        # Update memory with the full exchange (prompt + response)
        if saved_memory is not None:
            saved_memory.extract_from_text(prompt)
            saved_memory.extract_from_text(full_text)

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
