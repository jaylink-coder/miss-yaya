"""Yaya Super-Brain — Chain-of-Thought, ReAct, and Self-Reflection.

Cognitive architecture:
  1. THINK  — internal reasoning trace before answering
  2. ACT    — tool use / computation inside the thought
  3. VERIFY — self-check: is the answer consistent / correct?
  4. ANSWER — clean final response

Token convention in chat:
  <|think|> ... <|/think|>   — internal scratchpad (not shown to user)
  <|plan|>  ... <|/plan|>    — high-level plan for multi-step tasks
  <|verify|>... <|/verify|>  — self-check trace

These are injected into prompts to teach the model the format,
and stripped from final output shown to the user.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Token boundaries ──────────────────────────────────────────────────────────
THINK_OPEN    = "<|think|>"
THINK_CLOSE   = "<|/think|>"
PLAN_OPEN     = "<|plan|>"
PLAN_CLOSE    = "<|/plan|>"
VERIFY_OPEN   = "<|verify|>"
VERIFY_CLOSE  = "<|/verify|>"


def _extract_block(text: str, open_tag: str, close_tag: str) -> Tuple[str, str]:
    """Return (block_content, text_without_block)."""
    pattern = re.compile(re.escape(open_tag) + r"(.*?)" + re.escape(close_tag), re.DOTALL)
    match = pattern.search(text)
    if match:
        block = match.group(1).strip()
        rest = pattern.sub("", text, count=1).strip()
        return block, rest
    return "", text


def strip_reasoning(text: str) -> str:
    """Remove all internal reasoning blocks — return only the final answer."""
    for open_t, close_t in [
        (THINK_OPEN, THINK_CLOSE),
        (PLAN_OPEN,  PLAN_CLOSE),
        (VERIFY_OPEN, VERIFY_CLOSE),
    ]:
        text = re.sub(
            re.escape(open_t) + r".*?" + re.escape(close_t), "", text, flags=re.DOTALL
        )
    return text.strip()


# ── Working memory (per-conversation scratchpad) ──────────────────────────────

@dataclass
class WorkingMemory:
    """Tracks key facts extracted during the current conversation."""

    facts: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)   # name → description
    goals: List[str] = field(default_factory=list)

    def add_fact(self, fact: str):
        if fact and fact not in self.facts:
            self.facts.append(fact)

    def add_entity(self, name: str, description: str):
        self.entities[name] = description

    def set_goal(self, goal: str):
        if goal not in self.goals:
            self.goals.append(goal)

    def format_for_prompt(self) -> str:
        parts = []
        if self.goals:
            parts.append("Current goals: " + "; ".join(self.goals))
        if self.facts:
            parts.append("Known facts:\n" + "\n".join(f"  - {f}" for f in self.facts[-10:]))
        if self.entities:
            ents = "; ".join(f"{k}: {v}" for k, v in list(self.entities.items())[-5:])
            parts.append("Key entities: " + ents)
        return "\n".join(parts)

    def extract_from_text(self, text: str):
        """Heuristically pull facts/entities from model or user text."""
        # Numbers and measurements are often key facts
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?(?:\s*(?:km|kg|mph|GB|TB|%|dollars?|years?))?\b', text)
        for n in numbers[:3]:
            self.add_fact(n)

        # "X is Y" patterns → entities
        for m in re.finditer(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+is\s+([^.]{5,60})', text):
            self.add_entity(m.group(1), m.group(2).strip(".").strip())

    def clear(self):
        self.facts.clear()
        self.entities.clear()
        self.goals.clear()


# ── Chain-of-Thought engine ───────────────────────────────────────────────────

class ChainOfThought:
    """Wraps a generator to produce step-by-step reasoning before final answer.

    Usage:
        cot = ChainOfThought(generator.generate)
        answer = cot.answer("What is 17 * 24?")
        # Returns only the final answer text
    """

    SYSTEM_ADDENDUM = (
        "Before answering, reason step-by-step inside "
        f"{THINK_OPEN}...{THINK_CLOSE} tags. "
        "Be concise in your reasoning, then give a clear final answer."
    )

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        strip_thoughts: bool = True,
        max_retries: int = 2,
    ):
        self.generate_fn = generate_fn
        self.strip_thoughts = strip_thoughts
        self.max_retries = max_retries

    def _build_cot_prompt(self, prompt: str) -> str:
        if THINK_OPEN not in prompt:
            return prompt + f"\n\n{self.SYSTEM_ADDENDUM}"
        return prompt

    def answer(self, prompt: str) -> Tuple[str, str]:
        """Generate answer with CoT.

        Returns:
            (thought_trace, final_answer)
        """
        cot_prompt = self._build_cot_prompt(prompt)
        raw = self.generate_fn(cot_prompt)

        thought, answer = _extract_block(raw, THINK_OPEN, THINK_CLOSE)
        if not answer:
            answer = raw

        return thought, answer.strip()

    def answer_clean(self, prompt: str) -> str:
        """Return only the final answer (no reasoning trace)."""
        _, answer = self.answer(prompt)
        return answer


# ── Planner ───────────────────────────────────────────────────────────────────

@dataclass
class Plan:
    goal: str
    steps: List[str] = field(default_factory=list)
    completed: List[bool] = field(default_factory=list)

    @property
    def next_step(self) -> Optional[str]:
        for step, done in zip(self.steps, self.completed):
            if not done:
                return step
        return None

    @property
    def is_complete(self) -> bool:
        return all(self.completed)

    def mark_done(self, step_idx: int):
        if 0 <= step_idx < len(self.completed):
            self.completed[step_idx] = True

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}"]
        for i, (step, done) in enumerate(zip(self.steps, self.completed)):
            mark = "✓" if done else "○"
            lines.append(f"  {mark} Step {i+1}: {step}")
        return "\n".join(lines)


class Planner:
    """Decomposes a complex goal into ordered steps using the model."""

    PLANNING_PROMPT = (
        "Break the following task into clear, numbered steps. "
        "Be specific. Output ONLY the numbered list, nothing else.\n\nTask: {goal}"
    )

    def __init__(self, generate_fn: Callable[[str], str]):
        self.generate_fn = generate_fn

    def make_plan(self, goal: str) -> Plan:
        prompt = self.PLANNING_PROMPT.format(goal=goal)
        raw = self.generate_fn(prompt)

        steps = []
        for line in raw.strip().splitlines():
            line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            if line:
                steps.append(line)

        plan = Plan(goal=goal, steps=steps, completed=[False] * len(steps))
        return plan


# ── Self-reflection / Verifier ────────────────────────────────────────────────

class Verifier:
    """Asks the model to critique and optionally improve its own answer."""

    CRITIQUE_PROMPT = (
        "You previously answered:\n\"{answer}\"\n\n"
        "Question: Is this answer correct and complete? "
        "Check for errors, missing steps, or incorrect facts. "
        f"Think inside {VERIFY_OPEN}...{VERIFY_CLOSE} tags, "
        "then output the corrected answer (or the original if correct)."
    )

    def __init__(self, generate_fn: Callable[[str], str]):
        self.generate_fn = generate_fn

    def verify(self, answer: str) -> Tuple[str, str]:
        """Critique an answer.

        Returns:
            (critique, refined_answer)
        """
        prompt = self.CRITIQUE_PROMPT.format(answer=answer)
        raw = self.generate_fn(prompt)

        critique, refined = _extract_block(raw, VERIFY_OPEN, VERIFY_CLOSE)
        if not refined:
            refined = raw
        return critique.strip(), refined.strip()


# ── ReAct loop (Reason + Act) ─────────────────────────────────────────────────

@dataclass
class ReActStep:
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None


class ReActAgent:
    """Reason + Act loop: think → use tool → observe → repeat → answer.

    Format in model output:
        Thought: I need to calculate 17 * 24
        Action: calculator
        Action Input: 17 * 24
        Observation: 408
        Thought: The answer is 408
        Final Answer: 408
    """

    MAX_STEPS = 6

    SYSTEM_PROMPT = """You are Yaya, a smart AI assistant. Solve problems step by step.

You can use these tools:
  calculator — evaluate a math expression (e.g. "17 * 24 + 5")
  search     — look up a fact (e.g. "capital of Kenya")
  python     — run Python code (e.g. "sorted([3,1,2])")

Format:
  Thought: <your reasoning>
  Action: <tool name>
  Action Input: <tool input>
  (system provides Observation)
  ... repeat as needed ...
  Final Answer: <your answer>

If you don't need a tool, skip directly to Final Answer."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tool_registry=None,
    ):
        self.generate_fn = generate_fn
        self.registry = tool_registry
        self._steps: List[ReActStep] = []

    def _parse_step(self, text: str) -> ReActStep:
        step = ReActStep(thought="")

        thought_m = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        if thought_m:
            step.thought = thought_m.group(1).strip()

        action_m = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", text, re.DOTALL)
        if action_m:
            step.action = action_m.group(1).strip()

        input_m = re.search(r"Action Input:\s*(.+?)(?=Observation:|Thought:|Final Answer:|$)", text, re.DOTALL)
        if input_m:
            step.action_input = input_m.group(1).strip()

        answer_m = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)
        if answer_m:
            step.final_answer = answer_m.group(1).strip()

        return step

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        if self.registry is None:
            return f"(tool {tool_name!r} not available)"
        try:
            from src.agent.tools import ToolCall
            result = self.registry.execute(ToolCall(name=tool_name, arguments={"input": tool_input}))
            return result.result if result.success else f"Error: {result.error}"
        except Exception as e:
            return f"Error: {e}"

    def run(self, query: str) -> Tuple[str, List[ReActStep]]:
        """Run the ReAct loop.

        Returns:
            (final_answer, step_trace)
        """
        self._steps = []
        context = f"{self.SYSTEM_PROMPT}\n\nQuestion: {query}\n"

        for _ in range(self.MAX_STEPS):
            raw = self.generate_fn(context)
            step = self._parse_step(raw)
            self._steps.append(step)

            if step.final_answer:
                return step.final_answer, self._steps

            if step.action and step.action_input:
                observation = self._execute_tool(step.action, step.action_input)
                step.observation = observation
                context += (
                    f"\nThought: {step.thought}"
                    f"\nAction: {step.action}"
                    f"\nAction Input: {step.action_input}"
                    f"\nObservation: {observation}\n"
                )
            else:
                # No action — model should provide Final Answer
                context += f"\nThought: {step.thought}\nFinal Answer:"
                raw2 = self.generate_fn(context)
                answer = raw2.strip().splitlines()[0] if raw2.strip() else "(no answer)"
                return answer, self._steps

        # Max steps reached
        last_thought = self._steps[-1].thought if self._steps else ""
        return last_thought or "(reached step limit)", self._steps


# ── Super-brain: unified cognitive interface ──────────────────────────────────

class SuperBrain:
    """Yaya's unified cognitive engine.

    Combines:
    - Chain-of-Thought for single-turn reasoning
    - ReAct for tool-using multi-step tasks
    - Planner for complex goal decomposition
    - Verifier for self-checking important answers
    - WorkingMemory for intra-conversation context

    Usage:
        brain = SuperBrain(generator.generate, tool_registry=registry)
        answer = brain.think("What is the population of Nairobi?")
        answer = brain.solve("Plan a 3-day trip to Mombasa")
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tool_registry=None,
        verify_threshold: float = 0.0,   # always verify if > 0
    ):
        self.generate_fn = generate_fn
        self.cot       = ChainOfThought(generate_fn)
        self.planner   = Planner(generate_fn)
        self.verifier  = Verifier(generate_fn)
        self.react     = ReActAgent(generate_fn, tool_registry)
        self.memory    = WorkingMemory()
        self.verify_threshold = verify_threshold

    def think(self, query: str, verify: bool = False) -> Dict[str, Any]:
        """Answer with chain-of-thought. Optionally self-verify.

        Returns:
            {answer, thought, critique, verified_answer, elapsed_ms}
        """
        t0 = time.monotonic()
        thought, answer = self.cot.answer(query)
        self.memory.extract_from_text(answer)

        critique = ""
        verified = answer
        if verify:
            critique, verified = self.verifier.verify(answer)

        return {
            "answer":          verified,
            "thought":         thought,
            "critique":        critique,
            "raw_answer":      answer,
            "elapsed_ms":      int((time.monotonic() - t0) * 1000),
        }

    def solve(self, goal: str) -> Dict[str, Any]:
        """Decompose a complex goal into steps and execute each with CoT.

        Returns:
            {plan, step_outputs, final_answer}
        """
        plan = self.planner.make_plan(goal)
        step_outputs = []

        for i, step in enumerate(plan.steps):
            context = f"Overall goal: {goal}\nCurrent step: {step}"
            if step_outputs:
                context += "\nPrevious results:\n" + "\n".join(
                    f"  Step {j+1}: {o}" for j, o in enumerate(step_outputs)
                )
            _, answer = self.cot.answer(context)
            step_outputs.append(answer)
            plan.mark_done(i)
            self.memory.add_fact(f"Step {i+1} ({step}): {answer[:120]}")

        synthesis_prompt = (
            f"Goal: {goal}\n"
            "Completed steps:\n" +
            "\n".join(f"  {i+1}. {o}" for i, o in enumerate(step_outputs)) +
            "\n\nSummarise the final answer concisely."
        )
        _, final = self.cot.answer(synthesis_prompt)

        return {
            "plan":         plan.summary(),
            "step_outputs": step_outputs,
            "final_answer": final,
        }

    def act(self, query: str) -> Dict[str, Any]:
        """Use the ReAct loop for tool-using tasks.

        Returns:
            {answer, steps}
        """
        answer, steps = self.react.run(query)
        return {
            "answer": answer,
            "steps":  [
                {
                    "thought":     s.thought,
                    "action":      s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in steps
            ],
        }

    def chat(self, query: str, use_tools: bool = False) -> str:
        """Main entry point — auto-selects reasoning mode.

        - Tool-like queries  → ReAct
        - Multi-step goals   → solve()
        - Simple questions   → think()
        """
        # Inject working memory context
        mem_ctx = self.memory.format_for_prompt()
        enriched = f"{mem_ctx}\n\n{query}" if mem_ctx else query

        # Route to best strategy
        lower = query.lower()
        plan_keywords = ["plan", "steps to", "how do i", "how can i", "design", "create"]
        tool_keywords = ["calculate", "compute", "run", "execute", "search", "look up", "what is"]

        if use_tools and any(k in lower for k in tool_keywords):
            result = self.act(enriched)
            return result["answer"]
        elif any(k in lower for k in plan_keywords) and len(query) > 40:
            result = self.solve(enriched)
            return result["final_answer"]
        else:
            result = self.think(enriched)
            return result["answer"]
