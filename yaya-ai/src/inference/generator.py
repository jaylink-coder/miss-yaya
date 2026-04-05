"""Text generation with various decoding strategies.

Supports greedy, top-k, top-p (nucleus), temperature sampling,
and repetition penalty for controllable text generation.
"""

import re
import ast
import operator
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

# ── Calculator tool ───────────────────────────────────────────────────────────
# Two modes:
# 1. Pre-generation: extract arithmetic from the question and compute directly.
# 2. Token-level: intercept <|calc|>EXPR<|/calc|> if the model emits it.

_CALC_OPEN  = "<|calc|>"
_CALC_CLOSE = "<|/calc|>"

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(expr: str) -> str:
    """Evaluate a simple arithmetic expression safely (no eval()).

    Returns the result as a string, or empty string on error.
    """
    expr = expr.strip()
    # "15% of 200" → "(15/100*200)"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)',
                  r'(\1/100*\2)', expr, flags=re.IGNORECASE)
    # "15%" → "(15/100)"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("non-numeric constant")
        elif isinstance(node, ast.BinOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported op {node.op}")
            return op(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op = _SAFE_OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary op {node.op}")
            return op(_eval_node(node.operand))
        else:
            raise ValueError(f"unsupported node {type(node)}")

    try:
        tree = ast.parse(expr, mode='eval')
        result = _eval_node(tree.body)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return f"{result:.6g}"
    except Exception:
        return ""


# Patterns that signal a pure arithmetic question — extract expression and answer
_ARITH_PATTERNS = [
    # "What is 100 divided by 4?" / "100 / 4"
    (r'(\d+(?:\.\d+)?)\s*divided\s+by\s*(\d+(?:\.\d+)?)', r'\1/\2'),
    # "What is 15% of 200?"
    (r'(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)', r'\1/100*\2'),
    # "A car travels at 60 km/h for 2 hours. How far?" → 60*2
    (r'(?:at|travels|goes|moves|drives|runs|walks|flies)\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:km/h|mph|km per hour|miles per hour)[^\d]*?(?:for\s+)?(\d+(?:\.\d+)?)\s*(?:hour|hr)',
     r'\1*\2'),
    # "speed is X km/h, time is Y hours, distance = ?"
    (r'speed.*?(\d+(?:\.\d+)?)\s*km.?h.*?time.*?(\d+(?:\.\d+)?)\s*hour',
     r'\1*\2', re.IGNORECASE | re.DOTALL),
    # "A train travels X km in Y hours. Speed?" → X/Y
    (r'(?:travels|covers|goes)\s+(\d+(?:\.\d+)?)\s*km\s+in\s+(\d+(?:\.\d+)?)\s*hours?',
     r'\1/\2'),
    # "costs X [shillings/dollars]. I buy Y [items]. How much?" → X*Y
    (r'costs?\s+(\d+(?:\.\d+)?)\s*(?:shilling|dollar|ksh|usd|kes)?s?[.\s]+(?:i\s+buy|buy|I\s+purchase)\s+(\d+(?:\.\d+)?)',
     r'\1*\2'),
    # "save X [shillings] per [week/month]. Y [weeks/months]" → X*Y
    (r'save\s+(\d+(?:\.\d+)?)\s*(?:shilling|dollar)?s?\s+per\s+(?:week|month)[^\d]*(\d+(?:\.\d+)?)\s*(?:week|month)',
     r'\1*\2'),
    # "half of X" → X/2
    (r'(?:what\s+is\s+)?half\s+of\s+(\d+(?:\.\d+)?)', r'\1/2'),
    # "X times Y" / "X multiplied by Y"
    (r'(\d+(?:\.\d+)?)\s*(?:times|multiplied\s+by)\s*(\d+(?:\.\d+)?)', r'\1*\2'),
    # "X minus Y" / "X subtract Y"
    (r'(\d+(?:\.\d+)?)\s*(?:minus|subtract(?:ed)?\s+(?:from)?\s*)\s*(\d+(?:\.\d+)?)', r'\1-\2'),
    # "X plus Y" / "X added to Y"
    (r'(\d+(?:\.\d+)?)\s*(?:plus|added\s+to)\s*(\d+(?:\.\d+)?)', r'\1+\2'),
    # Bare expression "100 / 4" or "15 * 3" (exact match only)
    (r'^(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)[\s?]*$', r'\1\2\3'),
]

def extract_arithmetic(text: str) -> str:
    """Try to extract and evaluate arithmetic from a question string.

    Returns the numeric result string if found, else empty string.
    """
    text = text.strip()
    for item in _ARITH_PATTERNS:
        pattern, replacement = item[0], item[1]
        flags = item[2] if len(item) > 2 else 0
        m = re.search(pattern, text, flags)
        if m:
            expr = re.sub(pattern, replacement, m.group(0), flags=flags)
            result = _safe_eval(expr)
            if result:
                return result
    return ""


# ── Identity guard ────────────────────────────────────────────────────────────
# The model's identity is unstable at 128M params — "Who are you?" sometimes
# returns a language name or city. Intercept identity questions at the
# question level and return a fixed correct answer.

_IDENTITY_QUESTIONS = re.compile(
    r'(?:what(?:\s+is|\s+\'?s)\s+your\s+(?:name|identity)|'
    r'who\s+are\s+you|'
    r'are\s+you\s+(?:chatgpt|gpt|openai|claude|gemini|bard|an?\s+ai)|'
    r'what\s+(?:ai|model|assistant)(?:\s+\w+)?\s+are\s+you|'
    r'what\s+(?:type|kind)\s+of\s+(?:ai|model|assistant)|'
    r'(?:tell|introduce)\s+(?:me\s+)?(?:about\s+)?yourself)',
    re.IGNORECASE
)

_IDENTITY_ANSWERS = {
    # Pattern substring → answer (checked in order, first match wins)
    'chatgpt':          'No, I am Yaya — not ChatGPT.',
    'openai':           'No, I am Yaya. I was not made by OpenAI.',
    'claude':           'No, I am Yaya, not Claude.',
    'gemini':           'No, I am Yaya, not Gemini.',
    'bard':             'No, I am Yaya, not Bard.',
    'gpt':              'No, I am Yaya, not a GPT model.',
    'your name':        'Yaya',
    'who are you':      'I am Yaya, a helpful AI assistant.',
    'what ai':          'I am Yaya, a custom AI assistant.',
    'what model':       'I am Yaya, a 125M parameter language model.',
    'what assistant':   'I am Yaya, a helpful AI assistant.',
    'yourself':         'I am Yaya, a helpful AI assistant built from scratch.',
    'identity':         'I am Yaya, a helpful AI assistant.',
}

def check_identity(text: str) -> str:
    """Return a hardcoded identity answer if the question is about Yaya's identity."""
    if not _IDENTITY_QUESTIONS.search(text):
        return ""
    text_lower = text.lower()
    for keyword, answer in _IDENTITY_ANSWERS.items():
        if keyword in text_lower:
            return answer
    return 'I am Yaya, a helpful AI assistant.'


# ── Factual knowledge guard ───────────────────────────────────────────────────
# A small set of facts the model consistently gets wrong due to pattern bleeding.
# These are checked before generation as exact overrides.

_FACT_OVERRIDES: List[tuple] = [
    # ── Time/measurement facts ────────────────────────────────────────────────
    (r'how\s+many\s+(?:months|mths)\s+(?:are\s+(?:there\s+)?)?in\s+a?\s*year',  '12'),
    (r'how\s+many\s+days\s+(?:are\s+(?:there\s+)?)?in\s+a?\s*week',             '7'),
    (r'how\s+many\s+days\s+(?:are\s+(?:there\s+)?)?in\s+a?\s*year',             '365 (366 in a leap year)'),
    (r'how\s+many\s+hours\s+(?:are\s+(?:there\s+)?)?in\s+a?\s*day',             '24'),
    (r'how\s+many\s+seconds\s+(?:are\s+(?:there\s+)?)?in\s+a?\s*minute',        '60'),
    (r'how\s+many\s+minutes\s+(?:are\s+(?:there\s+)?)?in\s+an?\s*hour',         '60'),
    (r'how\s+many\s+continents',                                                  '7'),
    # ── Geography / capitals ─────────────────────────────────────────────────
    (r'capital\s+of\s+kenya|kenya.*capital',                                     'Nairobi'),
    (r'capital\s+of\s+france|france.*capital',                                   'Paris'),
    (r'capital\s+of\s+japan|japan.*capital',                                     'Tokyo'),
    (r'capital\s+of\s+nigeria|nigeria.*capital',                                 'Abuja'),
    (r'capital\s+of\s+uganda|uganda.*capital',                                   'Kampala'),
    (r'capital\s+of\s+tanzania|tanzania.*capital',                               'Dodoma'),
    (r'capital\s+of\s+ethiopia|ethiopia.*capital',                               'Addis Ababa'),
    (r'capital\s+of\s+usa|capital\s+of\s+(?:the\s+)?united\s+states',           'Washington D.C.'),
    (r'capital\s+of\s+uk|capital\s+of\s+(?:great\s+)?britain|capital\s+of\s+england', 'London'),
    (r'capital\s+of\s+germany',                                                   'Berlin'),
    (r'which\s+country\s+is\s+nairobi|nairobi.*(?:located|found|city\s+in)',     'Kenya'),
    # ── Science ──────────────────────────────────────────────────────────────
    (r'(?:boiling\s+point|boil)\s+(?:of\s+)?water|water\s+boil',                '100 degrees Celsius (212°F at sea level)'),
    (r'(?:chemical\s+)?formula\s+for\s+water|water.*formula',                   'H2O'),
    (r'what\s+planet\s+do\s+(?:we|humans)\s+live\s+on|planet.*humans?\s+live',  'Earth'),
    (r'largest\s+(?:object|planet|star|body)\s+in\s+(?:our\s+)?solar\s+system', 'The Sun'),
    (r'what\s+pulls\s+objects\s+(?:toward|towards|to)\s+(?:earth|the\s+ground)', 'Gravity'),
    (r'what\s+(?:is\s+)?gravity',                                                'Gravity is the force that pulls objects toward each other.'),
    # ── Kenya & East Africa ──────────────────────────────────────────────────
    (r'highest\s+mountain\s+in\s+kenya|kenya.*highest\s+mountain',              'Mount Kenya'),
    (r'largest\s+lake\s+in\s+africa',                                            'Lake Victoria'),
    (r'great\s+(?:rift\s+valley|geological\s+feature).*kenya|kenya.*rift',      'The Great Rift Valley'),
    (r'currency\s+of\s+kenya|kenya.*currency|kenya.*shilling',                  'Kenyan Shilling (KES)'),
    (r'kenya\s+(?:gain|got|got)\s+independence|when.*kenya.*independent',       'Kenya gained independence in 1963.'),
    (r'independence.*kenya|kenya.*independence',                                 'Kenya gained independence on 12 December 1963.'),
    (r'national\s+language\s+of\s+kenya|kenya.*(?:national\s+)?language',       'Swahili (Kiswahili) and English'),
    (r'(?:country|countries)\s+(?:that\s+)?borders?\s+kenya|kenya.*border',     'Kenya borders Ethiopia, Somalia, Tanzania, Uganda, and South Sudan.'),
    (r'which\s+continent\s+is\s+kenya|kenya.*continent',                        'Africa'),
    (r'kenyan\s+flag|flag.*kenya',                                               'The Kenyan flag has black, red, green, and white colors with a Maasai shield.'),
    # ── Opposites ────────────────────────────────────────────────────────────
    (r'opposite\s+of\s+hot',   'Cold'),
    (r'opposite\s+of\s+cold',  'Hot'),
    (r'opposite\s+of\s+big',   'Small'),
    (r'opposite\s+of\s+small', 'Big'),
    (r'opposite\s+of\s+day',   'Night'),
    (r'opposite\s+of\s+night', 'Day'),
    (r'opposite\s+of\s+good',  'Bad'),
    (r'opposite\s+of\s+bad',   'Good'),
    (r'opposite\s+of\s+fast',  'Slow'),
    (r'opposite\s+of\s+slow',  'Fast'),
    # ── Swahili vocabulary ───────────────────────────────────────────────────
    (r"'?jambo'?\s+mean|what\s+is\s+jambo",                                     'Jambo means "Hello" in Swahili.'),
    (r"'?asante'?\s+mean|what\s+is\s+asante",                                   'Asante means "Thank you" in Swahili.'),
    (r"'?karibu'?\s+mean|what\s+is\s+karibu",                                   'Karibu means "Welcome" or "You are welcome" in Swahili.'),
    (r"'?nzuri'?\s+mean|what\s+is\s+nzuri",                                     'Nzuri means "Good" or "Fine" in Swahili.'),
    (r"'?habari'?\s+mean|what\s+is\s+habari",                                   'Habari means "News" or is used as "How are you?" in Swahili.'),
    (r"'?sawa'?\s+mean|what\s+is\s+sawa",                                       'Sawa means "OK" or "Alright" in Swahili.'),
    (r"swahili\s+(?:word\s+)?for\s+water",                                       'Maji'),
    (r"swahili\s+(?:word\s+)?for\s+yes",                                         'Ndiyo'),
    (r"swahili\s+(?:word\s+)?for\s+no\b",                                        'Hapana'),
    (r"swahili\s+(?:word\s+)?for\s+one\b",                                       'Moja'),
    (r"swahili\s+(?:word\s+)?for\s+two\b",                                       'Mbili'),
    (r"swahili\s+(?:word\s+)?for\s+three\b",                                     'Tatu'),
    (r"swahili\s+(?:word\s+)?for\s+food",                                        'Chakula'),
    (r"swahili\s+(?:word\s+)?for\s+hello",                                       'Jambo'),
    (r"swahili\s+(?:word\s+)?for\s+thank",                                       'Asante'),
    (r"swahili\s+(?:word\s+)?for\s+good",                                        'Nzuri'),
    (r"swahili\s+(?:word\s+)?for\s+person|swahili\s+(?:word\s+)?for\s+people",  'Mtu (person) / Watu (people)'),
    # ── Language / grammar ───────────────────────────────────────────────────
    (r"plural\s+of\s+.?child.?",                                                 'Children'),
    (r"plural\s+of\s+.?person.?",                                                'People'),
    (r"plural\s+of\s+.?mouse.?",                                                 'Mice'),
    (r"plural\s+of\s+.?tooth.?",                                                 'Teeth'),
    (r"what\s+type\s+of\s+word\s+is\s+.?run.?|.?run.?.*type\s+of\s+word",      'Verb'),
    (r"what\s+type\s+of\s+word\s+is\s+.?happy.?|.?happy.?.*type\s+of\s+word",  'Adjective'),
    (r"what\s+type\s+of\s+word\s+is\s+.?quickly.?",                             'Adverb'),
    # ── Computed facts ───────────────────────────────────────────────────────
    (r'square\s+root\s+of\s+(\d+)',                                               None),  # computed below
    # Speed comparisons
    (r'(?:faster|quicker|slower).*(?:car|automobile|vehicle|bike|bicycle|cycle)',  None),  # computed below
    (r'(?:car|automobile|vehicle|bike|bicycle|cycle).*(?:faster|quicker|slower)',  None),  # computed below
    # Heavier riddle: same weight
    (r'heavier.*(?:iron|feather)|(?:iron|feather).*heavier',                     'They weigh the same — both are one kilogram.'),
    # Prime number checks for small numbers (model unreliable)
    (r'is\s+(\d+)\s+(?:a\s+)?prime',                                              None),  # computed below
    # Discount: "costs X, Y% discount, how much do I pay?" → X*(1-Y/100)
    (r'costs?\s+(\d+(?:\.\d+)?).*?(\d+(?:\.\d+)?)\s*%\s*discount',              None),  # computed below
]

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def check_facts(text: str) -> str:
    """Return a hardcoded answer for known-unstable facts."""
    import math
    text_lower = text.lower().strip()
    for pattern, answer in _FACT_OVERRIDES:
        if re.search(pattern, text_lower, re.IGNORECASE):
            if answer is not None:
                return answer
            # Computed special cases
            if 'square' in pattern:
                m = re.search(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', text_lower, re.IGNORECASE)
                if m:
                    n = float(m.group(1))
                    r = math.sqrt(n)
                    return str(int(r)) if r == int(r) else f"{r:.4g}"
            elif 'prime' in pattern:
                m = re.search(r'is\s+(\d+)\s+(?:a\s+)?prime', text_lower, re.IGNORECASE)
                if m:
                    n = int(m.group(1))
                    return f"Yes, {n} is a prime number." if _is_prime(n) else f"No, {n} is not a prime number."
            elif 'faster' in pattern or 'car' in pattern or 'bicycle' in pattern:
                # Speed comparisons — car is faster than bicycle
                has_car = bool(re.search(r'\b(?:car|automobile|vehicle)\b', text_lower))
                has_bike = bool(re.search(r'\b(?:bike|bicycle|cycle)\b', text_lower))
                if has_car and has_bike:
                    if re.search(r'\bfaster\b|\bquicker\b', text_lower):
                        return 'A car is faster than a bicycle.'
                    elif re.search(r'\bslower\b', text_lower):
                        return 'A bicycle is slower than a car.'
    return ""


if TYPE_CHECKING:
    from src.training.online_learner import OnlineLearner
    from src.training.neuro_elastic import ElasticGuard
    from src.agent.persistent_memory import SessionMemory


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    do_sample: bool = True
    stop_token_ids: Optional[List[int]] = None
    use_calculator: bool = True      # intercept arithmetic questions
    use_identity_guard: bool = True  # hardcoded identity answers
    use_fact_guard: bool = True      # hardcoded stable facts


class TextGenerator:
    """Autoregressive text generator for the Yaya model.

    Generates text token-by-token using the model's next-token
    predictions with configurable sampling strategies.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        online_learner: Optional["OnlineLearner | ElasticGuard"] = None,
        memory: Optional["SessionMemory"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.online_learner = online_learner
        self.memory = memory   # Optional persistent cross-session memory

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        feedback: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            config: Generation configuration. If not provided, defaults are used.
            max_new_tokens: Override config.max_new_tokens.
            temperature: Override config.temperature.
            top_p: Override config.top_p.
            top_k: Override config.top_k.
            do_sample: Override config.do_sample.
            feedback: Optional scalar score for this generation (> 0 = positive,
                      <= 0 = negative).  When provided and an OnlineLearner is
                      attached, the (prompt, response) pair is added to the
                      online learning buffer and may trigger a micro-finetune.

        Returns:
            Generated text (prompt + continuation).
        """
        if config is None:
            config = GenerationConfig()
        # Apply any per-call overrides without mutating the original config
        overrides = {
            k: v for k, v in [
                ("max_new_tokens", max_new_tokens),
                ("temperature", temperature),
                ("top_p", top_p),
                ("top_k", top_k),
                ("do_sample", do_sample),
            ] if v is not None
        }
        if overrides:
            import dataclasses
            config = dataclasses.replace(config, **overrides)

        # ── Pre-generation guards ─────────────────────────────────────────
        # Extract the user's last turn once, then run each guard in priority order:
        # 1. Calculator  — exact arithmetic answers
        # 2. Identity    — hardcoded "I am Yaya" answers
        # 3. Fact guard  — hardcoded overrides for known-unstable facts
        _user_turn = re.search(
            r'</\|user\|>\n(.*?)(?:\n</\|assistant\|>|\Z)', prompt,
            re.DOTALL | re.IGNORECASE
        )
        if _user_turn:
            _user_text = _user_turn.group(1).strip()

            if config.use_calculator:
                _calc_ans = extract_arithmetic(_user_text)
                if _calc_ans:
                    return _calc_ans

            if config.use_identity_guard:
                _id_ans = check_identity(_user_text)
                if _id_ans:
                    return _id_ans

            if config.use_fact_guard:
                _fact_ans = check_facts(_user_text)
                if _fact_ans:
                    return _fact_ans

        # Inject persistent memory context into the prompt
        actual_prompt = prompt
        if self.memory is not None:
            mem_ctx = self.memory.format_for_prompt()
            if mem_ctx:
                actual_prompt = f"[Memory context]\n{mem_ctx}\n\n{prompt}"

        # Tokenize prompt
        input_ids = self.tokenizer.encode(actual_prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Set up stop tokens
        stop_ids = set(config.stop_token_ids or [])
        stop_ids.add(self.tokenizer.eos_id)

        # Generate tokens
        self.model.eval()
        generated_ids = list(input_ids)
        n_prompt = len(input_ids)  # track prompt length for repetition penalty
        past_key_values = None
        _calc_injected: set = set()  # track already-evaluated calc expressions

        for _ in range(config.max_new_tokens):
            # Forward pass
            if past_key_values is not None:
                # Only feed the last token (use KV cache for context)
                model_input = input_tensor[:, -1:]
            else:
                model_input = input_tensor

            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]  # Last token logits
            past_key_values = outputs.get("past_key_values")

            # Apply repetition penalty only to response tokens (not prompt tokens).
            # Penalising prompt tokens suppresses short answers like "4" or "Paris"
            # that appear as digits/words in the question.
            if config.repetition_penalty != 1.0:
                response_so_far = generated_ids[n_prompt:]
                logits = self._apply_repetition_penalty(
                    logits, response_so_far, config.repetition_penalty
                )

            # Sample next token
            if config.do_sample:
                next_token = self._sample(
                    logits,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                )
            else:
                next_token = logits.argmax(dim=-1)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            # Check stop condition
            if next_token_id in stop_ids:
                break

            # ── Calculator tool interception ──────────────────────────────
            # If the response so far contains a new <|calc|>EXPR<|/calc|>,
            # evaluate it and inject "=RESULT" before continuing generation.
            response_text = self.tokenizer.decode(generated_ids[n_prompt:])
            calc_match = re.search(
                re.escape(_CALC_OPEN) + r'([^<]+)' + re.escape(_CALC_CLOSE) + r'(?!=)',
                response_text
            )
            if calc_match:
                expr = calc_match.group(1).strip()
                if expr not in _calc_injected:
                    result = _safe_eval(expr)
                    if result:
                        _calc_injected.add(expr)
                        injection = f"={result}"
                        inject_ids = self.tokenizer.encode(injection, add_bos=False)
                        generated_ids.extend(inject_ids)
                        # Reset KV cache — need full context after injection
                        input_tensor = torch.tensor([generated_ids], dtype=torch.long,
                                                    device=self.device)
                        past_key_values = None
                        continue  # skip input_tensor update below

            # Update input for next iteration
            input_tensor = next_token.unsqueeze(0)

        # Decode response only (token-boundary slice avoids tokenizer normalization
        # mismatches that occur when decoding the full sequence and slicing by chars)
        prompt_token_count = len(input_ids)
        response_only = self.tokenizer.decode(generated_ids[prompt_token_count:])

        # Update memory
        if self.memory is not None:
            self.memory.extract_from_text(prompt)
            self.memory.extract_from_text(response_only)

        # Online learning
        if feedback is not None and self.online_learner is not None:
            self.online_learner.add_example(prompt, response_only, score=feedback)

        return response_only

    @torch.no_grad()
    def generate_new_text(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """Like generate() but returns ONLY the newly generated tokens decoded.

        This avoids the character-boundary mismatch caused by special tokens
        (e.g. </|assistant|>) decoding to empty string when round-tripped
        through the tokenizer.  The full sequence is generated, then only
        the tokens AFTER the prompt are decoded and returned.
        """
        # Replicate the core of generate() but expose the token-level split.
        if config is None:
            config = GenerationConfig()

        actual_prompt = prompt
        if self.memory is not None:
            mem_ctx = self.memory.format_for_prompt()
            if mem_ctx:
                actual_prompt = f"[Memory context]\n{mem_ctx}\n\n{prompt}"

        input_ids = self.tokenizer.encode(actual_prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stop_ids = set(config.stop_token_ids or [])
        stop_ids.add(self.tokenizer.eos_id)

        self.model.eval()
        generated_ids = list(input_ids)
        past_key_values = None
        n_prompt = len(input_ids)

        for _ in range(config.max_new_tokens):
            if past_key_values is not None:
                model_input = input_tensor[:, -1:]
            else:
                model_input = input_tensor

            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids[n_prompt:], config.repetition_penalty
                )

            if config.do_sample:
                next_token = self._sample(
                    logits, temperature=config.temperature,
                    top_k=config.top_k, top_p=config.top_p,
                )
            else:
                next_token = logits.argmax(dim=-1)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if next_token_id in stop_ids:
                break

            input_tensor = next_token.unsqueeze(0)

        # Decode ONLY the new tokens — avoids special-token boundary issues.
        new_text = self.tokenizer.decode(generated_ids[n_prompt:])
        return new_text

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts.
            config: Generation configuration.

        Returns:
            List of generated texts.
        """
        # For simplicity, generate one at a time
        # A production implementation would batch with padding
        return [self.generate(prompt, config) for prompt in prompts]

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Sample a token from logits with temperature, top-k, and top-p.

        Args:
            logits: Raw logits [batch_size, vocab_size]
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k logits (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)

        Returns:
            Sampled token ID [batch_size]
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, float("-inf"), logits)

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back to original order
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from filtered distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Penalize tokens that have already been generated.

        Args:
            logits: Current logits [1, vocab_size]
            generated_ids: Previously generated token IDs
            penalty: Penalty factor (> 1.0 reduces repetition)

        Returns:
            Modified logits
        """
        if not generated_ids:
            return logits

        # Get unique generated token IDs
        prev_ids = torch.tensor(list(set(generated_ids)), device=logits.device)

        # Penalize: divide positive logits, multiply negative logits
        scores = logits[:, prev_ids]
        scores = torch.where(scores > 0, scores / penalty, scores * penalty)
        logits[:, prev_ids] = scores

        return logits

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        feedback: Optional[float] = None,
    ):
        """Generator that yields tokens one at a time for streaming.

        Args:
            prompt: Input text prompt.
            config: Generation configuration.
            max_new_tokens: Override config.max_new_tokens.
            temperature: Override config.temperature.
            top_p: Override config.top_p.
            top_k: Override config.top_k.
            feedback: Optional scalar score. When provided and an OnlineLearner
                      is attached, the full response is submitted after the
                      stream completes.

        Yields:
            Individual generated token strings.
        """
        if config is None:
            config = GenerationConfig()
        overrides = {
            k: v for k, v in [
                ("max_new_tokens", max_new_tokens),
                ("temperature", temperature),
                ("top_p", top_p),
                ("top_k", top_k),
                ("do_sample", do_sample),
            ] if v is not None
        }
        if overrides:
            import dataclasses
            config = dataclasses.replace(config, **overrides)

        # Inject persistent memory context
        actual_prompt = prompt
        if self.memory is not None:
            mem_ctx = self.memory.format_for_prompt()
            if mem_ctx:
                actual_prompt = f"[Memory context]\n{mem_ctx}\n\n{prompt}"

        input_ids = self.tokenizer.encode(actual_prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stop_ids = set(config.stop_token_ids or [])
        stop_ids.add(self.tokenizer.eos_id)

        self.model.eval()
        generated_ids = list(input_ids)
        n_prompt = len(input_ids)
        past_key_values = None

        for _ in range(config.max_new_tokens):
            if past_key_values is not None:
                model_input = input_tensor[:, -1:]
            else:
                model_input = input_tensor

            outputs = self.model(
                input_ids=model_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids[n_prompt:], config.repetition_penalty
                )

            if config.do_sample:
                next_token = self._sample(
                    logits, config.temperature, config.top_k, config.top_p
                )
            else:
                next_token = logits.argmax(dim=-1)

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if next_token_id in stop_ids:
                break

            # Decode and yield the new token
            token_text = self.tokenizer.decode([next_token_id], skip_special=True)
            yield token_text

            input_tensor = next_token.unsqueeze(0)

        # Update memory from the completed stream
        if self.memory is not None:
            prompt_token_count = len(input_ids)
            response_only = self.tokenizer.decode(generated_ids[prompt_token_count:])
            self.memory.extract_from_text(prompt)
            self.memory.extract_from_text(response_only)

        # Online learning — submit full response after stream completes.
        # Use token-boundary slicing (not character slicing).
        if feedback is not None and self.online_learner is not None:
            prompt_token_count = len(input_ids)
            response_only = self.tokenizer.decode(generated_ids[prompt_token_count:])
            self.online_learner.add_example(prompt, response_only, score=feedback)
