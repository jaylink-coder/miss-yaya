"""Text generation with various decoding strategies.

Supports greedy, top-k, top-p (nucleus), temperature sampling,
and repetition penalty for controllable text generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from src.training.online_learner import OnlineLearner
    from src.training.neuro_elastic import ElasticGuard


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_token_ids: Optional[List[int]] = None


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
        online_learner: Optional["OnlineLearner"] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.online_learner = online_learner

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

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Set up stop tokens
        stop_ids = set(config.stop_token_ids or [])
        stop_ids.add(self.tokenizer.eos_id)

        # Generate tokens
        self.model.eval()
        generated_ids = list(input_ids)
        past_key_values = None

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

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, config.repetition_penalty
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

            # Update input for next iteration
            input_tensor = next_token.unsqueeze(0)

        # Decode generated tokens
        output_text = self.tokenizer.decode(generated_ids)

        # Online learning — submit feedback if provided.
        # Use token-boundary slicing (not character slicing) to avoid tokenizer
        # normalization mismatches between prompt and decoded output.
        if feedback is not None and self.online_learner is not None:
            prompt_token_count = len(input_ids)
            response_only = self.tokenizer.decode(generated_ids[prompt_token_count:])
            self.online_learner.add_example(prompt, response_only, score=feedback)

        return output_text

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

        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stop_ids = set(config.stop_token_ids or [])
        stop_ids.add(self.tokenizer.eos_id)

        self.model.eval()
        generated_ids = list(input_ids)
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
                    logits, generated_ids, config.repetition_penalty
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

        # Online learning — submit full response after stream completes.
        # Use token-boundary slicing (not character slicing).
        if feedback is not None and self.online_learner is not None:
            prompt_token_count = len(input_ids)
            response_only = self.tokenizer.decode(generated_ids[prompt_token_count:])
            self.online_learner.add_example(prompt, response_only, score=feedback)
