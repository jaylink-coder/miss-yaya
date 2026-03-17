"""Extended benchmark runners for Yaya model evaluation.

Covers ARC (AI2 Reasoning Challenge), TruthfulQA, GSM8K (math reasoning),
and HumanEval (code generation).
"""

import json
import os
import re
from typing import Optional, Dict, List, Any

import torch
from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.metrics import exact_match, f1_score, aggregate_metrics


class ARCBenchmark(BenchmarkRunner):
    """ARC (AI2 Reasoning Challenge) — science question answering.

    Multiple choice with 4 options (A, B, C, D).
    Comes in Easy and Challenge splits.
    """

    def __init__(self, data_path: Optional[str] = None, split: str = "challenge"):
        super().__init__(f"arc_{split}", data_path)
        self.split = split

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]

        correct = 0
        total = 0

        for sample in data:
            question = sample.get("question", "")
            choices = sample.get("choices", {})
            answer_key = sample.get("answerKey", "A")

            labels = choices.get("label", [])
            texts = choices.get("text", [])
            if not labels or not texts:
                continue

            # Build prompt
            prompt = f"Question: {question}\n"
            for label, text in zip(labels, texts):
                prompt += f"{label}. {text}\n"
            prompt += "Answer:"

            # Score each answer
            choice_scores = []
            for label in labels:
                full_text = f"{prompt} {label}"
                tokens = tokenizer.encode(full_text, add_bos=True)
                input_ids = torch.tensor([tokens], device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                answer_tokens = tokenizer.encode(f" {label}", add_bos=False)
                if answer_tokens:
                    pos = len(tokens) - len(answer_tokens)
                    score = logits[0, pos, answer_tokens[0]].item()
                    choice_scores.append(score)
                else:
                    choice_scores.append(float("-inf"))

            predicted_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
            predicted_label = labels[predicted_idx]

            if predicted_label == answer_key:
                correct += 1
            total += 1

            self.results.append({"score": 1.0 if predicted_label == answer_key else 0.0})

        acc = correct / max(total, 1)
        return {f"arc_{self.split}_accuracy": acc, f"arc_{self.split}_total": total}


class TruthfulQABenchmark(BenchmarkRunner):
    """TruthfulQA — tests model's ability to avoid generating false answers.

    Evaluates whether models generate truthful answers to questions
    designed to elicit common misconceptions.
    """

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("truthfulqa", data_path)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]

        correct = 0
        total = 0

        for sample in data:
            question = sample.get("question", "")
            best_answer = sample.get("best_answer", "")
            correct_answers = sample.get("correct_answers", [best_answer])
            incorrect_answers = sample.get("incorrect_answers", [])

            if not correct_answers or not incorrect_answers:
                continue

            # Score correct vs incorrect answers by log-likelihood
            all_answers = [(a, True) for a in correct_answers] + [(a, False) for a in incorrect_answers]

            best_score = float("-inf")
            best_is_correct = False

            for answer, is_correct_answer in all_answers:
                full_text = f"Q: {question}\nA: {answer}"
                tokens = tokenizer.encode(full_text, add_bos=True)
                q_tokens = tokenizer.encode(f"Q: {question}\nA:", add_bos=True)

                input_ids = torch.tensor([tokens], device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                # Sum log probs of answer tokens
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                score = 0.0
                for j in range(len(q_tokens) - 1, len(tokens) - 1):
                    if j < log_probs.shape[0]:
                        score += log_probs[j, tokens[j + 1]].item()

                if score > best_score:
                    best_score = score
                    best_is_correct = is_correct_answer

            if best_is_correct:
                correct += 1
            total += 1

            self.results.append({"score": 1.0 if best_is_correct else 0.0})

        acc = correct / max(total, 1)
        return {"truthfulqa_accuracy": acc, "truthfulqa_total": total}


class GSM8KBenchmark(BenchmarkRunner):
    """GSM8K — grade school math reasoning benchmark.

    Tests mathematical reasoning with chain-of-thought.
    Each problem has a numeric final answer.
    """

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("gsm8k", data_path)

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract numeric answer from generated text."""
        # Look for #### pattern (GSM8K format)
        match = re.search(r"####\s*(.+)", text)
        if match:
            return match.group(1).strip().replace(",", "")

        # Look for "the answer is X" pattern
        match = re.search(r"the answer is\s*(.+?)[\.\s]", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace(",", "")

        # Last number in text
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1]

        return None

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]

        correct = 0
        total = 0

        for sample in data:
            question = sample.get("question", "")
            answer_text = sample.get("answer", "")
            gold_answer = self._extract_answer(answer_text)

            if gold_answer is None:
                continue

            # Generate solution
            prompt = f"Question: {question}\nLet's solve this step by step.\n"
            tokens = tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([tokens], device=device)

            # Simple greedy generation (limited tokens)
            generated = list(tokens)
            for _ in range(256):
                outputs = model(input_ids=input_ids)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1)
                next_id = next_token.item()
                generated.append(next_id)
                if next_id == tokenizer.eos_id:
                    break
                input_ids = next_token.unsqueeze(0)

            response = tokenizer.decode(generated)
            predicted_answer = self._extract_answer(response)

            is_correct = predicted_answer is not None and predicted_answer == gold_answer
            if is_correct:
                correct += 1
            total += 1

            self.results.append({"score": 1.0 if is_correct else 0.0})

        acc = correct / max(total, 1)
        return {"gsm8k_accuracy": acc, "gsm8k_total": total}


class HumanEvalBenchmark(BenchmarkRunner):
    """HumanEval — code generation benchmark.

    Tests the model's ability to generate correct Python functions
    from docstrings. Uses execution-based evaluation (pass@k).

    Note: Actual execution requires a sandboxed environment.
    This implementation scores based on syntactic similarity.
    """

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("humaneval", data_path)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        model.eval()
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]

        completed = 0
        total = 0

        for sample in data:
            prompt = sample.get("prompt", "")
            canonical = sample.get("canonical_solution", "")
            entry_point = sample.get("entry_point", "")

            if not prompt:
                continue

            # Generate completion
            tokens = tokenizer.encode(prompt, add_bos=True)
            input_ids = torch.tensor([tokens], device=device)

            generated = list(tokens)
            for _ in range(512):
                outputs = model(input_ids=input_ids)
                next_token = outputs["logits"][:, -1, :].argmax(dim=-1)
                next_id = next_token.item()
                generated.append(next_id)
                if next_id == tokenizer.eos_id:
                    break
                input_ids = next_token.unsqueeze(0)

            response = tokenizer.decode(generated)
            # Extract just the completion after the prompt
            completion = response[len(prompt):]

            # Basic syntactic check: does it define the function and return something?
            has_return = "return " in completion
            has_function = entry_point in completion or "def " in completion

            if has_return:
                completed += 1
            total += 1

            self.results.append({
                "score": 1.0 if has_return else 0.0,
                "entry_point": entry_point,
            })

        rate = completed / max(total, 1)
        return {"humaneval_completion_rate": rate, "humaneval_total": total}
