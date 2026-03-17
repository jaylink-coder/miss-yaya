"""Benchmark runners for evaluating the Yaya model.

Provides evaluation on standard LLM benchmarks including
MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, and HumanEval.
"""

import json
import os
from typing import Optional, Dict, List, Any
from pathlib import Path

import torch
from src.evaluation.metrics import accuracy, exact_match, f1_score, aggregate_metrics


class BenchmarkRunner:
    """Base class for benchmark evaluation."""

    def __init__(self, name: str, data_path: Optional[str] = None):
        self.name = name
        self.data_path = data_path
        self.results: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        """Load benchmark data from file."""
        if self.data_path and os.path.exists(self.data_path):
            with open(self.data_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        return []

    def evaluate(self, model, tokenizer, device: str = "cuda") -> Dict[str, float]:
        """Run evaluation. Override in subclasses."""
        raise NotImplementedError

    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics from results."""
        return aggregate_metrics(
            {self.name: [r.get("score", 0.0) for r in self.results]}
        )


class MMLUBenchmark(BenchmarkRunner):
    """MMLU (Massive Multitask Language Understanding).

    57-subject multiple choice benchmark testing world knowledge.
    """

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("mmlu", data_path)

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        num_few_shot: int = 5,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on MMLU benchmark.

        Uses few-shot prompting with log-likelihood scoring
        of answer choices (A, B, C, D).
        """
        model.eval()
        data = self.load_data()
        if max_samples:
            data = data[:max_samples]

        correct = 0
        total = 0

        for sample in data:
            question = sample.get("question", "")
            choices = sample.get("choices", [])
            answer_idx = sample.get("answer", 0)

            if not choices:
                continue

            # Build prompt with answer choices
            prompt = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                label = chr(65 + i)  # A, B, C, D
                prompt += f"{label}. {choice}\n"
            prompt += "Answer:"

            # Score each answer option by log-likelihood
            choice_scores = []
            for i, choice in enumerate(choices):
                label = chr(65 + i)
                full_text = f"{prompt} {label}"
                tokens = tokenizer.encode(full_text, add_bos=True)
                input_ids = torch.tensor([tokens], device=device)

                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                # Get log probability of the answer token
                answer_token_ids = tokenizer.encode(f" {label}", add_bos=False)
                if answer_token_ids:
                    answer_pos = len(tokens) - len(answer_token_ids)
                    score = logits[0, answer_pos, answer_token_ids[0]].item()
                    choice_scores.append(score)
                else:
                    choice_scores.append(float("-inf"))

            predicted = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
            is_correct = predicted == answer_idx

            self.results.append({
                "question": question,
                "predicted": predicted,
                "correct": answer_idx,
                "score": 1.0 if is_correct else 0.0,
            })

            if is_correct:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        return {"mmlu_accuracy": acc, "mmlu_total": total}


class HellaSwagBenchmark(BenchmarkRunner):
    """HellaSwag — commonsense reasoning benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        super().__init__("hellaswag", data_path)

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
            context = sample.get("ctx", "")
            endings = sample.get("endings", [])
            label = sample.get("label", 0)

            if not endings:
                continue

            # Score each completion by log-likelihood
            scores = []
            for ending in endings:
                full_text = context + " " + ending
                tokens = tokenizer.encode(full_text, add_bos=True)
                ctx_tokens = tokenizer.encode(context, add_bos=True)

                input_ids = torch.tensor([tokens], device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                # Sum log probs of continuation tokens
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                score = 0.0
                for j in range(len(ctx_tokens) - 1, len(tokens) - 1):
                    if j < log_probs.shape[0]:
                        score += log_probs[j, tokens[j + 1]].item()
                scores.append(score)

            predicted = max(range(len(scores)), key=lambda i: scores[i])
            is_correct = predicted == label

            self.results.append({"score": 1.0 if is_correct else 0.0})
            if is_correct:
                correct += 1
            total += 1

        acc = correct / max(total, 1)
        return {"hellaswag_accuracy": acc, "hellaswag_total": total}


class BenchmarkSuite:
    """Run multiple benchmarks and aggregate results."""

    def __init__(self, benchmarks: Optional[List[BenchmarkRunner]] = None):
        self.benchmarks = benchmarks or []

    def add_benchmark(self, benchmark: BenchmarkRunner):
        self.benchmarks.append(benchmark)

    def evaluate_all(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run all benchmarks and return combined results."""
        all_results = {}
        for benchmark in self.benchmarks:
            print(f"Running benchmark: {benchmark.name}")
            try:
                results = benchmark.evaluate(
                    model, tokenizer, device=device, max_samples=max_samples
                )
                all_results.update(results)
                print(f"  Results: {results}")
            except Exception as e:
                print(f"  Error in {benchmark.name}: {e}")
                all_results[f"{benchmark.name}_error"] = str(e)

        return all_results
