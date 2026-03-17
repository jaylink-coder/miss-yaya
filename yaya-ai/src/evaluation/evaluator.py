"""Main evaluation orchestrator.

Coordinates running benchmarks, computing metrics, and reporting results.
"""

import torch
from typing import Optional, Dict, List, Any

from src.evaluation.benchmarks import BenchmarkSuite, MMLUBenchmark, HellaSwagBenchmark
from src.evaluation.benchmarks_extended import (
    ARCBenchmark,
    TruthfulQABenchmark,
    GSM8KBenchmark,
    HumanEvalBenchmark,
)
from src.evaluation.metrics import perplexity


class Evaluator:
    """Orchestrates model evaluation across benchmarks and held-out data."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        benchmark_data_dir: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.benchmark_data_dir = benchmark_data_dir

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        dataloader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute perplexity on a held-out dataset.

        Args:
            dataloader: DataLoader yielding batches with input_ids and labels.
            max_batches: Maximum number of batches to evaluate.

        Returns:
            Dict with 'eval_loss' and 'eval_perplexity'.
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in dataloader:
            if max_batches and num_batches >= max_batches:
                break

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch.get("attention_mask"),
            )

            loss = outputs["loss"]
            num_tokens = (batch["labels"] != -100).sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = perplexity(avg_loss)

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": ppl,
            "eval_tokens": total_tokens,
        }

    def evaluate_benchmarks(
        self,
        benchmark_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run standard LLM benchmarks.

        Args:
            benchmark_names: List of benchmark names to run.
                            Defaults to all available benchmarks.
            max_samples: Max samples per benchmark (for quick eval).

        Returns:
            Dict of benchmark results.
        """
        suite = BenchmarkSuite()

        available = {
            "mmlu": MMLUBenchmark,
            "hellaswag": HellaSwagBenchmark,
            "arc_challenge": lambda dp: ARCBenchmark(dp, split="challenge"),
            "arc_easy": lambda dp: ARCBenchmark(dp, split="easy"),
            "truthfulqa": TruthfulQABenchmark,
            "gsm8k": GSM8KBenchmark,
            "humaneval": HumanEvalBenchmark,
        }

        names = benchmark_names or list(available.keys())

        for name in names:
            if name in available:
                data_path = None
                if self.benchmark_data_dir:
                    import os
                    data_path = os.path.join(self.benchmark_data_dir, f"{name}.jsonl")
                suite.add_benchmark(available[name](data_path=data_path))

        return suite.evaluate_all(
            self.model,
            self.tokenizer,
            device=self.device,
            max_samples=max_samples,
        )

    def full_evaluation(
        self,
        eval_dataloader=None,
        benchmark_names: Optional[List[str]] = None,
        max_eval_batches: int = 100,
        max_benchmark_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run full evaluation suite.

        Args:
            eval_dataloader: DataLoader for perplexity evaluation.
            benchmark_names: Benchmarks to run.
            max_eval_batches: Max batches for perplexity eval.
            max_benchmark_samples: Max samples per benchmark.

        Returns:
            Combined results dict.
        """
        results = {}

        # Perplexity evaluation
        if eval_dataloader is not None:
            print("Evaluating perplexity...")
            ppl_results = self.evaluate_perplexity(eval_dataloader, max_eval_batches)
            results.update(ppl_results)
            print(f"  Loss: {ppl_results['eval_loss']:.4f}")
            print(f"  Perplexity: {ppl_results['eval_perplexity']:.2f}")

        # Benchmark evaluation
        if benchmark_names:
            print("Running benchmarks...")
            bench_results = self.evaluate_benchmarks(benchmark_names, max_benchmark_samples)
            results.update(bench_results)

        return results
