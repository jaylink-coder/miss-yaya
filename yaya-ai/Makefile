.PHONY: install install-dev test lint format clean train serve sft dpo export agent rag safety

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[train,eval,serve,dev]"

install-flash:
	pip install flash-attn --no-build-isolation

# Testing
test:
	pytest tests/ -v --tb=short

test-model:
	pytest tests/test_model.py tests/test_attention.py -v

test-data:
	pytest tests/test_data.py -v

test-inference:
	pytest tests/test_inference.py -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Code quality
lint:
	ruff check src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

# Training
train-tokenizer:
	python scripts/train_tokenizer.py \
		--input_dir data/raw/text \
		--vocab_size 64000 \
		--output_dir data/tokenizer

prepare-data:
	python scripts/prepare_data.py \
		--input_dir data/raw/text \
		--output_dir data/processed/train \
		--tokenizer_path data/tokenizer/yaya_tokenizer.model

train-1.5b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/train_1_5b.yaml

train-7b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_7b.yaml \
		--train_config configs/training/train_7b.yaml

# Evaluation
evaluate:
	python scripts/evaluate.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--benchmarks mmlu,hellaswag

# Serving
serve:
	python scripts/serve.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--port 8000

# Interactive generation
generate:
	python scripts/generate.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--interactive

# Data Engine
build-dataset:
	python scripts/build_dataset.py --config configs/data/sources.yaml

build-dataset-test:
	python scripts/build_dataset.py --config configs/data/sources.yaml --test --max_samples 1000

dataset-stats:
	python -c "from src.data.shard_mixer import DatasetStatsReporter; DatasetStatsReporter('data/processed/shards').print_report()"

# SFT (Supervised Fine-Tuning)
sft-1.5b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/sft_1_5b.yaml \
		--resume checkpoints/yaya-1.5b/latest

# DPO (Alignment)
dpo-1.5b:
	python scripts/train_dpo.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/dpo_1_5b.yaml \
		--checkpoint checkpoints/yaya-1.5b-sft/latest \
		--preference_data data/preferences/train.jsonl

# Export
export-safetensors:
	python scripts/export_model.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--format safetensors --output exports/yaya-1.5b-safetensors

export-onnx:
	python scripts/export_model.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--format onnx --output exports/yaya-1.5b.onnx

export-gguf:
	python scripts/export_model.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--format gguf --output exports/yaya-1.5b-gguf

# Full evaluation suite
evaluate-full:
	python scripts/evaluate.py \
		--checkpoint checkpoints/latest \
		--model_config configs/model/yaya_1_5b.yaml \
		--benchmarks mmlu,hellaswag,arc_challenge,truthfulqa,gsm8k

# Testing (extended)
test-vision:
	pytest tests/test_vision.py -v

test-training:
	pytest tests/test_training.py -v

test-all:
	pytest tests/ -v --tb=short

# Setup
setup:
	python scripts/setup_configs.py

# Docker
docker-build:
	docker build -t yaya-ai .

docker-train:
	docker compose up train

docker-serve:
	docker compose up serve

# Agent Framework
test-agent:
	pytest tests/test_agent.py tests/test_structured_output.py -v

generate-tool-data:
	python -c "from src.agent.data_generator import ToolUseDataGenerator; g = ToolUseDataGenerator(); ex = g.generate_all(); g.save_jsonl(ex, 'data/sft/tool_use.jsonl')"

# RAG
test-rag:
	pytest tests/test_rag.py -v

# Safety
test-safety:
	pytest tests/test_safety.py -v

# Serving
test-serving:
	pytest tests/test_serving.py -v

serve-production:
	python -c "from src.serving.production_server import ProductionServer; from src.serving.middleware import ServerConfig; s = ProductionServer(ServerConfig(port=8000)); print(s.health_check())"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
