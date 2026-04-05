.PHONY: install install-dev install-flash install-hf \
        test test-model test-data test-inference test-cov test-vision test-training test-all \
        test-agent test-rag test-safety test-serving test-neuroplasticity test-continual \
        test-curriculum test-reward-model test-persistent-memory \
        sft-tiny sft-tiny-focused sft-tiny-merged sft-tiny-filtered dpo-tiny eval-tiny eval-tiny-all eval-loop eval-loop-retrain chat-tiny web-ui continuous-learn self-eval self-improve training-status \
        lint format typecheck \
        train-tokenizer prepare-data \
        train-1b train-1b-2gpu train-1.5b train-7b \
        sft-1b sft-1b-2gpu sft-1.5b \
        lora-sft-1.5b lora-continual-1.5b merge-lora-1.5b \
        gen-math-data \
        math-stage1 math-stage2 math-stage3 math-stage4 math-stage5 math-stage6 math-stage7 math-stage8 \
        math-combined math-eval math-curriculum \
        train-moe-1.5b sft-moe-1.5b \
        align-dpo-1b align-dpo dpo-1.5b \
        train-reward-model \
        evaluate evaluate-full \
        serve serve-production generate \
        build-dataset build-dataset-test dataset-stats \
        export-safetensors export-onnx export-gguf \
        setup docker-build docker-train docker-serve \
        generate-tool-data \
        download-openhermes download-pretrain-data download-pretrain-data-quick \
        mix-pretrain-data mix-pretrain-data-3b mix-pretrain-data-dry pretrain-pipeline \
        gen-data gen-data-large \
        finetune finetune-full finetune-qlora \
        eval-quick eval-full-hf eval-interactive \
        deploy deploy-quantized \
        pipeline-full clean \
        validate-sft-data validate-sft-data-verbose \
        sft dpo export agent rag safety \
        maml-1.5b \
        reasoning reasoning-resume eval-reasoning chat-reasoning \
        train-125m

# MAML meta-learning
maml-1.5b:
	python scripts/train_sft.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/maml_1_5b.yaml \
		--pretrain_checkpoint checkpoints/yaya-1.5b/latest

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

test-neuroplasticity:
	pytest tests/test_neuroplasticity.py -v --tb=short

test-continual:
	pytest tests/test_neuroplasticity.py -v --tb=short -k "ElasticGuard or DynamicEWC or ExpertUtil or SyntheticReplay or ForgettingTracker or MAML"

test-curriculum:
	pytest tests/test_neuroplasticity.py -v --tb=short -k "Curriculum or DifficultyScorer or BuildCurriculum"

test-reward-model:
	pytest tests/test_neuroplasticity.py -v --tb=short -k "RewardModel or BradleyTerry or RewardModelTrainer"

test-persistent-memory:
	pytest tests/test_neuroplasticity.py -v --tb=short -k "PersistentMemory or SessionMemory"

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

train-1b:
	python scripts/train.py \
		--model_config configs/model/yaya_1b.yaml \
		--train_config configs/training/train_1b.yaml

train-1b-2gpu:
	torchrun --nproc_per_node=2 scripts/train.py \
		--model_config configs/model/yaya_1b.yaml \
		--train_config configs/training/train_1b.yaml

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
## Serve Yaya REST API (auto-detects best checkpoint)
serve:
	python scripts/serve.py --port 8000

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

# Validate SFT data before training
validate-sft-data:
	python scripts/validate_sft_data.py

validate-sft-data-verbose:
	python scripts/validate_sft_data.py --verbose

# SFT (Supervised Fine-Tuning)
sft-1b:
	python scripts/train_sft.py \
		--model_config configs/model/yaya_1b.yaml \
		--train_config configs/training/sft_1b.yaml \
		--pretrain_checkpoint checkpoints/yaya-1b/latest

sft-1b-2gpu:
	torchrun --nproc_per_node=2 scripts/train_sft.py \
		--model_config configs/model/yaya_1b.yaml \
		--train_config configs/training/sft_1b.yaml \
		--pretrain_checkpoint checkpoints/yaya-1b/latest

sft-1.5b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/sft_1_5b.yaml \
		--resume checkpoints/yaya-1.5b/latest

# LoRA fine-tuning (parameter-efficient, trains ~0.5% of params)
lora-sft-1.5b:
	python scripts/train_sft.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/lora_sft_1_5b.yaml \
		--pretrain_checkpoint checkpoints/yaya-1.5b/latest

# LoRA + EWC continual learning (fine-tune new domain without forgetting)
lora-continual-1.5b:
	python scripts/train_sft.py \
		--model_config configs/model/yaya_1_5b.yaml \
		--train_config configs/training/lora_sft_continual_1_5b.yaml \
		--pretrain_checkpoint checkpoints/yaya-1.5b-lora-sft/latest \
		--compute_ewc_fisher

# Merge LoRA adapters into base weights and export a standard checkpoint
merge-lora-1.5b:
	python -c "\
import torch; \
from src.utils.config import load_model_config; \
from src.model.yaya_model import YayaForCausalLM; \
from src.model.lora import inject_lora, merge_lora, LoRAConfig; \
cfg = load_model_config('configs/model/yaya_1_5b.yaml'); \
model = YayaForCausalLM(cfg); \
inject_lora(model, LoRAConfig(rank=16)); \
adapters = torch.load('checkpoints/yaya-1.5b-lora-sft/latest/lora_adapters.pt', weights_only=True); \
model.load_state_dict(adapters, strict=False); \
merge_lora(model); \
torch.save(model.state_dict(), 'checkpoints/yaya-1.5b-lora-merged/model.pt'); \
print('Merged checkpoint saved.')"

# MoE pre-training
train-moe-1.5b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_1_5b_moe.yaml \
		--train_config configs/training/train_1_5b.yaml \
		--resume checkpoints/yaya-1.5b-moe/latest

# MoE SFT
sft-moe-1.5b:
	torchrun --nproc_per_node=8 scripts/train.py \
		--model_config configs/model/yaya_1_5b_moe.yaml \
		--train_config configs/training/sft_1_5b.yaml \
		--resume checkpoints/yaya-1.5b-moe/latest

# Reward model training (train from preference pairs, then use for RLHF scoring)
train-reward-model:
	python -c "\
from src.utils.config import load_model_config; \
from src.model.yaya_model import YayaForCausalLM; \
from src.training.reward_model import RewardModel, RewardModelConfig, RewardModelTrainer; \
import torch; \
cfg = load_model_config('configs/model/yaya_1_5b.yaml'); \
model = YayaForCausalLM(cfg); \
rm_cfg = RewardModelConfig(hidden_size=cfg.hidden_size); \
rm = RewardModel(model, config=rm_cfg, freeze_backbone=True); \
print('RewardModel ready — head params:', sum(p.numel() for p in rm.head.parameters())); \
print('Backbone frozen params:', sum(p.numel() for p in rm.backbone.parameters() if not p.requires_grad))"

# DPO (Alignment)
align-dpo-1b:
	python scripts/align_dpo_hf.py \
		--model_path outputs/yaya-1b-sft-merged \
		--data_dir data/sft \
		--output_dir outputs/yaya-1b-dpo

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

# ── HuggingFace Fine-Tuning Pipeline ────────────────────────────────────────
install-hf:
	pip install -r requirements-hf.txt

# Download OpenHermes 2.5 SFT data (100K examples) and merge with yaya_instruct.jsonl
download-openhermes:
	python scripts/download_openhermes.py --max_examples 100000

# Download pretrain data into data/raw/ domain dirs
download-pretrain-data:
	python scripts/download_pretrain_data.py --all

download-pretrain-data-quick:
	python scripts/download_pretrain_data.py --all --max_examples 50000

# Tokenize per-domain + mix into final training shards (data/processed/mixed/)
mix-pretrain-data:
	python scripts/mix_pretrain_data.py

# Kaggle/Colab budget: cap at 3B tokens total
mix-pretrain-data-3b:
	python scripts/mix_pretrain_data.py --total_tokens 3000000000

# Preview what would be mixed without writing
mix-pretrain-data-dry:
	python scripts/mix_pretrain_data.py --dry_run

# Full pretrain data pipeline: download → mix → train
pretrain-pipeline: download-pretrain-data mix-pretrain-data train-1b

# Step 1: Generate training data
gen-data:
	python scripts/generate_training_data.py --sft_count 200 --dpo_count 100

gen-data-large:
	python scripts/generate_training_data.py --sft_count 2000 --dpo_count 500

# Step 2: Fine-tune (LoRA on Qwen 2.5 1.5B by default)
finetune:
	python scripts/finetune_hf.py \
		--base_model Qwen/Qwen2.5-1.5B-Instruct \
		--data_dir data/sft \
		--output_dir outputs/yaya-sft \
		--method lora --epochs 3

finetune-full:
	python scripts/finetune_hf.py \
		--base_model Qwen/Qwen2.5-1.5B-Instruct \
		--data_dir data/sft \
		--output_dir outputs/yaya-sft \
		--method full --epochs 3

finetune-qlora:
	python scripts/finetune_hf.py \
		--base_model Qwen/Qwen2.5-1.5B-Instruct \
		--data_dir data/sft \
		--output_dir outputs/yaya-sft \
		--method qlora --epochs 3

# Step 3: DPO alignment
align-dpo:
	python scripts/align_dpo_hf.py \
		--model_path outputs/yaya-sft-merged \
		--data_dir data/sft \
		--output_dir outputs/yaya-dpo

# Step 4: Evaluate
eval-quick:
	python scripts/eval_model.py --model_path outputs/yaya-sft --quick

eval-full-hf:
	python scripts/eval_model.py \
		--model_path outputs/yaya-dpo \
		--benchmarks mmlu,hellaswag,arc_easy \
		--output outputs/eval_results.json

eval-interactive:
	python scripts/eval_model.py --model_path outputs/yaya-dpo --interactive

# Step 5: Deploy
deploy:
	python scripts/deploy_model.py --model_path outputs/yaya-dpo --port 8000

deploy-quantized:
	python scripts/deploy_model.py --model_path outputs/yaya-dpo --port 8000 --quantize int4

# Full pipeline (generate data -> finetune -> align -> evaluate)
pipeline-full: gen-data finetune align-dpo eval-quick

# Continuous learning: harvest conversations → filter → retrain SFT
learn-tiny:
	python scripts/continuous_learn.py \
		--conversations data/memory/conversations.jsonl \
		--output data/sft/yaya_instruct.jsonl \
		--min_quality 0.6
	python scripts/train_sft.py \
		--model_config configs/model/yaya_tiny.yaml \
		--train_config configs/training/sft_tiny.yaml \
		--pretrain_checkpoint checkpoints/yaya-tiny/checkpoint-00010000

learn-125m:
	python scripts/continuous_learn.py \
		--conversations data/memory/conversations.jsonl \
		--output data/sft/yaya_instruct.jsonl \
		--min_quality 0.6
	python scripts/train_sft.py \
		--model_config configs/model/yaya_125m.yaml \
		--train_config configs/training/sft_125m.yaml \
		--pretrain_checkpoint checkpoints/yaya-125m/latest

# ─── Yaya-tiny convenience targets ─────────────────────────────────────────

## ─────────────────────────────────────────────────────────────
## YAYA-125M — Primary model (129M params, GPT-2 scale)
## All training, chat, and eval targets default to yaya-125m.
## ─────────────────────────────────────────────────────────────

## Pretrain yaya-125m from scratch on available text/reasoning data
pretrain:
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/train_125m.yaml' -RedirectStandardOutput 'logs\pretrain_125m.log' -RedirectStandardError 'logs\pretrain_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## SFT on reasoning+math dataset (starts from pretrain checkpoint or fresh)
sft:
	@CKPT=$$(cat checkpoints/yaya-125m/latest 2>/dev/null || echo ""); \
	if [ -n "$$CKPT" ]; then \
		echo "Starting 125m SFT from pretrain: $$CKPT"; \
		powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml --pretrain_checkpoint checkpoints/yaya-125m/$$CKPT' -RedirectStandardOutput 'logs\sft_125m.log' -RedirectStandardError 'logs\sft_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"; \
	else \
		echo "No pretrain checkpoint — training 125m SFT from random init"; \
		powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml' -RedirectStandardOutput 'logs\sft_125m.log' -RedirectStandardError 'logs\sft_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"; \
	fi

## Resume 125m SFT from latest checkpoint
sft-resume:
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml --resume' -RedirectStandardOutput 'logs\sft_125m.log' -RedirectStandardError 'logs\sft_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## DPO alignment after SFT (auto-detects best 125m SFT checkpoint)
dpo:
	PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -u scripts/train_dpo.py \
		--model_config configs/model/yaya_125m.yaml \
		--max_steps 3000 \
		--save_steps 500 \
		--log_steps 50

## Quick 5-question sanity check — runs in <30s on CPU, auto-finds latest checkpoint
quick-check:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/quick_check.py

## Full 35-question benchmark (saves results + updates dashboard)
benchmark:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/benchmark.py

## Evaluate instruction-following (auto-detects best 125m checkpoint)
eval:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_instruction.py --model_config configs/model/yaya_125m.yaml

## Evaluate math ability (auto-detects best 125m checkpoint)
eval-math:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_math.py

## Evaluate all checkpoints in the 125m-sft directory
eval-all:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_instruction.py \
		--model_config configs/model/yaya_125m.yaml \
		--all_checkpoints checkpoints/yaya-125m-sft

## Run eval loop: eval → augment dataset → print retrain command
eval-loop:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_loop.py

## Run eval loop and immediately launch retraining
eval-loop-retrain:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_loop.py --retrain

## Interactive CLI chat — auto-picks best 125m checkpoint
chat:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/chat_tiny.py

## Interactive chat with calculator tool enabled
chat-calc:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/chat_tiny.py --tool-calc

## Launch Gradio web chat UI
web-ui:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/web_ui.py

## Convert chat logs → SFT training data
continuous-learn:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/continuous_learn.py \
		--output data/sft/yaya_reasoning_combined.jsonl

## Broader self-evaluation benchmark
self-eval:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/self_eval.py

## Full self-improvement: self-eval → augment data → suggest retrain
self-improve:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/self_eval.py && \
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/self_improve.py \
		--eval_report data/eval/self_eval_report.json \
		--output data/sft/yaya_reasoning_combined.jsonl

## Show training log + checkpoint status
training-status:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/training_status.py

## ── Legacy aliases (kept for backward compatibility) ──────────────────────────
chat-tiny: chat
eval-tiny: eval
dpo-tiny: dpo
sft-tiny: sft
sft-tiny-filtered: sft
sft-tiny-focused: sft
sft-tiny-merged: sft

## ─────────────────────────────────────────────────────────────
## MATH CURRICULUM TRAINING
## ─────────────────────────────────────────────────────────────

## Generate all math training data (8 stages + combined, 3298 samples)
gen-math-data:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/generate_math_data.py

## Stage 1 — Arithmetic (600 samples)
## Starts from: yaya-tiny-sft-filtered latest checkpoint
math-stage1:
	@CKPT=$$(cat checkpoints/yaya-tiny-sft-filtered/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No yaya-tiny-sft-filtered checkpoint found. Run sft-tiny-filtered first."; exit 1; fi; \
	echo "Starting math-stage1 from filtered checkpoint: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage1.yaml --pretrain_checkpoint checkpoints/yaya-tiny-sft-filtered/$$CKPT' -RedirectStandardOutput 'logs\math_stage1.log' -RedirectStandardError 'logs\math_stage1_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 2 — Fractions, Decimals, Percentages (500 samples)
## Starts from: math-stage1 latest checkpoint
math-stage2:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage1/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage1 checkpoint found. Run math-stage1 first."; exit 1; fi; \
	echo "Starting math-stage2 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage2.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage1/$$CKPT' -RedirectStandardOutput 'logs\math_stage2.log' -RedirectStandardError 'logs\math_stage2_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 3 — Pre-Algebra (400 samples)
math-stage3:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage2/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage2 checkpoint found. Run math-stage2 first."; exit 1; fi; \
	echo "Starting math-stage3 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage3.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage2/$$CKPT' -RedirectStandardOutput 'logs\math_stage3.log' -RedirectStandardError 'logs\math_stage3_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 4 — Algebra (400 samples)
math-stage4:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage3/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage3 checkpoint found. Run math-stage3 first."; exit 1; fi; \
	echo "Starting math-stage4 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage4.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage3/$$CKPT' -RedirectStandardOutput 'logs\math_stage4.log' -RedirectStandardError 'logs\math_stage4_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 5 — Geometry (400 samples)
math-stage5:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage4/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage4 checkpoint found. Run math-stage4 first."; exit 1; fi; \
	echo "Starting math-stage5 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage5.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage4/$$CKPT' -RedirectStandardOutput 'logs\math_stage5.log' -RedirectStandardError 'logs\math_stage5_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 6 — Statistics & Probability (348 samples)
math-stage6:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage5/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage5 checkpoint found. Run math-stage5 first."; exit 1; fi; \
	echo "Starting math-stage6 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage6.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage5/$$CKPT' -RedirectStandardOutput 'logs\math_stage6.log' -RedirectStandardError 'logs\math_stage6_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 7 — Word Problems (350 samples)
math-stage7:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage6/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage6 checkpoint found. Run math-stage6 first."; exit 1; fi; \
	echo "Starting math-stage7 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage7.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage6/$$CKPT' -RedirectStandardOutput 'logs\math_stage7.log' -RedirectStandardError 'logs\math_stage7_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Stage 8 — Calculus & Beyond (300 samples)
math-stage8:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage7/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage7 checkpoint found. Run math-stage7 first."; exit 1; fi; \
	echo "Starting math-stage8 from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_stage8.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage7/$$CKPT' -RedirectStandardOutput 'logs\math_stage8.log' -RedirectStandardError 'logs\math_stage8_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Combined — all 3298 math samples mixed (run after all stages)
math-combined:
	@CKPT=$$(cat checkpoints/yaya-tiny-math-stage8/latest 2>/dev/null || echo ""); \
	if [ -z "$$CKPT" ]; then echo "ERROR: No math-stage8 checkpoint found. Run math-stage8 first."; exit 1; fi; \
	echo "Starting math-combined from: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/math_combined.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage8/$$CKPT' -RedirectStandardOutput 'logs\math_combined.log' -RedirectStandardError 'logs\math_combined_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Eval math knowledge (runs 20 math questions against latest math checkpoint)
math-eval:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_math.py

## Show math training progress (all stage logs)
math-curriculum:
	@echo "=== Math Curriculum Progress ===" && \
	for stage in 1 2 3 4 5 6 7 8 combined; do \
		logfile="logs/math_stage$${stage}.log"; \
		[ "$$stage" = "combined" ] && logfile="logs/math_combined.log"; \
		if [ -f "$$logfile" ]; then \
			last=$$(grep -oP 'step=\K[0-9]+' "$$logfile" 2>/dev/null | tail -1 || echo "?"); \
			loss=$$(grep -oP 'loss=\K[0-9.]+' "$$logfile" 2>/dev/null | tail -1 || echo "?"); \
			echo "  Stage $$stage: step=$$last  loss=$$loss  ($$logfile)"; \
		else \
			echo "  Stage $$stage: not started"; \
		fi; \
	done

## ─────────────────────────────────────────────────────────────────────────────
## REASONING — Chain-of-Thought + Calculator Tool Use
## ─────────────────────────────────────────────────────────────────────────────

## Build reasoning dataset (math + CoT + calc-tool examples), then train.
## Starts from: yaya-tiny-math-stage2 latest checkpoint.
reasoning:
	@echo "=== Step 1: Generate reasoning data ===" && \
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/generate_reasoning_data.py && \
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/build_reasoning_dataset.py && \
	echo "=== Step 2: Launch reasoning training ===" && \
	CKPT=$$(cat checkpoints/yaya-tiny-math-stage2/latest 2>/dev/null || echo "checkpoint-00000500"); \
	echo "Starting reasoning training from math-stage2: $$CKPT"; \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/sft_reasoning.yaml --pretrain_checkpoint checkpoints/yaya-tiny-math-stage2/$$CKPT' -RedirectStandardOutput 'logs\reasoning.log' -RedirectStandardError 'logs\reasoning_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Resume reasoning training from latest reasoning checkpoint.
reasoning-resume:
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_tiny.yaml --train_config configs/training/sft_reasoning.yaml --resume' -RedirectStandardOutput 'logs\reasoning.log' -RedirectStandardError 'logs\reasoning_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

## Evaluate reasoning + math (20 questions, shows calc tool usage).
eval-reasoning:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/eval_math.py --checkpoint $$(cat checkpoints/yaya-tiny-reasoning/latest 2>/dev/null | xargs -I{} echo checkpoints/yaya-tiny-reasoning/{} || echo checkpoints/yaya-tiny-math-stage2/checkpoint-00000500)

## Interactive chat with calculator tool support.
chat-reasoning:
	PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/chat_tiny.py --checkpoint $$(cat checkpoints/yaya-tiny-reasoning/latest 2>/dev/null | xargs -I{} echo checkpoints/yaya-tiny-reasoning/{} || echo checkpoints/yaya-tiny-math-stage2/checkpoint-00000500) --tool-calc

## ─────────────────────────────────────────────────────────────────────────────
## 125M — Real-scale training (GPT-2 scale, 129M params)
## ─────────────────────────────────────────────────────────────────────────────

## Pretrain Yaya-125M from scratch on available text data.
train-125m:
	@echo "Yaya-125M pretrain — 129M params, requires GPU (T4/A100)" && \
	powershell.exe -Command "cd 'C:\Users\USER\Yaya\yaya-ai'; \$$proc = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-u scripts/train_sft.py --model_config configs/model/yaya_125m.yaml --train_config configs/training/sft_125m.yaml' -RedirectStandardOutput 'logs\train_125m.log' -RedirectStandardError 'logs\train_125m_err.log' -WindowStyle Hidden -PassThru; Write-Host PID: \$$(\$$proc.Id)"

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
