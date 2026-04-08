# Yaya AI — SFT & Alignment Data

> Last updated: 2026-04-07

## Core Training Data

| File | Examples | Purpose |
|---|---|---|
| `openhermes.jsonl` | ~100K | OpenHermes 2.5 instruction data (downloaded from HF) |
| `yaya_instruct.jsonl` | ~100K+ | Merged instruction set (OpenHermes + custom) |
| `yaya_instruct_clean.jsonl` | ~30K | Cleaned/filtered subset of yaya_instruct |
| `yaya_reasoning_combined.jsonl` | ~3,455 | Local chain-of-thought reasoning examples |
| `yaya_short_qa.jsonl` | 2,683 | Direct Q&A pairs (arithmetic, facts, identity, reasoning, language) |
| `yaya_concise_sft.jsonl` | 3,683 | Examples with "Answer directly, no lists" system prompts |
| `yaya_factual_qa.jsonl` | ~500 | Factual knowledge Q&A |
| `yaya_qa_focused.jsonl` | ~2,500 | Q&A-focused training data |

## Phase-Specific Data

| File | Purpose |
|---|---|
| `yaya_phase2_multiturn.jsonl` | Multi-turn conversation examples |
| `yaya_phase3_reasoning.jsonl` | Reasoning and logic examples |
| `yaya_phase4_swahili.jsonl` | Swahili language examples |
| `yaya_phase5_knowledge.jsonl` | World knowledge examples |
| `yaya_phase6_code.jsonl` | Code generation examples |
| `yaya_patch_sft.jsonl` | Patch training data (targeted fixes) |

## Domain-Specific Data

| File | Purpose |
|---|---|
| `yaya_kenya.jsonl` | Kenya-specific knowledge |
| `yaya_kenya_swahili.jsonl` | Kenya + Swahili bilingual data |
| `yaya_ml.jsonl` | Machine learning knowledge |
| `yaya_cot.jsonl` | Chain-of-thought examples |
| `yaya_targeted.jsonl` | Targeted weakness remediation data |
| `math/` | 8-stage math curriculum data (3,298 samples) |
| `teach/quick_facts.jsonl` | 1,000 direct Q&A facts |

## DPO / Alignment Data

| File | Examples | Purpose |
|---|---|---|
| `yaya_dpo_combined.jsonl` | 9,560 | Combined DPO preference pairs (anti-list + general) |
| `yaya_dpo_synthetic.jsonl` | ~5K | Synthetically generated DPO pairs |
| `yaya_dpo_train.jsonl` | ~400 | Original DPO training split |
| `yaya_dpo_eval.jsonl` | ~200 | DPO evaluation split |
| `dpo_pairs.jsonl` | ~30 | Small initial DPO pair set |

## Evaluation Splits

| File | Purpose |
|---|---|
| `yaya_instruct_eval.jsonl` | Eval split from instruct data |
| `yaya_instruct_clean_eval.jsonl` | Eval split from cleaned instruct data |

## Backups (gitignored, not committed)

| File | Notes |
|---|---|
| `yaya_instruct_backup.jsonl` | Backup before merge operations |
| `yaya_instruct_backup2.jsonl` | Second backup |
| `yaya_instruct_merged.jsonl` | Intermediate merge artifact |
| `yaya_instruct_filtered.jsonl` | Intermediate filter artifact |
| `yaya_instruct_filtered_eval.jsonl` | Eval split of filtered data |

## Data Format

All files use JSONL (one JSON object per line) with the chat-messages format:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

DPO files use the preference-pair format:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

## Tokenizer

`../tokenizer/yaya_tokenizer.model` — SentencePiece BPE, vocab size 32,768.
