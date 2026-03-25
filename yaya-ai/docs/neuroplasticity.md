# Yaya Neuroplasticity & Neuroelasticity

A map of every biological/algorithmic concept to its Yaya implementation,
with code pointers, config knobs, and usage guidance.

---

## What is neuro-elasticity?

**Neuro-elasticity = adaptive plasticity + controlled consolidation.**

| Property | What it means | Yaya mechanism |
|---|---|---|
| Adaptive plasticity | Model rewires to learn new patterns quickly | LoRA, OnlineLearner, MAML |
| Controlled consolidation | Prevents catastrophic forgetting | EWC, SyntheticReplay, ElasticGuard |
| Resilience | Recovers from bad updates without corruption | ElasticGuard circuit breaker + rollback |
| Expert routing | New tasks get new capacity without disturbing old | MoE |

---

## Core mechanisms

### 1. LoRA — Parameter-Efficient Fast Adaptation

**What it does:** Injects trainable low-rank matrices `lora_A` × `lora_B` alongside
frozen base weights.  Only ~0.5% of parameters are updated per task.

**Files:**
- [`src/model/lora.py`](../src/model/lora.py) — `LoRAConfig`, `LoRALinear`, `inject_lora()`, `merge_lora()`

**Key config (in training YAML):**
```yaml
training:
  lora_rank: 16        # Rank of adapter matrices (4 for edge, 16 for cloud)
  lora_alpha: 32.0     # Scaling factor (effective lr = alpha/rank)
  lora_dropout: 0.05
  lora_target_modules: q_proj,k_proj,v_proj,o_proj
```

**When to use:** Any fine-tuning where you want to preserve base weights.
Mandatory for MAML inner loop and OnlineLearner on edge devices.

---

### 2. EWC — Elastic Weight Consolidation

**What it does:** Penalizes changing parameters that were important for prior tasks,
using the diagonal Fisher Information Matrix as importance weights.

`L_total = L_task + (λ/2) * Σ_i F_i * (θ_i − θ*_i)²`

**Dynamic λ:** `ewc.penalty(dynamic_lambda=True)` scales λ by `1 + mean_squared_drift`
— the further the model drifts, the stronger the anchor becomes.

**Files:**
- [`src/training/ewc.py`](../src/training/ewc.py) — `EWC`, `compute_fisher()`, `penalty()`, `drift_magnitude()`

**Key config:**
```yaml
training:
  ewc_lambda: 5000.0          # 0 = disabled
  ewc_fisher_samples: 200     # Batches used to estimate Fisher diagonal
```

**Workflow:**
```python
ewc = EWC(model, lambda_ewc=5000.0)
ewc.compute_fisher(reference_dataloader, num_samples=200)
# In training loop:
loss = cross_entropy_loss + ewc.penalty(dynamic_lambda=True)
```

---

### 3. Online Learning — Inference-Time Feedback Loop

**What it does:** Collects `(prompt, response, score)` feedback at inference time
into a rolling buffer, then micro-finetuning fires automatically every N examples.

**Files:**
- [`src/training/online_learner.py`](../src/training/online_learner.py) — `OnlineLearner`, `OnlineLearnerConfig`

**Key config:**
```yaml
training:
  online_learning_enabled: true
  online_buffer_capacity: 1000   # Rolling window of examples
  online_finetune_every: 50      # Trigger micro-finetune after N additions
  online_micro_steps: 10         # Gradient steps per trigger
  online_micro_lr: 5.0e-5
```

**Usage in generator:**
```python
generator = TextGenerator(model, tokenizer, online_learner=learner)
response = generator.generate(prompt, feedback=0.9)  # score triggers learning
```

---

### 4. Synthetic Replay — Privacy-Friendly Generative Rehearsal

**What it does:** Stores representative prompts (anchors) from prior tasks.
At replay time, generates synthetic completions from those anchors and adds
a language-modelling loss to reinforce the model's own prior outputs.

No raw user data is stored — only prompt strings.  Ideal for edge/privacy deployments.

**Files:**
- [`src/training/synthetic_replay.py`](../src/training/synthetic_replay.py) — `SyntheticReplay`, `ReplayConfig`

**Key config:**
```yaml
training:
  synthetic_replay_enabled: true
  synthetic_replay_anchors: 20    # Max stored prompts
  synthetic_replay_samples: 2     # Synthetic completions per anchor
  synthetic_replay_max_tokens: 64
  synthetic_replay_weight: 0.3    # Weight of replay loss vs task loss
```

**Manual usage:**
```python
replay = SyntheticReplay(model, tokenizer)
replay.add_anchor("Explain photosynthesis:")
replay.add_anchor("Translate to Swahili:")

# In training step:
total_loss = task_loss + replay.replay_loss()
```

---

### 5. Mixture-of-Experts — Dynamic Capacity Routing

**What it does:** Replaces selected dense FFN layers with N independent expert networks.
A learned router selects the top-K experts per token, so new tasks can use underutilized
experts without disturbing others.

**Expert utilization monitoring** detects collapse before it degrades performance.

**Files:**
- [`src/model/moe.py`](../src/model/moe.py) — `MoEFeedForward`, `MoERouter`, `convert_to_moe()`

**Key config (in model YAML):**
```yaml
architecture:
  moe_enabled: true
  moe_num_experts: 8
  moe_top_k: 2
  moe_layers: alternate      # "all", "alternate", or "0,2,4,..."
  moe_load_balance_coeff: 0.01
```

**Monitoring expert health:**
```python
for layer in model.model.layers:
    if hasattr(layer.mlp, "routing_stats"):
        stats = layer.mlp.routing_stats()
        if stats["collapse_detected"]:
            print(f"Expert collapse in layer {i}!")
```

---

### 6. MAML — Meta-Learning for Fast Adaptation

**What it does:** Trains the model's initial weights to be maximally adaptable —
1–5 gradient steps on a new task's support set yields strong query performance.
Uses LoRA-only inner loop for parameter efficiency (~200x cheaper than full MAML).

**Files:**
- [`src/training/maml.py`](../src/training/maml.py) — `MAML`, `MAMLConfig`

**Key config:**
```yaml
training:
  maml_enabled: true
  maml_inner_lr: 0.01      # Task-specific step size
  maml_inner_steps: 5      # Gradient steps per task
  maml_meta_batch_size: 4  # Tasks per outer update
```

**Fast adaptation at inference:**
```python
maml = MAML(model, MAMLConfig(inner_lr=0.01, inner_steps=5))
original = maml.snapshot_params()

# Adapt to a new domain with 5 examples
adapted = maml.adapt(support_batch, steps=5)
maml.apply_params(adapted)
response = model.generate(...)

# Restore base weights
maml.restore_params(original)
```

---

### 7. ElasticGuard — Neuroelastic Resilience Layer

**What it does:** Wraps `OnlineLearner` with four protective mechanisms:

| Mechanism | Protection | Config key |
|---|---|---|
| Rollback | Reverts adapter weights if loss spikes | `elastic_loss_spike_ratio` |
| Circuit breaker | Halts learning on NaN / repeated spikes | `elastic_cooldown_seconds` |
| Feedback validation | Clamps scores, rate-limits, rejects outliers | `elastic_max_per_minute` |
| Adapter health | Warns before adapter saturation | `elastic_max_adapter_norm` |

**Files:**
- [`src/training/neuro_elastic.py`](../src/training/neuro_elastic.py) — `ElasticGuard`, `ElasticConfig`

**Key config:**
```yaml
training:
  elastic_guard_enabled: true
  elastic_loss_spike_ratio: 2.5
  elastic_cooldown_seconds: 60.0
  elastic_max_per_minute: 120
  elastic_max_adapter_norm: 100.0
```

**Health monitoring:**
```python
report = guard.health_report()
# {circuit_tripped, cooldown_remaining_s, rollbacks, adapter_norms, ...}
```

---

### 8. Forgetting Tracker — Continual Learning Metrics

**What it does:** Records per-task performance across training phases and computes
standard continual learning metrics (forgetting, backward transfer, plasticity).

**Files:**
- [`src/training/continual_metrics.py`](../src/training/continual_metrics.py) — `ForgettingTracker`, `TaskRecord`

**Key config:**
```yaml
training:
  track_forgetting: true
  task_id: my_task_name   # Name for this training phase
```

**Manual usage:**
```python
tracker = ForgettingTracker()
tracker.record("sentiment", phase=0, score=0.92)
tracker.record("translation", phase=0, score=0.45)  # zero-shot
tracker.record("sentiment", phase=1, score=0.81)    # after training on translation
tracker.record("translation", phase=1, score=0.88)

report = tracker.report()
print(f"avg_forgetting={report['avg_forgetting']:.3f}")
print(f"backward_transfer={report['backward_transfer']:.3f}")
```

---

## Edge / Mobile Deployment

Use `configs/training/edge_online.yaml` for low-power devices:

| Parameter | Edge value | Cloud value | Why |
|---|---|---|---|
| `lora_rank` | 4 | 16 | ~4× fewer adapter params |
| `online_buffer_capacity` | 50 | 1000 | ~20× less RAM |
| `online_micro_steps` | 3 | 10 | ~3× less compute per trigger |
| `elastic_max_per_minute` | 30 | 120 | Aggressive rate limiting |
| `elastic_cooldown_seconds` | 300 | 60 | 5-min cooldown (avoid thrashing) |
| `synthetic_replay_anchors` | 10 | 20 | Half the anchor storage |
| `dtype` | `float16` | `bfloat16` | float16 available on mobile GPU |

---

## Stability–Plasticity Tuning Guide

| Goal | Adjustment |
|---|---|
| Reduce forgetting | Increase `ewc_lambda`, enable `synthetic_replay` |
| Increase adaptability | Increase `online_micro_steps`, lower `elastic_loss_spike_ratio` |
| Protect edge device | Enable `elastic_guard_enabled`, reduce `online_micro_steps` |
| Monitor health | Enable `track_forgetting`, call `guard.health_report()` |
| Fast new-task learning | Enable MAML, reduce `maml_inner_lr` |
| Privacy on device | Enable `synthetic_replay`, disable raw buffer logging |

---

## Full Neuroplastic Stack (all mechanisms active)

```yaml
training:
  # LoRA adapters
  lora_rank: 16
  lora_alpha: 32.0

  # EWC consolidation
  ewc_lambda: 5000.0
  ewc_fisher_samples: 200

  # Online learning
  online_learning_enabled: true
  online_buffer_capacity: 1000
  online_finetune_every: 50
  online_micro_steps: 10
  online_micro_lr: 5.0e-5

  # Synthetic replay
  synthetic_replay_enabled: true
  synthetic_replay_anchors: 20
  synthetic_replay_weight: 0.3

  # ElasticGuard
  elastic_guard_enabled: true
  elastic_loss_spike_ratio: 2.5
  elastic_cooldown_seconds: 60.0

  # MAML (server only)
  maml_enabled: true
  maml_inner_lr: 0.01
  maml_inner_steps: 5

  # Metrics
  track_forgetting: true
  task_id: full_stack
```

---

## References

| Concept | Paper |
|---|---|
| EWC | Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks" |
| MAML | Finn, Abbeel, Levine (2017) "Model-Agnostic Meta-Learning for Fast Adaptation" |
| Synthetic replay | Shin et al. (2017) "Continual Learning with Deep Generative Models" |
| MoE routing | Lepikhin et al. (2021) "GShard"; Fedus et al. (2021) "Switch Transformer" |
| Forgetting metrics | Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory" |
| LoRA | Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" |
| Scaling laws | Kaplan et al. (2020) "Scaling Laws for Neural Language Models" |
