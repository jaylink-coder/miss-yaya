"""Configuration loading and model config dataclasses."""

from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    enabled: bool = False
    image_size: int = 336
    patch_size: int = 14
    vision_hidden_size: int = 1024
    vision_layers: int = 24
    vision_heads: int = 16
    projection_dim: int = 2048


def _resolve_moe_layer(moe_layers: str, layer_idx: int) -> bool:
    """Determine if a given layer index should use MoE.

    Single source of truth — shared by ModelConfig and MoEConfig so the
    routing logic never diverges.
    """
    if moe_layers == "all":
        return True
    if moe_layers == "alternate":
        return layer_idx % 2 == 1
    try:
        indices = {int(x.strip()) for x in moe_layers.split(",")}
        return layer_idx in indices
    except ValueError:
        return False


@dataclass
class ModelConfig:
    """Yaya model architecture configuration."""
    model_name: str = "yaya-1.5b"
    vocab_size: int = 64000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    initializer_range: float = 0.02
    dtype: str = "bfloat16"
    vision: VisionConfig = field(default_factory=VisionConfig)

    # MoE (Mixture-of-Experts) — opt-in, disabled by default
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_layers: str = "alternate"          # "all", "alternate", or "0,2,4,..."
    moe_load_balance_coeff: float = 0.01
    moe_router_jitter: float = 0.01

    def is_moe_layer(self, layer_idx: int) -> bool:
        if not self.moe_enabled:
            return False
        return _resolve_moe_layer(self.moe_layers, layer_idx)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    max_steps: int = 100000
    max_seq_length: int = 4096
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    seed: int = 42

    # Checkpointing
    save_steps: int = 1000
    save_dir: str = "checkpoints"
    keep_last_n: int = 5

    # Logging
    log_steps: int = 10
    wandb_project: str = "yaya-ai"
    wandb_run_name: Optional[str] = None

    # Evaluation
    eval_steps: int = 5000
    eval_samples: int = 1000

    # Data
    train_data: str = "data/processed/train"
    eval_data: str = "data/processed/eval"
    tokenizer_path: str = "data/tokenizer/yaya_tokenizer.model"
    num_workers: int = 4

    # Distributed
    distributed_strategy: str = "deepspeed_zero2"
    cpu_offload: bool = False

    # Advanced training
    grad_noise_eta: float = 0.0       # Gradient noise (0 = off; try 0.01 to escape minima)
    layer_lr_decay: float = 1.0       # Layer-wise LR decay (1.0 = off; try 0.9 for fine-tune)
    ema_decay: float = 0.0            # EMA weight decay (0 = off; try 0.9999 for eval)

    # LoRA — parameter-efficient fine-tuning (0 = disabled)
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj"

    # EWC — continual learning / catastrophic forgetting prevention (0 = disabled)
    ewc_lambda: float = 0.0
    ewc_fisher_samples: int = 200

    # Online learning — learn from inference-time feedback (disabled by default)
    online_learning_enabled: bool = False
    online_buffer_capacity: int = 1000
    online_finetune_every: int = 50
    online_micro_steps: int = 10
    online_micro_lr: float = 5e-5

    # ElasticGuard — neuroelastic resilience wrapper for online learning
    elastic_guard_enabled: bool = False
    elastic_loss_spike_ratio: float = 2.5
    elastic_max_grad_norm: float = 20.0
    elastic_cooldown_seconds: float = 60.0
    elastic_max_per_minute: int = 120
    elastic_max_adapter_norm: float = 100.0

    # Synthetic replay — generative rehearsal of past-task anchors
    synthetic_replay_enabled: bool = False
    synthetic_replay_anchors: int = 20
    synthetic_replay_samples: int = 2
    synthetic_replay_max_tokens: int = 64
    synthetic_replay_weight: float = 0.3

    # Continual learning metrics — track forgetting across eval phases
    track_forgetting: bool = False
    task_id: str = "default"  # Task name for ForgettingTracker.record()

    # MAML meta-learning — fast adaptation from few examples
    maml_enabled: bool = False
    maml_inner_lr: float = 0.01
    maml_inner_steps: int = 5
    maml_meta_batch_size: int = 4

    # Sparse plasticity — top-K gradient masking in online learning (0 = off)
    sparse_gradient_k: float = 0.0

    # Alignment monitoring — capability drift detection (disabled by default)
    alignment_monitor_enabled: bool = False
    alignment_kl_threshold: float = 0.5
    alignment_entropy_threshold: float = 0.3
    alignment_score_regression_threshold: float = 0.10

    # Human-in-the-loop oversight in ElasticGuard
    human_review_enabled: bool = False
    human_review_z_threshold: float = 3.0

    # Curriculum learning — difficulty-aware training schedule
    curriculum_enabled: bool = False
    curriculum_strategy: str = "linear"         # "linear" | "step" | "competence"
    curriculum_warmup_easy_steps: int = 10_000  # steps before hard examples
    curriculum_easy_ceiling: float = 0.4        # max difficulty during warmup
    curriculum_competence_loss_threshold: float = 2.5  # for "competence" strategy

    # Reward model — lightweight scorer for RLHF / auto-feedback quality
    reward_model_enabled: bool = False
    reward_model_path: str = ""    # path to saved reward head weights (empty = untrained)


def load_model_config(path: str) -> ModelConfig:
    """Load model config from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    arch = raw.get("architecture", raw)
    vision_raw = raw.get("vision", {})
    vision_cfg = VisionConfig(**vision_raw) if vision_raw else VisionConfig()

    hidden_size = arch.get("hidden_size", 2048)
    num_heads = arch.get("num_attention_heads", 16)
    num_kv_heads = arch.get("num_key_value_heads", 4)

    # Validate architecture consistency
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_heads})"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"num_key_value_heads ({num_kv_heads}) for GQA"
        )

    vocab_size = arch.get("vocab_size", 64000)
    if vocab_size % 8 != 0:
        import warnings
        warnings.warn(
            f"vocab_size ({vocab_size}) is not a multiple of 8. "
            f"This may hurt GPU efficiency. Consider padding to {((vocab_size + 7) // 8) * 8}."
        )

    dtype = raw.get("dtype", "bfloat16")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {dtype!r}. Use float32, float16, or bfloat16.")

    return ModelConfig(
        model_name=raw.get("model_name", "yaya"),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=arch.get("intermediate_size", 5632),
        num_hidden_layers=arch.get("num_hidden_layers", 24),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=arch.get("max_position_embeddings", 4096),
        rope_theta=arch.get("rope_theta", 10000.0),
        rms_norm_eps=arch.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=arch.get("tie_word_embeddings", True),
        hidden_act=arch.get("hidden_act", "silu"),
        attention_dropout=arch.get("attention_dropout", 0.0),
        attention_bias=arch.get("attention_bias", False),
        mlp_bias=arch.get("mlp_bias", False),
        initializer_range=raw.get("initializer_range", 0.02),
        dtype=dtype,
        vision=vision_cfg,
        moe_enabled=arch.get("moe_enabled", False),
        moe_num_experts=arch.get("moe_num_experts", 8),
        moe_top_k=arch.get("moe_top_k", 2),
        moe_layers=arch.get("moe_layers", "alternate"),
        moe_load_balance_coeff=arch.get("moe_load_balance_coeff", 0.01),
        moe_router_jitter=arch.get("moe_router_jitter", 0.01),
    )


def load_training_config(path: str) -> TrainingConfig:
    """Load training config from YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    training = raw.get("training", {})
    ckpt = raw.get("checkpointing", {})
    logging = raw.get("logging", {})
    eval_cfg = raw.get("eval", {})
    data = raw.get("data", {})
    dist = raw.get("distributed", {})

    return TrainingConfig(
        per_device_batch_size=training.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 8),
        learning_rate=training.get("learning_rate", 3e-4),
        weight_decay=training.get("weight_decay", 0.1),
        adam_beta1=training.get("adam_beta1", 0.9),
        adam_beta2=training.get("adam_beta2", 0.95),
        adam_epsilon=training.get("adam_epsilon", 1e-8),
        max_grad_norm=training.get("max_grad_norm", 1.0),
        lr_scheduler=training.get("lr_scheduler", "cosine"),
        warmup_steps=training.get("warmup_steps", 2000),
        min_lr_ratio=training.get("min_lr_ratio", 0.1),
        max_steps=training.get("max_steps", 100000),
        max_seq_length=training.get("max_seq_length", 4096),
        dtype=training.get("dtype", "bfloat16"),
        gradient_checkpointing=dist.get("gradient_checkpointing", True),
        seed=raw.get("seed", 42),
        save_steps=ckpt.get("save_steps", 1000),
        save_dir=ckpt.get("save_dir", "checkpoints"),
        keep_last_n=ckpt.get("keep_last_n", 5),
        log_steps=logging.get("log_steps", 10),
        wandb_project=logging.get("wandb_project", "yaya-ai"),
        wandb_run_name=logging.get("wandb_run_name"),
        eval_steps=eval_cfg.get("eval_steps", 5000),
        eval_samples=eval_cfg.get("eval_samples", 1000),
        train_data=data.get("train_data", "data/processed/train"),
        eval_data=data.get("eval_data", "data/processed/eval"),
        tokenizer_path=data.get("tokenizer_path", "data/tokenizer/yaya_tokenizer.model"),
        num_workers=data.get("num_workers", 4),
        distributed_strategy=dist.get("strategy", "deepspeed_zero2"),
        cpu_offload=dist.get("cpu_offload", False),
        grad_noise_eta=training.get("grad_noise_eta", 0.0),
        layer_lr_decay=training.get("layer_lr_decay", 1.0),
        ema_decay=training.get("ema_decay", 0.0),
        lora_rank=training.get("lora_rank", 0),
        lora_alpha=training.get("lora_alpha", 32.0),
        lora_dropout=training.get("lora_dropout", 0.05),
        lora_target_modules=training.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj"),
        ewc_lambda=training.get("ewc_lambda", 0.0),
        ewc_fisher_samples=training.get("ewc_fisher_samples", 200),
        online_learning_enabled=training.get("online_learning_enabled", False),
        online_buffer_capacity=training.get("online_buffer_capacity", 1000),
        online_finetune_every=training.get("online_finetune_every", 50),
        online_micro_steps=training.get("online_micro_steps", 10),
        online_micro_lr=training.get("online_micro_lr", 5e-5),
        elastic_guard_enabled=training.get("elastic_guard_enabled", False),
        elastic_loss_spike_ratio=training.get("elastic_loss_spike_ratio", 2.5),
        elastic_max_grad_norm=training.get("elastic_max_grad_norm", 20.0),
        elastic_cooldown_seconds=training.get("elastic_cooldown_seconds", 60.0),
        elastic_max_per_minute=training.get("elastic_max_per_minute", 120),
        elastic_max_adapter_norm=training.get("elastic_max_adapter_norm", 100.0),
        synthetic_replay_enabled=training.get("synthetic_replay_enabled", False),
        synthetic_replay_anchors=training.get("synthetic_replay_anchors", 20),
        synthetic_replay_samples=training.get("synthetic_replay_samples", 2),
        synthetic_replay_max_tokens=training.get("synthetic_replay_max_tokens", 64),
        synthetic_replay_weight=training.get("synthetic_replay_weight", 0.3),
        track_forgetting=training.get("track_forgetting", False),
        task_id=training.get("task_id", "default"),
        maml_enabled=training.get("maml_enabled", False),
        maml_inner_lr=training.get("maml_inner_lr", 0.01),
        maml_inner_steps=training.get("maml_inner_steps", 5),
        maml_meta_batch_size=training.get("maml_meta_batch_size", 4),
        sparse_gradient_k=training.get("sparse_gradient_k", 0.0),
        alignment_monitor_enabled=training.get("alignment_monitor_enabled", False),
        alignment_kl_threshold=training.get("alignment_kl_threshold", 0.5),
        alignment_entropy_threshold=training.get("alignment_entropy_threshold", 0.3),
        alignment_score_regression_threshold=training.get(
            "alignment_score_regression_threshold", 0.10
        ),
        human_review_enabled=training.get("human_review_enabled", False),
        human_review_z_threshold=training.get("human_review_z_threshold", 3.0),
        curriculum_enabled=training.get("curriculum_enabled", False),
        curriculum_strategy=training.get("curriculum_strategy", "linear"),
        curriculum_warmup_easy_steps=training.get("curriculum_warmup_easy_steps", 10_000),
        curriculum_easy_ceiling=training.get("curriculum_easy_ceiling", 0.4),
        curriculum_competence_loss_threshold=training.get(
            "curriculum_competence_loss_threshold", 2.5
        ),
        reward_model_enabled=training.get("reward_model_enabled", False),
        reward_model_path=training.get("reward_model_path", ""),
    )
