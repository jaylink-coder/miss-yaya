"""Generate Machine Learning SFT data for Yaya.

Covers: supervised/unsupervised/reinforcement learning, key algorithms,
neural networks, training concepts, evaluation, and practical Python.
Uses CoT format so Yaya reasons before answering.
"""

import json, os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

T_OPEN  = "<|think|>"
T_CLOSE = "<|/think|>"
SYSTEM  = ("You are Yaya, a brilliant AI assistant who deeply understands "
           "machine learning and explains it clearly with examples.")

def ex(user, thought, answer):
    return {"messages": [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": f"{T_OPEN}\n{thought}\n{T_CLOSE}\n\n{answer}"},
    ]}

EXAMPLES = [

# ── FUNDAMENTALS ─────────────────────────────────────────────────────────────
ex("What is machine learning?",
"ML is a subfield of AI where models learn patterns from data instead of being\n"
"programmed with explicit rules. Three main paradigms:\n"
"1. Supervised: labelled data (input→output pairs)\n"
"2. Unsupervised: no labels, find structure\n"
"3. Reinforcement: agent learns by rewards",
"**Machine learning** is teaching computers to learn from data rather than following hard-coded rules.\n\n"
"| Type | Data | Goal | Example |\n"
"|------|------|------|---------|\n"
"| **Supervised** | Labelled (X, Y) | Predict Y from X | Spam detection |\n"
"| **Unsupervised** | Unlabelled X | Find structure | Customer segmentation |\n"
"| **Reinforcement** | Rewards/penalties | Maximise reward | Game-playing AI |\n\n"
"Yaya herself is an example: trained on (text, next-token) pairs — supervised learning on language."),

ex("What is the difference between overfitting and underfitting?",
"Bias-variance tradeoff:\n"
"- Underfitting: model too simple, high bias, misses signal\n"
"- Overfitting: model memorises training data, fails on new data (high variance)\n"
"- Goal: find the sweet spot with enough capacity but good generalisation",
"**Underfitting** — model is too simple to learn the pattern:\n"
"- High training error AND high test error\n"
"- Fix: more complex model, more features, longer training\n\n"
"**Overfitting** — model memorises training data but doesn't generalise:\n"
"- Low training error but HIGH test error\n"
"- Fix: more data, dropout, regularisation (L1/L2), early stopping\n\n"
"```\n"
"             Underfit    Just right    Overfit\n"
"Train loss:    high         low          low\n"
"Test loss:     high         low          HIGH\n"
"```\n\n"
"The gap between train and test loss tells you if you're overfitting."),

ex("Explain gradient descent in simple terms.",
"Gradient descent minimises a loss function by iteratively moving in the direction\n"
"of steepest descent (negative gradient).\n"
"Analogy: blindfolded on a hilly landscape, take small steps downhill until you reach a valley.",
"**Gradient descent** finds the minimum of a loss function by taking small steps downhill.\n\n"
"**Algorithm:**\n"
"```\n"
"θ = initial_weights\n"
"for each step:\n"
"    gradient = ∂Loss/∂θ  (how much each weight increases the loss)\n"
"    θ = θ - learning_rate × gradient\n"
"```\n\n"
"**Variants:**\n"
"| Variant | Uses | Speed | Noise |\n"
"|---------|------|-------|-------|\n"
"| Batch GD | Full dataset | Slow | Smooth |\n"
"| Stochastic GD | 1 sample | Fast | Very noisy |\n"
"| **Mini-batch GD** | Small batch | **Best balance** | Moderate |\n\n"
"Modern LLMs use mini-batch SGD with Adam (adaptive learning rates per parameter)."),

ex("What is backpropagation?",
"Backprop computes gradients of the loss w.r.t. all weights by applying the chain rule\n"
"backwards through the computation graph.\n"
"Forward pass: compute predictions and loss\n"
"Backward pass: compute ∂Loss/∂W for every weight layer by layer",
"**Backpropagation** is the algorithm that computes how much each weight contributed to the error.\n\n"
"**Steps:**\n"
"1. **Forward pass** — feed input, compute output and loss\n"
"2. **Backward pass** — apply chain rule from output → input:\n"
"   `∂Loss/∂W₁ = ∂Loss/∂W₃ × ∂W₃/∂W₂ × ∂W₂/∂W₁`\n"
"3. **Update** — `W = W - lr × ∂Loss/∂W`\n\n"
"**Why it works:** The chain rule allows gradients to flow backwards through any differentiable function, no matter how deep the network.\n\n"
"PyTorch does this automatically with `loss.backward()`."),

ex("What is a learning rate and how do I choose one?",
"Learning rate controls step size in gradient descent.\n"
"Too high: overshoots, loss explodes\n"
"Too low: trains too slowly, may get stuck\n"
"Strategies: LR finder, warmup + cosine decay, cyclical LR",
"**Learning rate (lr)** controls how much weights change each step.\n\n"
"**Effects:**\n"
"- `lr` too high → loss diverges or oscillates\n"
"- `lr` too low → trains slowly, may get stuck in local minima\n"
"- `lr` just right → fast convergence to good solution\n\n"
"**How to choose:**\n"
"1. **LR Finder** (best): ramp LR from 1e-7 to 10, find steepest loss drop → use LR/10\n"
"2. **Rule of thumb**: start with 3e-4 for Adam (Karpathy's constant)\n"
"3. **Schedule**: warmup for 1000 steps, then cosine decay\n\n"
"```python\n"
"# Typical 1B model training\n"
"peak_lr = 3e-4\n"
"warmup_steps = 2000\n"
"schedule = 'wsd'  # warmup-stable-decay\n"
"```"),

# ── NEURAL NETWORKS ───────────────────────────────────────────────────────────
ex("What is an activation function and why does a neural network need it?",
"Without activation functions, stacking linear layers is still just a linear function.\n"
"Non-linear activations let the network learn complex patterns.\n"
"Common ones: ReLU (simple, fast), GELU (smooth), SiLU/Swish (used in LLMs), Sigmoid (binary output)",
"**Activation functions** add non-linearity so a neural network can learn complex patterns.\n\n"
"Without them: `Linear(Linear(x)) = Linear(x)` — no matter how many layers, it's just a line.\n\n"
"| Activation | Formula | Used in |\n"
"|-----------|---------|--------|\n"
"| ReLU | `max(0, x)` | CNNs, older nets |\n"
"| GELU | `x·Φ(x)` (smooth ReLU) | BERT, GPT-2 |\n"
"| **SiLU/Swish** | `x·σ(x)` | **LLaMA, Yaya** |\n"
"| Sigmoid | `1/(1+e⁻ˣ)` | Binary classification |\n"
"| Softmax | normalised exp | Output layer (vocab) |\n\n"
"Yaya uses **SwiGLU** (a gated variant of SiLU) in her feed-forward layers — it consistently outperforms ReLU for language models."),

ex("What is dropout and how does it prevent overfitting?",
"Dropout randomly zeros out neurons during training with probability p.\n"
"Forces the network to not rely on any single neuron.\n"
"Acts as training many different network architectures and averaging them (ensemble effect).",
"**Dropout** randomly deactivates neurons during each training step.\n\n"
"```python\n"
"# During training: each neuron active with probability (1-p)\n"
"x = dropout(x, p=0.1)  # 10% chance each neuron is zeroed\n"
"# During inference: all neurons active, outputs scaled by (1-p)\n"
"```\n\n"
"**Why it works:**\n"
"- Prevents co-adaptation (neurons can't rely on specific neighbours)\n"
"- Equivalent to training ~2ⁿ different networks and averaging\n"
"- Acts as regularisation, similar to L2 weight decay\n\n"
"**In transformers:** attention dropout (drop attention weights) is more common than neuron dropout. Yaya uses `attention_dropout=0.0` at pretraining scale — large models with enough data rarely need it."),

ex("Explain the transformer architecture.",
"Transformers (Vaswani et al. 2017) replaced RNNs for sequence tasks.\n"
"Key innovation: attention mechanism — all positions attend to all others in parallel.\n"
"Components: token embeddings + positional encoding, multi-head attention, FFN, layer norm",
"**The Transformer** (2017) revolutionised AI by replacing sequential RNNs with parallel attention.\n\n"
"**Core components:**\n"
"```\n"
"Input tokens\n"
"    │\n"
"Token Embeddings + Positional Encoding\n"
"    │\n"
"[× N layers]\n"
"  ├─ LayerNorm\n"
"  ├─ Multi-Head Self-Attention  (every token attends to every other)\n"
"  ├─ Residual connection (+)\n"
"  ├─ LayerNorm\n"
"  ├─ Feed-Forward Network (2 linear layers + activation)\n"
"  └─ Residual connection (+)\n"
"    │\n"
"Output projection → vocabulary logits\n"
"```\n\n"
"**Why it works:** Attention lets distant tokens communicate directly (no vanishing gradient over distance). Residual connections allow very deep networks to train stably.\n\n"
"**Yaya's architecture:** 20 transformer layers, GQA attention, RoPE positional encoding, SwiGLU FFN, RMSNorm."),

ex("What is attention in neural networks?",
"Attention computes a weighted sum of values based on query-key similarity.\n"
"Each query token looks at all key tokens, computes a similarity score,\n"
"applies softmax to get weights, then takes weighted sum of values.",
"**Attention** lets each token decide which other tokens to focus on.\n\n"
"**Math:**\n"
"```\n"
"Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V\n"
"\n"
"where:\n"
"  Q = queries  (what am I looking for?)\n"
"  K = keys     (what do I have?)\n"
"  V = values   (what do I return?)\n"
"  d_k = key dimension (scaling prevents softmax saturation)\n"
"```\n\n"
"**Multi-Head Attention:** run attention H times in parallel with different projections, then concatenate. Each head learns different relationship types.\n\n"
"**Grouped-Query Attention (GQA):** multiple query heads share fewer KV heads, saving memory. Yaya uses 16 query heads but only 4 KV heads."),

# ── TRAINING TECHNIQUES ───────────────────────────────────────────────────────
ex("What is batch normalisation and why is it important?",
"BatchNorm normalises layer inputs to zero mean, unit variance across a batch.\n"
"Reduces internal covariate shift, allows higher learning rates, acts as regularisation.\n"
"LLMs use LayerNorm or RMSNorm instead (BatchNorm doesn't work well with variable-length text).",
"**Batch Normalisation** normalises each layer's inputs to reduce training instability.\n\n"
"```python\n"
"# For each feature dimension across the batch:\n"
"μ = mean(batch_features)\n"
"σ = std(batch_features)\n"
"x_norm = (x - μ) / (σ + ε)  # normalise\n"
"y = γ * x_norm + β           # learnable scale and shift\n"
"```\n\n"
"**Benefits:** faster training, allows higher LR, mild regularisation effect.\n\n"
"**For transformers:** use **RMSNorm** instead:\n"
"```python\n"
"x_norm = x / rms(x) * weight  # no mean centering needed\n"
"```\n"
"RMSNorm is simpler, faster, and works across variable sequence lengths. Yaya uses RMSNorm."),

ex("What is the difference between L1 and L2 regularisation?",
"Both add penalty terms to loss to discourage large weights.\n"
"L1 (Lasso): penalty = λ|w|, creates sparse solutions (many weights go to zero)\n"
"L2 (Ridge): penalty = λw², penalises large weights but not to zero\n"
"In deep learning, L2 is called weight decay and is standard.",
"Both penalise large weights but behave differently:\n\n"
"| | L1 (Lasso) | L2 (Ridge / Weight Decay) |\n"
"|--|-----------|---------------------------|\n"
"| Penalty | `λ Σ|wᵢ|` | `λ Σwᵢ²` |\n"
"| Effect | Drives weights to **exactly 0** | Shrinks weights towards 0 |\n"
"| Produces | Sparse models (feature selection) | Dense small weights |\n"
"| Use when | Many irrelevant features | General regularisation |\n\n"
"**In LLMs:** L2 (weight decay) is standard. AdamW correctly decouples weight decay from the adaptive learning rate. Yaya trains with `weight_decay=0.1`."),

ex("How does the Adam optimiser work?",
"Adam = Adaptive Moment Estimation.\n"
"Maintains per-parameter running estimates of:\n"
"  m = first moment (mean of gradients, like momentum)\n"
"  v = second moment (mean of squared gradients, like RMSProp)\n"
"Then adapts learning rate per parameter: params with large gradient variance get smaller LR.",
"**Adam** adapts the learning rate for each parameter individually.\n\n"
"```python\n"
"# At each step t:\n"
"m = β₁ * m + (1 - β₁) * grad          # momentum (β₁=0.9)\n"
"v = β₂ * v + (1 - β₂) * grad²         # adaptive scale (β₂=0.95)\n"
"m̂ = m / (1 - β₁ᵗ)                     # bias correction\n"
"v̂ = v / (1 - β₂ᵗ)                     # bias correction\n"
"θ = θ - lr * m̂ / (√v̂ + ε)             # update\n"
"```\n\n"
"**Why it's powerful:** parameters that receive large/inconsistent gradients get a smaller effective LR. Rare parameters get larger updates.\n\n"
"**AdamW** (used by Yaya) decouples weight decay from the adaptive scaling — this is important because classic Adam applies weight decay through the gradient, which interacts poorly with the adaptive scaling."),

# ── EVALUATION ────────────────────────────────────────────────────────────────
ex("What is perplexity and how do I interpret it?",
"Perplexity = exp(cross-entropy loss) for language models.\n"
"Measures how 'surprised' the model is by the test text.\n"
"Lower is better. Human-level English: ~30-60. GPT-2: ~35. GPT-3: ~20. GPT-4: estimated <10.",
"**Perplexity** measures how well a language model predicts a test sequence.\n\n"
"```\n"
"Perplexity = exp(average cross-entropy loss)\n"
"           = exp(-1/N × Σ log P(token_i | context))\n"
"```\n\n"
"**Intuition:** a perplexity of K means the model is as confused as if it had to choose uniformly among K words at each step.\n\n"
"| Model | Perplexity (Penn Treebank) |\n"
"|-------|---|\n"
"| Random (vocab=50K) | 50,000 |\n"
"| N-gram | ~100 |\n"
"| GPT-2 | ~35 |\n"
"| GPT-3 | ~20 |\n"
"| GPT-4 | <10 (est.) |\n\n"
"**Yaya training target:** perplexity < 30 after pretraining on 3B Wikipedia tokens."),

# ── PRACTICAL PYTHON ──────────────────────────────────────────────────────────
ex("How do I build a simple neural network in PyTorch?",
"PyTorch uses nn.Module as the base class.\n"
"Define layers in __init__, define forward pass in forward().\n"
"Training loop: forward, compute loss, backward, optimizer step.",
"""```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)

# Training loop
model = SimpleNet(784, 256, 10)  # e.g., MNIST
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for x, y in dataloader:
    pred = model(x)           # forward pass
    loss = loss_fn(pred, y)   # compute loss
    loss.backward()           # backpropagation
    optimizer.step()          # update weights
    optimizer.zero_grad()     # reset gradients
```"""),

ex("What is transfer learning and when should I use it?",
"Transfer learning: take a model pretrained on large data, fine-tune on your specific task.\n"
"Works because early layers learn universal features (edges, syntax) useful for many tasks.\n"
"Use when: you have limited labelled data, task is similar to pretraining domain.",
"**Transfer learning** reuses a model trained on a large task for a new, smaller task.\n\n"
"**Why it works:** A pretrained model has already learned:\n"
"- Text models: grammar, facts, reasoning patterns\n"
"- Image models: edges, textures, object parts\n"
"These features are useful across many downstream tasks.\n\n"
"**When to use it:**\n"
"- You have < 100K labelled examples (fine-tune beats training from scratch)\n"
"- Your task domain overlaps with pretraining data\n"
"- You have limited compute\n\n"
"**Yaya's pipeline:** pretrain on 3B Wikipedia tokens → SFT on 100K instruction examples → DPO alignment. Each stage builds on the previous."),

ex("Explain the bias-variance tradeoff.",
"Bias: systematic error from model being too simple (underfits)\n"
"Variance: sensitivity to training data fluctuations (overfits)\n"
"Total error = bias² + variance + irreducible noise\n"
"Increasing model complexity decreases bias but increases variance — find the balance.",
"**Bias-variance tradeoff:**\n\n"
"```\n"
"Total Error = Bias² + Variance + Irreducible Noise\n"
"```\n\n"
"| | High Bias | High Variance |\n"
"|--|-----------|---------------|\n"
"| Cause | Too simple model | Too complex model |\n"
"| Training error | High | Low |\n"
"| Test error | High | High |\n"
"| Fix | More capacity | More data or regularisation |\n\n"
"**Intuition:**\n"
"- **Bias** = how wrong on average (systematic error)\n"
"- **Variance** = how much predictions vary with different training sets\n\n"
"Modern deep learning breaks this tradeoff: very large models (like GPT-4 or Yaya-1B) have low bias AND low variance when trained on enough data — the **double descent** phenomenon."),
]

def main():
    out = "data/sft/yaya_ml.jsonl"
    os.makedirs("data/sft", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in EXAMPLES:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Generated {len(EXAMPLES)} ML examples -> {out}")

    instruct = "data/sft/yaya_instruct.jsonl"
    if os.path.exists(instruct):
        with open(instruct, "a", encoding="utf-8") as f:
            for e in EXAMPLES:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Appended {len(EXAMPLES)} ML examples -> {instruct}")

if __name__ == "__main__":
    main()
