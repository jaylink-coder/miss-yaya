# Yaya-125M Training Roadmap — True AI Project

**Goal**: Transform Yaya from a benchmark-passing QA bot into a genuine conversational AI  
**Model**: 129M parameters, GQA, RoPE, SwiGLU, 32K vocab  
**Constraint**: ≤2000 steps per Kaggle run (T4 GPU, ~12h)  
**Philosophy**: Each phase builds on the previous — never restart from scratch

---

## Capability Tiers

| Tier | Description | Current State |
|------|-------------|---------------|
| T1 | Direct Q&A (facts, arithmetic) | DONE — 100% benchmark |
| T2 | Instruction following + multi-turn | Phase 2 |
| T3 | Reasoning + explanation | Phase 3 |
| T4 | Swahili fluency | Phase 4 |
| T5 | Knowledge depth (science, health, business) | Phase 5 |
| T6 | Code generation + structured output | Phase 6 |
| T7 | Alignment (tone, safety, accuracy) | Phase 7 DPO |
| T8 | Conversational persona + creativity | Phase 8 |

---

## Phase 0 — Foundation (COMPLETE)
**Checkpoint**: `patch-checkpoint-00000500`  
**Benchmark**: 87/87 (100%) on extended suite  
**What it can do**: Direct single-turn Q&A, arithmetic, factual recall (guarded)  
**What it cannot do**: Multi-turn, reasoning steps, Swahili conversation, code

---

## Phase 1 — Kenya/Swahili Injection (READY TO RUN)
**Checkpoint target**: `p1-checkpoint-00000500`  
**Steps**: 500 · lr=3e-6 · from patch-500  
**Data**: 645 Kenya/Swahili examples × 4 + existing patch data  
**Target**: Model knows Kenya facts and Swahili vocab WITHOUT guards  
**Success**: Kenya & Swahili benchmark passes without fact/guard intercepts

---

## Phase 2 — Instruction Following & Multi-Turn (BUILD NEXT)
**Checkpoint target**: `p2-checkpoint-00002000`  
**Steps**: 2000 · lr=8e-6 · from p1  
**Data**: ~3500 multi-turn conversations (3–6 turns each)  
**Training focus**:
- Context maintenance across turns ("it", "that", "the one you mentioned")
- Follow-up questions handled naturally
- Complex multi-step instructions ("First do X, then Y, finally Z")
- Polite refusals for impossible requests
- Clarifying questions when request is ambiguous

**Success criteria**:
- Handles 5-turn conversations coherently
- References previous context correctly
- Follows format instructions (list, numbered, paragraph)

---

## Phase 3 — Chain-of-Thought Reasoning (BUILD NEXT)
**Checkpoint target**: `p3-checkpoint-00002000`  
**Steps**: 2000 · lr=6e-6 · from p2  
**Data**: ~4500 reasoning examples with visible thinking steps  
**Training focus**:
- "Let me think through this step by step..."
- Math word problems with full working shown
- Logic puzzles with deductive steps
- Analogical reasoning ("X is to Y as A is to B")
- Estimation and approximation skills
- "I don't know" for genuinely unknown facts

**Success criteria**:
- Shows working on math problems (not just answers)
- Deductive logic correct 80%+ of time
- Honest about uncertainty

---

## Phase 4 — Swahili Fluency (BUILD NEXT)
**Checkpoint target**: `p4-checkpoint-00002000`  
**Steps**: 2000 · lr=5e-6 · from p3  
**Data**: ~5000 Swahili/bilingual examples  
**Training focus**:
- Full Swahili conversations (not just vocabulary)
- Code-switching (Swahili + English mixed, natural in Kenya)
- Swahili explanations of concepts
- Kenyan colloquialisms (Sheng words, informal registers)
- Asking and answering questions in Swahili

**Success criteria**:
- Can hold a 5-turn conversation entirely in Swahili
- Code-switching feels natural (not awkward)
- Understands Sheng/informal Kenyan expressions

---

## Phase 5 — Knowledge Depth (BUILD NEXT)
**Checkpoint target**: `p5-checkpoint-00002000`  
**Steps**: 2000 · lr=4e-6 · from p4  
**Data**: ~6000 Q&A with detailed explanations  
**Training focus**:
- Science (biology, chemistry, physics at high-school level)
- Health & medicine (common diseases, symptoms, when to see a doctor)
- History (Kenya, Africa, world)
- Business & economics (how markets work, inflation, budgeting)
- Agriculture (important in Kenya — crops, soil, seasons)
- Technology (how phones, internet, apps work)

**Success criteria**:
- Can explain what malaria is and how to prevent it
- Explains inflation in simple terms
- Describes how the internet works

---

## Phase 6 — Code & Structured Output (BUILD NEXT)
**Checkpoint target**: `p6-checkpoint-00001500`  
**Steps**: 1500 · lr=4e-6 · from p5  
**Data**: ~2500 code examples  
**Training focus**:
- Python (variables, loops, functions, lists, file I/O)
- JavaScript basics
- HTML/CSS basics
- JSON output formatting
- Code explanation ("What does this code do?")
- Bug finding ("What is wrong with this code?")

**Success criteria**:
- Writes a working Python function from description
- Outputs valid JSON when asked
- Explains code correctly

---

## Phase 7 — DPO3 Alignment (BUILD NEXT)
**Checkpoint target**: `p7-checkpoint-00001500`  
**Steps**: 1500 · lr=8e-7 · from p6  
**Data**: ~4000 preference pairs  
**Training focus**:
- Prefer honest "I don't know" over hallucinated wrong answers
- Prefer concise direct answers over padded verbose ones
- Prefer admitting limitations over overconfident claims
- Prefer helpful clarifying questions over guessing
- Refuse harmful requests politely but firmly

**Success criteria**:
- Says "I don't know" when genuinely uncertain
- Doesn't generate dangerous content
- Answers are appropriately concise

---

## Phase 8 — Conversational Persona (BUILD NEXT)
**Checkpoint target**: `p8-checkpoint-00001500`  
**Steps**: 1500 · lr=3e-6 · from p7  
**Data**: ~3000 personality & creative examples  
**Training focus**:
- Warm, encouraging tone
- Simple jokes (appropriate, not offensive)
- Storytelling (short, engaging narratives)
- Empathy ("That sounds difficult, here's how I can help")
- Motivational responses
- Creative tasks (write a poem, suggest a name)
- Graceful topic changes

**Success criteria**:
- Responses feel warm, not robotic
- Can write a short poem on request
- Consistent personality across session

---

## Benchmark Expansion

| Phase | New benchmark categories added |
|-------|-------------------------------|
| P0    | 87 questions (current) |
| P2    | + Multi-turn (10 conversations × 3 turns) |
| P3    | + Reasoning with working (15 problems) |
| P4    | + Swahili conversation (20 turns) |
| P5    | + Knowledge explanation (20 questions) |
| P6    | + Code generation (10 problems) |
| P7    | + Alignment (10 tricky prompts) |
| P8    | + Creativity (5 tasks) |
| **Total** | **~200+ test cases** |

---

## Checkpoint Naming Convention

```
p1-checkpoint-00000500   ← Kenya/Swahili injection (500 steps)
p2-checkpoint-00002000   ← Instruction following (2000 steps)
p3-checkpoint-00002000   ← Chain-of-thought (2000 steps)
p4-checkpoint-00002000   ← Swahili fluency (2000 steps)
p5-checkpoint-00002000   ← Knowledge depth (2000 steps)
p6-checkpoint-00001500   ← Code (1500 steps)
p7-checkpoint-00001500   ← DPO alignment (1500 steps)
p8-checkpoint-00001500   ← Persona (1500 steps)
```

Total new training: **~14000 steps** = 14 Kaggle runs

---

## The Core Insight

At 125M parameters, Yaya cannot memorize the world. The strategy is:
- **Guards** handle deterministic facts (arithmetic, datetime, identity)
- **Training** handles patterns, reasoning, tone, and language fluency
- **Phases** are curriculum: simple → complex, direct → nuanced

A 125M model trained on the right data, in the right order, with the right guards can  
deliver 90%+ of the experience of a much larger model for the tasks users actually care about.

**Yaya's unique edge**: Swahili fluency + Kenya knowledge + tool use in a fast, local model.
