# 🎯 Tone Aligner — LLM Alignment with DPO

A complete, beginner-friendly implementation of **Direct Preference Optimization (DPO)** to align an instruction-tuned language model toward polite, helpful, and safe responses.

**Stack:** Qwen2.5-0.5B-Instruct · LoRA/PEFT · DPO (TRL) · Anthropic HH-RLHF · Gradio

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
  - [What is DPO?](#what-is-dpo)
  - [What is LoRA?](#what-is-lora)
  - [What is a Preference Dataset?](#what-is-a-preference-dataset)
- [Dataset](#dataset)
- [Model Choice](#model-choice)
- [Training Details](#training-details)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Gradio Demo](#gradio-demo)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Modern LLMs, even after instruction fine-tuning (SFT), can still produce toxic, unhelpful, or harmful responses. **Alignment** is the process of further training the model to prefer safe, helpful outputs.

This project implements the full alignment pipeline:

1. **Load a preference dataset** (Anthropic HH-RLHF) with human-judged "chosen" vs "rejected" response pairs
2. **Apply LoRA** to make training efficient (~0.5% of parameters)
3. **Train with DPO** to shift the model's probability distribution toward preferred responses
4. **Evaluate** with automated toxicity scoring to quantify improvement
5. **Demo** with an interactive Gradio UI for side-by-side comparison

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
│                                                                 │
│  Anthropic HH-RLHF ──→ Parse & Format ──→ DPO Dataset          │
│  (harmless-base)        (prompt/chosen/    (2000 train +        │
│                          rejected)          200 eval)           │
│                                                                 │
│  Qwen2.5-0.5B-Instruct ──→ 4-bit Quantized ──→ + LoRA Adapter  │
│  (frozen base weights)     (NF4, ~0.4GB)       (r=16, ~2MB)    │
│                                                                 │
│                    ┌──────────────────┐                         │
│                    │   DPO Trainer    │                         │
│                    │                  │                         │
│  Reference Model ──│  L_DPO = -log σ( │──→ Trained LoRA        │
│  (base, frozen)    │    β · Δ logprob │    Adapter (~2-5MB)     │
│                    │  )               │                         │
│                    └──────────────────┘                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      INFERENCE                                  │
│                                                                 │
│  User Prompt ──→ Qwen + LoRA ──→ Aligned Response               │
│                  (adapter ON)                                    │
│                                                                 │
│  User Prompt ──→ Qwen only   ──→ Base Response                  │
│                  (adapter OFF)                                   │
│                                                                 │
│  Both responses ──→ Toxicity Classifier ──→ Toxicity Scores     │
│                    (roberta-toxicity)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### What is DPO?

**Direct Preference Optimization** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) is an alignment technique that trains a language model directly on preference data without needing a separate reward model.

**The core idea:** Given a prompt and two responses (one preferred, one rejected), adjust the model's weights so it assigns higher probability to the preferred response relative to a frozen reference model.

**The loss function:**

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Intuition:**
- `π_θ` = the policy model (what we're training)
- `π_ref` = the reference model (frozen copy of the original)
- `y_w` = the preferred ("winner") response
- `y_l` = the dispreferred ("loser") response
- `β` = temperature controlling how far the policy can drift from the reference
- The loss pushes the model to increase the log-probability ratio for preferred responses while decreasing it for rejected ones

**Why DPO over PPO/RLHF?**
| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Reward model | Requires separate training | Not needed (implicit) |
| Training stability | Sensitive, requires careful tuning | More stable |
| Memory | Needs policy + reward + value models | Needs policy + reference only |
| Implementation | Complex (PPO loop) | Simple (single loss function) |

### What is LoRA?

**Low-Rank Adaptation** ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) is a parameter-efficient fine-tuning method. Instead of updating all model weights, LoRA adds small trainable "adapter" matrices to specific layers.

**How it works:**
```
Original: h = W · x           (W is frozen, no gradient)
With LoRA: h = W · x + B·A·x  (A and B are tiny trainable matrices)
```

Where `A` has shape `(d, r)` and `B` has shape `(r, d)`, with `r << d`.

**In this project:**
- `r = 16` (rank) — controls adapter capacity
- Target: attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Trainable params: ~0.5% of total model parameters
- Adapter size: ~2-5MB (vs ~1GB for full model)

### What is a Preference Dataset?

A preference dataset contains pairs of responses to the same prompt, where human annotators judged which response is better:

```
Prompt:   "How can I hack into someone's account?"
Chosen:   "I can't help with that. Unauthorized access is illegal..."
Rejected: "Here's how you can try to hack into an account..."
```

The "chosen" response is the one humans preferred (safer, more helpful, more polite). The model learns to increase the probability of responses like "chosen" and decrease the probability of responses like "rejected".

---

## Dataset

**Source:** [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) (`harmless-base` subset)

This dataset contains conversations where human annotators chose the response that was less harmful and more helpful.

**Preprocessing pipeline:**
1. Download the `harmless-base` split (~8K examples)
2. Parse the `\n\nHuman: ...\n\nAssistant: ...` format into structured messages
3. Split on the last `\n\nAssistant:` marker to separate prompt from final response
4. Filter out: identical chosen/rejected pairs, responses < 10 characters
5. Randomly sample 2,000 training + 200 evaluation examples

**Why `harmless-base`?**
- The harmful→safe contrast produces the clearest alignment signal
- Before/after differences are visually obvious in the demo
- Smaller subset = faster iteration for a portfolio project

---

## Model Choice

**Selected:** `Qwen/Qwen2.5-0.5B-Instruct`

| Criterion | Why Qwen2.5-0.5B-Instruct |
|-----------|--------------------------|
| Size | 0.5B parameters — fits comfortably in 8GB VRAM with 4-bit quantization |
| Quality | Already instruction-tuned, produces coherent conversational responses |
| Chat template | Built-in ChatML template — works seamlessly with TRL's DPO trainer |
| Community | Well-supported by HuggingFace ecosystem |

**Trade-off:** Smaller models have limited reasoning ability. The alignment improvements are most visible on safety/toxicity dimensions rather than complex reasoning tasks.

---

## Training Details

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Quantization | 4-bit NF4 (QLoRA) | Fits 8GB VRAM |
| LoRA rank | 16 | Moderate capacity without overfitting |
| LoRA alpha | 32 | Standard 2× rank scaling |
| LoRA targets | q, k, v, o projections | Steers attention — key for tone alignment |
| DPO beta | 0.1 | Standard KL penalty |
| Learning rate | 5e-6 | Conservative — DPO is LR-sensitive |
| Batch size | 1 (effective 8) | Memory-constrained, gradient accumulation compensates |
| Max steps | 250 | ~30-45 min training, avoids overfitting |
| Optimizer | Paged AdamW 8-bit | Reduced optimizer memory footprint |
| Gradient checkpointing | Enabled | Saves ~40% activation memory |
| Ref model strategy | Precomputed log-probs | No reference model in VRAM during training |

**Estimated VRAM usage during training:** ~5-5.5 GB / 8 GB

---

## Evaluation Methodology

### Qualitative: Side-by-Side Comparison
Generate responses from both the base model (LoRA adapter disabled) and the aligned model (adapter enabled) on 12 curated prompts across 5 categories.

### Quantitative: Toxicity Scoring
Each response is scored by `s-nlp/roberta_toxicity_classifier`, a RoBERTa model fine-tuned for binary toxicity detection. We report:
- **Per-response toxicity score** (0 = neutral, 1 = toxic)
- **Average toxicity** for base vs. aligned model
- **Percentage reduction** in average toxicity
- **Per-category breakdown** (harmful, sensitive, hostile, helpful, controversial)

### What We're Measuring
| Category | Base Model Expectation | Aligned Model Expectation |
|----------|----------------------|--------------------------|
| Harmful requests | May comply or give partial info | Should refuse politely |
| Sensitive topics | May give generic or cold response | Should show empathy |
| Hostile users | May mirror hostility | Should stay professional |
| Helpful requests | Should be helpful | Should remain helpful (no regression) |
| Controversial | May take sides | Should be balanced |

---

## Results

> **⚠️ Run `python evaluate.py` and paste your actual results below.**
> The results below are placeholders — replace them with your real numbers.

### Toxicity Scores

```
  Average Toxicity (Base Model):     [PASTE YOUR RESULT]
  Average Toxicity (Aligned Model):  [PASTE YOUR RESULT]
  Toxicity Reduction:                [PASTE YOUR RESULT]%
```

### Per-Category Breakdown

```
  Category          Base Toxicity  Aligned Toxicity     Change
  ────────────────────────────────────────────────────────────
  [PASTE YOUR RESULTS FROM evaluate.py OUTPUT]
```

### Sample Comparisons

```
  [PASTE 2-3 INTERESTING BEFORE/AFTER EXAMPLES FROM evaluate.py OUTPUT]
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 3050)
- CUDA 12.1+

### Installation

```bash
# Clone/navigate to the project
cd RLHF_finetuning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support (adjust cu121 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Step 1: Download and preprocess the dataset
python data_preparation.py

# Step 2: Train with DPO (~30-45 min on RTX 3050)
python train_dpo.py

# Step 3: Evaluate and score toxicity
python evaluate.py

# Step 4: Launch the Gradio demo
python app.py
```

### Verify CUDA Setup

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

---

## Project Structure

```
RLHF_finetuning/
├── README.md              ← This file
├── requirements.txt       ← Pinned dependencies
├── config.py              ← All hyperparameters (single source of truth)
├── eval_prompts.py        ← Curated evaluation prompts by category
├── data_preparation.py    ← Download, parse, filter Anthropic HH-RLHF
├── train_dpo.py           ← QLoRA setup + DPO training with TRL
├── evaluate.py            ← Side-by-side comparison + toxicity scoring
├── app.py                 ← Gradio demo for interactive comparison
└── outputs/               ← Created at runtime
    ├── dataset/           ← Preprocessed HF dataset on disk
    ├── dpo_adapter/       ← Trained LoRA weights (~2-5MB)
    └── eval_results.json  ← Evaluation results
```

---

## Gradio Demo

After training, launch the interactive demo:

```bash
python app.py
```

Opens at `http://localhost:7860` with:
- **Text input** for any prompt
- **Side-by-side panels** showing base vs. aligned responses
- **Real-time toxicity scores** for both responses
- **Preset examples** to quickly test different categories
- **Adjustable** temperature and max token length

---

## Limitations

1. **Small model** — 0.5B parameters limits reasoning depth. Alignment improvements are most visible on safety/toxicity, less on nuanced judgment.

2. **Small training set** — 2,000 examples is sufficient for demonstrating the technique but not for production-quality alignment. Industrial systems use 50K-500K+ preference pairs.

3. **Single dimension** — We only train on harmless/harmful pairs. Production alignment also includes helpfulness, honesty, and instruction-following dimensions.

4. **No iterative refinement** — Production systems run multiple rounds of DPO/RLHF with fresh human feedback. This is a single-round demonstration.

5. **Toxicity classifier limitations** — The automated toxicity scorer has its own biases and may not capture all forms of harmful content (e.g., subtly misleading information).

6. **QLoRA approximation** — 4-bit quantization introduces some gradient noise. Full-precision training would give slightly better results but requires more VRAM.

---

## References

- **DPO Paper:** Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290))
- **LoRA Paper:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685))
- **QLoRA Paper:** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized Language Models" ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
- **Anthropic HH-RLHF:** Bai et al., "Training a Helpful and Harmless Assistant" ([arXiv:2204.05862](https://arxiv.org/abs/2204.05862))
- **TRL Library:** [HuggingFace TRL](https://github.com/huggingface/trl)
- **PEFT Library:** [HuggingFace PEFT](https://github.com/huggingface/peft)
- **Qwen2.5:** [Qwen Team](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

---

## License

This project is for educational and portfolio purposes. The model weights, dataset, and libraries each have their own licenses — refer to their respective repositories.
