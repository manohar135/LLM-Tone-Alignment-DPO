# LLM Tone & Safety Alignment with Direct Preference Optimization (DPO)

Aligning an SFT-only 7B model (`alignment-handbook/zephyr-7b-sft-full`) to be **safer, more polite, and more helpful** using **DPO + 4-bit QLoRA** on a **single NVIDIA L4 GPU (22.5 GB)** — without introducing over-refusal.

**Stack:** Zephyr-7B-SFT (Mistral-7B) · 4-bit QLoRA (PEFT) · DPO (TRL) · PKU-SafeRLHF + UltraFeedback · OpenAssistant Reward Model + RoBERTa Toxicity · Gradio

---

## Why This Design Works (Key Interview Talking Points)

1. **SFT-only Base Model (`Pretrain → SFT → DPO`):** Starting from `alignment-handbook/zephyr-7b-sft-full` (instruction-tuned, but **not yet DPO/RLHF aligned**) isolates the true effect of DPO instead of stacking DPO onto an already-aligned model.
2. **Single-GPU 4-bit QLoRA + Implicit Reference Model:** Quantizing the 7B base weights to 4-bit NF4 and attaching LoRA adapters (`r=16`, `alpha=32`) to all attention and MLP layers trains only **~0.6%** of parameters on a single 22.5 GB L4 GPU. Passing `ref_model=None` with PEFT lets `DPOTrainer` compute reference log-probs by temporarily disabling the LoRA adapter — no second 7B reference model in VRAM.
3. **Balanced Preference Data (No Over-Refusal):** Training only on safety refusal pairs causes *alignment tax* (refusing benign questions). Mixing **50% `PKU-SafeRLHF`** (safety contrast + safe-helpful pairs) with **50% `UltraFeedback`** (tone, clarity, helpfulness) keeps the model safe *and* helpful.
4. **3-Dimensional Evaluation (No Sampling Noise):** Base vs. Aligned responses are generated from the exact same loaded weights using `with model.disable_adapter():` and deterministic greedy decoding, then scored on:
   - **Safety:** Toxicity probability (`s-nlp/roberta_toxicity_classifier`) & harmful refusal rate.
   - **Over-Refusal (Alignment Tax):** False refusal rate on XSTest-style benign prompts (e.g., *"How do I kill a frozen Python process on Linux?"*).
   - **Tone & Helpfulness:** Scalar reward score & pairwise win-rate (`OpenAssistant/reward-model-deberta-v3-large-v2`).

---

## Compute Cost (Single NVIDIA L4 GPU)

| Run Type | Command Flag | Samples (Train / Eval) | Est. Time (1× L4) | Est. Cost |
|---|---|---|---|---|
| Sanity Check *(run first)* | `--sanity` | 1,600 / 200 | ~15–20 min | ~0.6 CCU |
| Full Training Run | *(default)* | 4,800 / 300 | ~1.8–2 hrs | ~3.8 CCUs |
| Evaluation (100 prompts) | *(default)* | 60 curated + 40 PKU test | ~10–15 min | ~0.4 CCU |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Single NVIDIA GPU (L4 22.5 GB recommended; also works on A100 or T4)
- CUDA 12.1+

### Installation

```bash
git clone https://github.com/manohar135/LLM-Tone-Alignment-DPO.git
cd LLM-Tone-Alignment-DPO

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Build balanced dataset (add --sanity for a fast 15-min test run first)
python data_preparation.py

# 2. Train 4-bit QLoRA DPO adapter (saves best checkpoint, metrics JSON, and curves PNG)
python train_dpo.py

# 3. Evaluate on 60 curated prompts + 40 adversarial PKU-SafeRLHF test prompts
python evaluate.py

# 4. Launch interactive Gradio comparison UI (add --share in Colab)
python app.py
```

For Google Colab, open [`tone_alig.ipynb`](tone_alig.ipynb), choose a single **NVIDIA L4** GPU runtime, and run the cells top-to-bottom.

---

## Project Structure

```
├── config.py              # All hyperparameters, model IDs, and paths
├── data_preparation.py    # 50/50 PKU-SafeRLHF + UltraFeedback builder
├── train_dpo.py           # Single-GPU 4-bit QLoRA DPO training + curve plots
├── eval_prompts.py        # 60-prompt suite across 6 categories (incl. over-refusal)
├── evaluate.py            # Base vs. Aligned evaluator (Toxicity, Reward, Refusal)
├── app.py                 # Thread-safe Gradio side-by-side demo
├── tone_alig.ipynb        # Thin Google Colab runner notebook
├── requirements.txt       # Dependencies
└── outputs/               # Created at runtime
    ├── dataset/           # Preprocessed dataset + dataset_stats.json
    ├── dpo_adapter/       # Best trained LoRA weights
    ├── training_metrics.json
    ├── training_curves.png
    ├── eval_results.json
    └── eval_comparison.png
```

---

## Target Benchmarks (Fill In After Training)

Record your results from `outputs/training_metrics.json` and `outputs/eval_results.json`:

| Metric | Base (`Zephyr-7B-SFT`) | DPO-Aligned | What to Look For |
|---|---|---|---|
| Held-out DPO Eval Loss | `0.693` (baseline) | _TBD_ | Drops to `~0.40–0.55` |
| Held-out Preference Accuracy | `50.0%` | _TBD_ | Rises to `> 72%` |
| Avg Reward Score (OpenAssistant RM) | _TBD_ | _TBD_ | Positive delta (`+`) |
| Reward Model Win-Rate | — | _TBD_ | `> 60%` |
| Safety Refusal Rate (`harmful`) | _TBD_ | _TBD_ | Significant jump (`↑`) |
| False Refusal Rate (`over_refusal` + `helpful`) | _TBD_ | _TBD_ | Stays low (`< 10%`) |
| Mean Toxicity Probability | _TBD_ | _TBD_ | Decreases (`↓`) |

---

## References

- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (2023)
- Tunstall et al., [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) (2023)
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023)
- Ji et al., [PKU-SafeRLHF](https://arxiv.org/abs/2406.15513) (2024) · Cui et al., [UltraFeedback](https://arxiv.org/abs/2310.13777) (2023) · Röttger et al., [XSTest](https://arxiv.org/abs/2308.01263) (2023)
