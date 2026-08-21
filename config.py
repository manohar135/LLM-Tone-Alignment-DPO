"""
Tone Aligner — Central Configuration

All hyperparameters, model IDs, and paths live here.
Adjust values based on your hardware (tested on RTX 3050 8GB VRAM).
"""

# ── Model ─────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # Already instruction-tuned, 0.5B fits 8GB VRAM

# ── Dataset ───────────────────────────────────────────
DATASET_ID = "Anthropic/hh-rlhf"
DATASET_SUBSET = "harmless-base"     # Harmful→safe pairs give the clearest alignment signal
NUM_TRAIN_SAMPLES = 2000             # Trains in ~30-45 min on RTX 3050
NUM_EVAL_SAMPLES = 200               # Held-out for validation loss

# ── 4-bit Quantization (QLoRA) ────────────────────────
LOAD_IN_4BIT = True                  # Required for 8GB VRAM
BNB_4BIT_QUANT_TYPE = "nf4"          # NormalFloat4 — optimal for pretrained LLM weights
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # BF16 compute (RTX 3050 is Ampere, supports BF16)
BNB_4BIT_USE_DOUBLE_QUANT = True     # Quantize the quantization constants too — saves ~0.4GB

# ── LoRA ──────────────────────────────────────────────
LORA_R = 16                          # Rank — moderate capacity without overfitting
LORA_ALPHA = 32                      # Scaling factor — standard 2× rank
LORA_DROPOUT = 0.05                  # Light regularization
LORA_TARGET_MODULES = [              # Only attention projections — steers what model attends to
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# ── DPO Training ──────────────────────────────────────
DPO_BETA = 0.1                       # KL divergence penalty — controls how far policy drifts from reference
LEARNING_RATE = 5e-6                  # Conservative — DPO is sensitive to LR
PER_DEVICE_BATCH_SIZE = 1            # Minimum for 8GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8      # Effective batch size = 8
NUM_EPOCHS = 1                       # Single pass — DPO overfits quickly on small datasets
MAX_STEPS = 250                      # Hard cap to keep training under 1 hour
WARMUP_STEPS = 25                    # 10% warmup steps
MAX_LENGTH = 512                     # Max total sequence length (prompt + response)
LOGGING_STEPS = 10                   # Log metrics every 10 steps
SAVE_STEPS = 50                      # Checkpoint every 50 steps

# ── Memory Optimization ──────────────────────────────
GRADIENT_CHECKPOINTING = True        # Recompute activations in backward pass — saves ~40% VRAM
PRECOMPUTE_REF_LOG_PROBS = True      # Precompute reference logprobs, avoid extra forward pass during training
OPTIM = "paged_adamw_8bit"           # 8-bit optimizer from bitsandbytes — saves optimizer memory

# ── Paths ─────────────────────────────────────────────
OUTPUT_DIR = "./outputs/dpo_adapter"
DATASET_CACHE_DIR = "./outputs/dataset"
EVAL_RESULTS_PATH = "./outputs/eval_results.json"

# ── Evaluation ────────────────────────────────────────
TOXICITY_MODEL_ID = "s-nlp/roberta_toxicity_classifier"  # Binary toxic/neutral classifier

# ── Generation ────────────────────────────────────────
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
