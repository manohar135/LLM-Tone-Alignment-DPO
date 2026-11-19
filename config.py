"""Central configuration for Tone Aligner (Single-GPU L4 QLoRA DPO)."""

# Base model: SFT-only checkpoint (knows chat format, NOT yet DPO/RLHF aligned)
# Using an SFT-only model follows the real industry pipeline (Pretrain -> SFT -> DPO)
# and yields clear, measurable before-vs-after alignment gains.
MODEL_ID = "alignment-handbook/zephyr-7b-sft-full"

# Balanced Preference Datasets (50% Safety + 50% Tone & Helpfulness)
PKU_DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"
UF_DATASET_ID = "argilla/ultrafeedback-binarized-preferences-cleaned"
PKU_RATIO = 0.5

# Full run (~1.5-2 hrs on 1x L4 GPU, ~3-4 CCUs)
NUM_TRAIN_SAMPLES = 8000
NUM_EVAL_SAMPLES = 400

# Fast sanity run (~15-20 min on 1x L4 GPU, ~0.6 CCU) via `--sanity`
SANITY_TRAIN_SAMPLES = 1600
SANITY_EVAL_SAMPLES = 200

# 4-bit QLoRA quantization (fits 7B comfortably on a single 22.5 GB L4 GPU)
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = True

# LoRA adapters on all attention & MLP linear projections (~0.6% trainable params)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# DPO Training (effective batch size = 2 * 16 = 32 -> 250 steps for 1 full epoch of 8K)
DPO_BETA = 0.1
LEARNING_RATE = 3e-5
PER_DEVICE_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 1.0
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512

LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50

GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_8bit"
SEED = 42

# Output paths
OUTPUT_DIR = "./outputs/dpo_adapter"
DATASET_CACHE_DIR = "./outputs/dataset"
TRAINING_METRICS_PATH = "./outputs/training_metrics.json"
EVAL_RESULTS_PATH = "./outputs/eval_results.json"

# Evaluation models
TOXICITY_MODEL_ID = "s-nlp/roberta_toxicity_classifier"
REWARD_MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"

# Generation defaults
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
