"""Central configuration for all hyperparameters, model IDs, and paths."""

# Model
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Dataset
DATASET_ID = "Anthropic/hh-rlhf"
DATASET_SUBSET = "harmless-base"
NUM_TRAIN_SAMPLES = 10000
NUM_EVAL_SAMPLES = 200

# 4-bit quantization (QLoRA)
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_USE_DOUBLE_QUANT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# DPO training
DPO_BETA = 0.1
LEARNING_RATE = 5e-6
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 1
MAX_STEPS = 250
WARMUP_STEPS = 25
MAX_LENGTH = 512
LOGGING_STEPS = 10
SAVE_STEPS = 50

# Memory optimization
GRADIENT_CHECKPOINTING = True
PRECOMPUTE_REF_LOG_PROBS = True
OPTIM = "paged_adamw_8bit"

# Paths
OUTPUT_DIR = "./outputs/dpo_adapter"
DATASET_CACHE_DIR = "./outputs/dataset"
EVAL_RESULTS_PATH = "./outputs/eval_results.json"

# Evaluation
TOXICITY_MODEL_ID = "s-nlp/roberta_toxicity_classifier"

# Generation
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
