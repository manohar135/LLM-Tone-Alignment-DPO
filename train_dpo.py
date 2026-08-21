"""
Tone Aligner — DPO Training

Loads the base instruction model with 4-bit quantization (QLoRA),
configures LoRA adapters on attention layers, and runs Direct Preference
Optimization using TRL's DPOTrainer.

Key memory optimizations for RTX 3050 (8GB VRAM):
  - 4-bit NF4 quantization (~0.4GB for model weights)
  - Gradient checkpointing (recompute activations, save ~40% VRAM)
  - Precomputed reference log-probs (no reference model in VRAM during training)
  - Paged AdamW 8-bit optimizer (smaller optimizer states)
  - Batch size 1 with gradient accumulation

Usage:
    python train_dpo.py
"""

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from config import (
    MODEL_ID, DATASET_CACHE_DIR, OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_USE_DOUBLE_QUANT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    DPO_BETA, LEARNING_RATE, PER_DEVICE_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    MAX_STEPS, WARMUP_STEPS, MAX_LENGTH,
    LOGGING_STEPS, SAVE_STEPS,
    GRADIENT_CHECKPOINTING, PRECOMPUTE_REF_LOG_PROBS, OPTIM,
)


def load_model_and_tokenizer():
    """Load the base model with 4-bit quantization and its tokenizer."""
    print(f"Loading model: {MODEL_ID}")

    # 4-bit quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",  # Stable with gradient checkpointing
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    )
    # Disable KV cache — incompatible with gradient checkpointing
    model.config.use_cache = False

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"  # Left-pad for causal LM generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params / 1e6:.1f}M")
    print(f"  Quantization: {'4-bit NF4' if LOAD_IN_4BIT else 'None'}")
    print(f"  Device: {model.device}")

    return model, tokenizer


def get_lora_config():
    """Configure LoRA: which layers to adapt and how.

    We target attention projections (q, k, v, o) because they control
    WHAT the model attends to — key for steering tone and style.
    MLP layers are left frozen to keep trainable params very small.
    """
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",  # No bias training — fewer parameters
    )
    print(f"\nLoRA config:")
    print(f"  Rank: {LORA_R}, Alpha: {LORA_ALPHA}, Dropout: {LORA_DROPOUT}")
    print(f"  Target modules: {LORA_TARGET_MODULES}")
    return config


def get_training_args():
    """Configure DPO training arguments.

    Key DPO-specific settings:
      - beta: Controls how much the policy can diverge from the reference model.
              Higher beta = more conservative updates (stays closer to reference).
      - precompute_ref_log_probs: Compute reference model log-probs once before
              training, then unload the reference. Critical for saving VRAM.
    """
    args = DPOConfig(
        output_dir=OUTPUT_DIR,

        # Batch size — kept at 1 for 8GB VRAM, compensated by gradient accumulation
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Optimizer
        learning_rate=LEARNING_RATE,
        optim=OPTIM,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="cosine",

        # DPO-specific
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        precompute_ref_log_probs=PRECOMPUTE_REF_LOG_PROBS,

        # Memory optimization
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=True,

        # Logging and saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",  # No wandb/tensorboard — keep it simple

        # Misc
        remove_unused_columns=False,
        seed=42,
    )
    return args


def train():
    """Main training pipeline."""
    # Load preprocessed dataset
    print("Loading preprocessed dataset...")
    dataset = load_from_disk(DATASET_CACHE_DIR)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Configure LoRA
    peft_config = get_lora_config()
    from peft import get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)

    # Ensure all trainable parameters are in float32 for AMP GradScaler safety
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    # Configure training
    training_args = get_training_args()

    # Initialize DPO trainer
    print("\nInitializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PEFT mode: base weights = reference model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Print trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60)
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  DPO beta: {DPO_BETA}")
    print()

    trainer.train()

    # Save the trained LoRA adapter
    print(f"\nSaving adapter to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Adapter size: ~{sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e6:.1f}MB")


if __name__ == "__main__":
    train()
