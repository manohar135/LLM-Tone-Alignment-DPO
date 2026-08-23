"""DPO training with QLoRA on the preprocessed preference dataset."""

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
        attn_implementation="eager",
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer


def get_lora_config():
    """Configure LoRA adapters for attention and MLP projections."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )


def get_training_args():
    """Configure DPO training arguments."""
    return DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        optim=OPTIM,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="cosine",
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        precompute_ref_log_probs=PRECOMPUTE_REF_LOG_PROBS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,
        bf16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )


def train():
    """Main training pipeline."""
    print("Loading preprocessed dataset...")
    dataset = load_from_disk(DATASET_CACHE_DIR)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    model, tokenizer = load_model_and_tokenizer()

    peft_config = get_lora_config()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)

    # Cast trainable params to float32 for gradient stability
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    training_args = get_training_args()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    print(f"\nStarting DPO training...")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Max steps: {MAX_STEPS}, LR: {LEARNING_RATE}, Beta: {DPO_BETA}")

    trainer.train()

    print(f"\nSaving adapter to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    adapter_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print(f"Training complete! Adapter size: ~{adapter_size / 1e6:.1f}MB")


if __name__ == "__main__":
    train()
