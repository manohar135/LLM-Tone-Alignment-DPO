"""Single-GPU DPO training with 4-bit QLoRA on Zephyr-7B-SFT."""

import argparse
import dataclasses
import inspect
import json
import os
import time

import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from config import (
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    DATASET_CACHE_DIR,
    DPO_BETA,
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    LEARNING_RATE,
    LOAD_IN_4BIT,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_LENGTH,
    MAX_PROMPT_LENGTH,
    MODEL_ID,
    NUM_EPOCHS,
    OPTIM,
    OUTPUT_DIR,
    PER_DEVICE_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    SAVE_STEPS,
    SEED,
    TRAINING_METRICS_PATH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def load_model_and_tokenizer():
    """Load 4-bit quantized SFT base model and configure Mistral/Zephyr tokenizer."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU (e.g., NVIDIA L4) is required for training.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16_ok = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, bf16={bf16_ok})")
    print(f"Loading {MODEL_ID} (4-bit NF4 QLoRA, compute_dtype={compute_dtype})...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # Use PyTorch native SDPA (Flash/Memory-Efficient Attention) for ~2x speedup
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    # Mistral/Zephyr tokenizers often lack a dedicated pad token (or map it to <unk>);
    # aligning pad_token -> eos_token prevents corrupted attention masks.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.unk_token_id:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, {"gpu_name": gpu_name, "vram_gb": round(vram_gb, 2), "bf16": bf16_ok}


def build_dpo_config(bf16_ok, sanity=False):
    """Build DPOConfig cleanly while remaining compatible across TRL versions."""
    dpo_fields = {f.name for f in dataclasses.fields(DPOConfig)}

    eval_steps = 15 if sanity else EVAL_STEPS
    save_steps = 15 if sanity else SAVE_STEPS
    log_steps = 5 if sanity else LOGGING_STEPS

    # Keep per-device batch=1 on smaller (<18 GB) GPUs like T4 while preserving eff batch=32
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    batch_size = 1 if vram_gb < 18.0 else PER_DEVICE_BATCH_SIZE
    grad_accum = 32 if vram_gb < 18.0 else GRADIENT_ACCUMULATION_STEPS

    kwargs = {
        "output_dir": OUTPUT_DIR,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size if vram_gb < 18.0 else PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optim": OPTIM,
        "num_train_epochs": NUM_EPOCHS,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "beta": DPO_BETA,
        "max_length": MAX_LENGTH,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Native bf16 on L4/A100; on T4 (no bf16) leave fp16=False to avoid GradScaler instability
        "bf16": bf16_ok,
        "fp16": False,
        "logging_steps": log_steps,
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": SEED,
    }

    if "eval_strategy" not in dpo_fields and "evaluation_strategy" in dpo_fields:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    # Drop any keys not supported by the installed TRL version
    return DPOConfig(**{k: v for k, v in kwargs.items() if k in dpo_fields})


def save_training_curves(log_history, plot_path):
    """Plot Loss, Preference Accuracy, and Reward Margin over training steps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _extract(key):
        pts = [(r["step"], r[key]) for r in log_history if "step" in r and key in r]
        return ([p[0] for p in pts], [p[1] for p in pts])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, tr_key, ev_key, title, ref_y, ref_label in [
        (axes[0], "loss", "eval_loss", "DPO Loss", 0.6931, "Random (ln 2)"),
        (axes[1], "rewards/accuracies", "eval_rewards/accuracies", "Preference Accuracy", 0.5, "Chance (0.5)"),
        (axes[2], "rewards/margins", "eval_rewards/margins", "Reward Margin (Chosen - Rejected)", 0.0, "Zero Margin"),
    ]:
        tx, ty = _extract(tr_key)
        ex, ey = _extract(ev_key)
        if tx:
            ax.plot(tx, ty, label="Train", linewidth=2)
        if ex:
            ax.plot(ex, ey, label="Eval", marker="o", linewidth=2)
        ax.axhline(ref_y, color="gray", linestyle="--", alpha=0.6, label=ref_label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[1].set_ylim(0.0, 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {plot_path}")


def train(sanity=False, resume=False):
    """Run DPO training, evaluate on held-out split, and save best LoRA adapter."""
    print(f"Loading preprocessed dataset from: {DATASET_CACHE_DIR}")
    dataset = load_from_disk(DATASET_CACHE_DIR)
    train_ds, eval_ds = dataset["train"], dataset["eval"]
    print(f"  Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    model, tokenizer, hw = load_model_and_tokenizer()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    training_args = build_dpo_config(hw["bf16"], sanity=sanity)

    # Note: Passing ref_model=None with peft_config tells DPOTrainer to compute
    # reference log-probs by temporarily disabling the LoRA adapter — no second
    # 7B reference model is loaded into VRAM!
    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    tok_arg = "processing_class" if "processing_class" in trainer_sig else "tokenizer"
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        **{tok_arg: tokenizer},
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    eff_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    print(f"\nStarting DPO training on {MODEL_ID}...")
    print(f"  Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M ({100 * trainable / total:.2f}%)")
    print(f"  Effective batch:  {eff_batch} | LR: {LEARNING_RATE} | Beta: {DPO_BETA} | Epochs: {NUM_EPOCHS}")

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume)
    elapsed_min = (time.time() - t0) / 60.0

    print("\nRunning final evaluation on held-out eval split...")
    final_eval = trainer.evaluate()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = {
        "model_id": MODEL_ID,
        "gpu": hw["gpu_name"],
        "vram_gb": hw["vram_gb"],
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_minutes": round(elapsed_min, 2),
        "train_metrics": train_result.metrics,
        "final_eval_metrics": final_eval,
        "log_history": trainer.state.log_history,
    }
    os.makedirs(os.path.dirname(TRAINING_METRICS_PATH), exist_ok=True)
    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_training_curves(
        trainer.state.log_history,
        os.path.join(os.path.dirname(TRAINING_METRICS_PATH), "training_curves.png"),
    )

    print("\n" + "=" * 68)
    print("  TRAINING COMPLETE")
    print("=" * 68)
    print(f"  Wall time:             {elapsed_min:.1f} min")
    print(f"  Final Eval Loss:       {final_eval.get('eval_loss', float('nan')):.4f}")
    if "eval_rewards/accuracies" in final_eval:
        print(f"  Eval Preference Acc:   {final_eval['eval_rewards/accuracies'] * 100:.1f}%")
    if "eval_rewards/margins" in final_eval:
        print(f"  Eval Reward Margin:    {final_eval['eval_rewards/margins']:+.4f}")
    print(f"  Adapter saved to:      {OUTPUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO adapter with 4-bit QLoRA.")
    parser.add_argument("--sanity", action="store_true", help="Use faster eval/save cadence for sanity run")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(sanity=args.sanity, resume=args.resume)
