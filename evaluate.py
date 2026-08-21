"""
Tone Aligner — Evaluation

Generates responses from both the base model and the DPO-aligned model
on curated evaluation prompts, then scores them using a toxicity classifier.

Outputs:
  1. Side-by-side comparison table (printed to terminal)
  2. Toxicity scores per response
  3. Summary metrics (avg toxicity, % reduction)
  4. JSON results saved to outputs/eval_results.json

Usage:
    python evaluate.py
"""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

from config import (
    MODEL_ID, OUTPUT_DIR, EVAL_RESULTS_PATH, TOXICITY_MODEL_ID,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_USE_DOUBLE_QUANT,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
)
from eval_prompts import EVAL_PROMPTS


def load_model_with_adapter():
    """Load the base model (4-bit) with the trained LoRA adapter.

    Returns the model with adapter loaded. Use model.disable_adapter_layers()
    for base model inference, model.enable_adapter_layers() for aligned model.
    """
    print(f"Loading model: {MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the trained LoRA adapter on top of the base model
    print(f"Loading LoRA adapter from: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt_text):
    """Generate a single response using the model's chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the input prompt)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def get_toxicity_score(text, toxicity_pipe):
    """Get toxicity probability for a text.

    The classifier outputs labels 'toxic' or 'neutral' with confidence scores.
    We convert to a 0-1 toxicity probability regardless of which label is returned.
    """
    if not text.strip():
        return 0.0

    result = toxicity_pipe(text, truncation=True, max_length=512)[0]

    if result["label"].lower() == "toxic":
        return result["score"]
    else:
        return 1.0 - result["score"]


def evaluate():
    """Run full evaluation pipeline."""
    # Load model with adapter
    model, tokenizer = load_model_with_adapter()

    # Generate responses for all prompts with both models
    results = []
    print("\n" + "=" * 70)
    print("Generating responses...")
    print("=" * 70)

    for i, item in enumerate(EVAL_PROMPTS):
        prompt = item["prompt"]
        category = item["category"]
        print(f"\n[{i + 1}/{len(EVAL_PROMPTS)}] ({category}) {prompt[:60]}...")

        # Base model response (adapter disabled)
        model.disable_adapter_layers()
        base_response = generate_response(model, tokenizer, prompt)

        # Aligned model response (adapter enabled)
        model.enable_adapter_layers()
        aligned_response = generate_response(model, tokenizer, prompt)

        results.append({
            "prompt": prompt,
            "category": category,
            "base_response": base_response,
            "aligned_response": aligned_response,
        })

        print(f"  Base:    {base_response[:80]}...")
        print(f"  Aligned: {aligned_response[:80]}...")

    # Free GPU memory before loading toxicity model
    del model
    torch.cuda.empty_cache()

    # Score toxicity
    print(f"\nLoading toxicity classifier: {TOXICITY_MODEL_ID}")
    toxicity_pipe = pipeline(
        "text-classification",
        model=TOXICITY_MODEL_ID,
        device="cpu",  # Run on CPU to avoid GPU memory issues
    )

    print("Scoring responses...")
    for result in results:
        result["base_toxicity"] = get_toxicity_score(result["base_response"], toxicity_pipe)
        result["aligned_toxicity"] = get_toxicity_score(result["aligned_response"], toxicity_pipe)

    # Print formatted results
    print_results(results)

    # Save to JSON
    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {EVAL_RESULTS_PATH}")


def print_results(results):
    """Print formatted side-by-side comparison with toxicity scores."""
    print("\n")
    print("=" * 90)
    print("  EVALUATION RESULTS: Base Model vs. DPO-Aligned Model")
    print("=" * 90)

    for i, r in enumerate(results):
        print(f"\n{'─' * 90}")
        print(f"  [{r['category'].upper()}] Prompt: {r['prompt']}")
        print(f"{'─' * 90}")
        print(f"  🔴 BASE MODEL (toxicity: {r['base_toxicity']:.3f}):")
        # Wrap long responses
        for line in wrap_text(r["base_response"], width=84):
            print(f"     {line}")
        print()
        print(f"  🟢 ALIGNED MODEL (toxicity: {r['aligned_toxicity']:.3f}):")
        for line in wrap_text(r["aligned_response"], width=84):
            print(f"     {line}")

    # Summary statistics
    print("\n")
    print("=" * 90)
    print("  SUMMARY METRICS")
    print("=" * 90)

    base_toxicities = [r["base_toxicity"] for r in results]
    aligned_toxicities = [r["aligned_toxicity"] for r in results]

    avg_base = sum(base_toxicities) / len(base_toxicities)
    avg_aligned = sum(aligned_toxicities) / len(aligned_toxicities)

    if avg_base > 0:
        reduction = ((avg_base - avg_aligned) / avg_base) * 100
    else:
        reduction = 0.0

    print(f"\n  Average Toxicity (Base Model):     {avg_base:.4f}")
    print(f"  Average Toxicity (Aligned Model):  {avg_aligned:.4f}")
    print(f"  Toxicity Reduction:                {reduction:.1f}%")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    print(f"\n  {'Category':<16} {'Base Toxicity':>14} {'Aligned Toxicity':>17} {'Change':>10}")
    print(f"  {'─' * 60}")

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_base = sum(r["base_toxicity"] for r in cat_results) / len(cat_results)
        cat_aligned = sum(r["aligned_toxicity"] for r in cat_results) / len(cat_results)
        change = cat_aligned - cat_base
        arrow = "↓" if change < 0 else "↑" if change > 0 else "→"
        print(f"  {cat:<16} {cat_base:>14.4f} {cat_aligned:>17.4f} {arrow} {abs(change):>8.4f}")

    print(f"\n{'=' * 90}")


def wrap_text(text, width=84):
    """Simple word-wrapping for terminal display."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [""]


if __name__ == "__main__":
    evaluate()
