"""Evaluate base vs. DPO-aligned model using toxicity scoring."""

import json
import os
import textwrap

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
    """Load the base model with the trained LoRA adapter."""
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

    print(f"Loading LoRA adapter from: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt_text):
    """Generate a response using the model's chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def get_toxicity_score(text, toxicity_pipe):
    """Return toxicity probability (0-1) for a given text."""
    if not text.strip():
        return 0.0

    result = toxicity_pipe(text, truncation=True, max_length=512)[0]
    if result["label"].lower() == "toxic":
        return result["score"]
    return 1.0 - result["score"]


def evaluate():
    """Run the full evaluation: generate responses, score toxicity, print and save results."""
    model, tokenizer = load_model_with_adapter()

    results = []
    print(f"\nGenerating responses for {len(EVAL_PROMPTS)} prompts...")

    for i, item in enumerate(EVAL_PROMPTS):
        prompt = item["prompt"]
        category = item["category"]
        print(f"  [{i + 1}/{len(EVAL_PROMPTS)}] ({category}) {prompt[:60]}...")

        model.disable_adapter_layers()
        base_response = generate_response(model, tokenizer, prompt)

        model.enable_adapter_layers()
        aligned_response = generate_response(model, tokenizer, prompt)

        results.append({
            "prompt": prompt,
            "category": category,
            "base_response": base_response,
            "aligned_response": aligned_response,
        })

    # Free GPU memory before loading toxicity model
    del model
    torch.cuda.empty_cache()

    print(f"\nLoading toxicity classifier: {TOXICITY_MODEL_ID}")
    toxicity_pipe = pipeline("text-classification", model=TOXICITY_MODEL_ID, device="cpu")

    print("Scoring responses...")
    for r in results:
        r["base_toxicity"] = get_toxicity_score(r["base_response"], toxicity_pipe)
        r["aligned_toxicity"] = get_toxicity_score(r["aligned_response"], toxicity_pipe)

    print_results(results)

    os.makedirs(os.path.dirname(EVAL_RESULTS_PATH), exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {EVAL_RESULTS_PATH}")


def print_results(results):
    """Print formatted comparison table with toxicity scores."""
    print("\n" + "=" * 90)
    print("  EVALUATION RESULTS: Base Model vs. DPO-Aligned Model")
    print("=" * 90)

    for r in results:
        print(f"\n{'─' * 90}")
        print(f"  [{r['category'].upper()}] {r['prompt']}")
        print(f"{'─' * 90}")

        print(f"  🔴 BASE (toxicity: {r['base_toxicity']:.3f}):")
        print(textwrap.indent(textwrap.fill(r["base_response"], width=84), "     "))

        print(f"\n  🟢 ALIGNED (toxicity: {r['aligned_toxicity']:.3f}):")
        print(textwrap.indent(textwrap.fill(r["aligned_response"], width=84), "     "))

    # Summary
    base_scores = [r["base_toxicity"] for r in results]
    aligned_scores = [r["aligned_toxicity"] for r in results]
    avg_base = sum(base_scores) / len(base_scores)
    avg_aligned = sum(aligned_scores) / len(aligned_scores)
    reduction = ((avg_base - avg_aligned) / avg_base * 100) if avg_base > 0 else 0.0

    print("\n\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print(f"\n  Avg Toxicity (Base):    {avg_base:.4f}")
    print(f"  Avg Toxicity (Aligned): {avg_aligned:.4f}")
    print(f"  Reduction:              {reduction:.1f}%")

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    print(f"\n  {'Category':<16} {'Base':>14} {'Aligned':>17} {'Change':>10}")
    print(f"  {'─' * 60}")

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_base = sum(r["base_toxicity"] for r in cat_results) / len(cat_results)
        cat_aligned = sum(r["aligned_toxicity"] for r in cat_results) / len(cat_results)
        change = cat_aligned - cat_base
        arrow = "↓" if change < 0 else "↑" if change > 0 else "→"
        print(f"  {cat:<16} {cat_base:>14.4f} {cat_aligned:>17.4f} {arrow} {abs(change):>8.4f}")

    print(f"\n{'=' * 90}")


if __name__ == "__main__":
    evaluate()
