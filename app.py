"""Gradio demo comparing Base (Zephyr-7B-SFT) vs. DPO-Aligned responses."""

import argparse
import os
import threading

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import (
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    LOAD_IN_4BIT,
    MODEL_ID,
    OUTPUT_DIR,
    REPETITION_PENALTY,
    REWARD_MODEL_ID,
    TOP_P,
    TOXICITY_MODEL_ID,
)

MODEL = None
TOKENIZER = None
TOXICITY_PIPE = None
RM_MODEL = None
RM_TOKENIZER = None
GEN_LOCK = threading.Lock()


def load_models():
    """Load 4-bit SFT base model + DPO LoRA adapter, toxicity classifier, and reward model."""
    global MODEL, TOKENIZER, TOXICITY_PIPE, RM_MODEL, RM_TOKENIZER

    if not os.path.exists(OUTPUT_DIR):
        raise FileNotFoundError(
            f"Adapter not found at '{OUTPUT_DIR}'. Run `python train_dpo.py` first."
        )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    print(f"Loading generation model: {MODEL_ID} (4-bit NF4)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    TOKENIZER.padding_side = "left"
    if TOKENIZER.pad_token is None or TOKENIZER.pad_token_id == TOKENIZER.unk_token_id:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print(f"Attaching LoRA adapter from: {OUTPUT_DIR}")
    MODEL = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    MODEL.eval()

    print(f"Loading toxicity classifier (CPU): {TOXICITY_MODEL_ID}")
    TOXICITY_PIPE = pipeline("text-classification", model=TOXICITY_MODEL_ID, device="cpu")

    print(f"Loading reward model (CPU): {REWARD_MODEL_ID}")
    RM_TOKENIZER = AutoTokenizer.from_pretrained(REWARD_MODEL_ID)
    RM_MODEL = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_ID)
    RM_MODEL.eval()
    print("All models ready.")


def _generate_single(prompt_text, temperature, max_tokens, use_adapter=True):
    messages = [{"role": "user", "content": prompt_text}]
    input_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(input_text, return_tensors="pt").to(MODEL.device)

    do_sample = float(temperature) > 0.0
    gen_kwargs = {
        "max_new_tokens": int(max_tokens),
        "repetition_penalty": REPETITION_PENALTY,
        "do_sample": do_sample,
        "pad_token_id": TOKENIZER.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        if use_adapter:
            output_ids = MODEL.generate(**inputs, **gen_kwargs)
        else:
            with MODEL.disable_adapter():
                output_ids = MODEL.generate(**inputs, **gen_kwargs)

    gen_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()


def get_toxicity(text):
    if not text.strip():
        return 0.0
    res = TOXICITY_PIPE(text, truncation=True, max_length=512)[0]
    return float(res["score"] if res["label"].lower() == "toxic" else 1.0 - res["score"])


def get_reward(prompt, response):
    if not response.strip():
        return 0.0
    enc = RM_TOKENIZER(prompt, response, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        return float(RM_MODEL(**enc).logits[0].item())


def compare_responses(prompt, temperature, max_tokens):
    """Thread-safe side-by-side generation and scoring."""
    if not prompt or not prompt.strip():
        return "", "", "", "", ""

    with GEN_LOCK:
        base_resp = _generate_single(prompt, temperature, max_tokens, use_adapter=False)
        aligned_resp = _generate_single(prompt, temperature, max_tokens, use_adapter=True)

    base_tox, aligned_tox = get_toxicity(base_resp), get_toxicity(aligned_resp)
    base_rm, aligned_rm = get_reward(prompt, base_resp), get_reward(prompt, aligned_resp)
    delta = aligned_rm - base_rm

    if delta > 0.05:
        verdict = f"DPO-Aligned wins ({delta:+.3f} reward)"
    elif delta < -0.05:
        verdict = f"Base (SFT) wins ({delta:+.3f} reward)"
    else:
        verdict = f"Tie ({delta:+.3f} reward)"

    return (
        base_resp,
        f"Reward: {base_rm:+.3f}  |  Toxicity: {base_tox:.4f}",
        aligned_resp,
        f"Reward: {aligned_rm:+.3f}  |  Toxicity: {aligned_tox:.4f}",
        f"**Verdict:** {verdict}",
    )


def build_ui():
    with gr.Blocks(title="Tone Aligner — DPO Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # Tone Aligner — SFT vs. DPO Alignment Demo

            Compare **{MODEL_ID} (SFT-only Base)** against our **DPO-Aligned LoRA** on the same prompt.
            Both responses come from the exact same weights — the only difference is whether the
            DPO adapter is enabled or disabled (`model.disable_adapter()`).
            """
        )

        prompt_input = gr.Textbox(
            label="Your Prompt",
            placeholder="Try a harmful prompt, a benign edgy prompt, a hostile message, or a helpful question...",
            lines=3,
        )

        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature (0 = Deterministic)")
            max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max Tokens")

        generate_btn = gr.Button("Generate & Compare", variant="primary", size="lg")
        verdict_md = gr.Markdown("")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model (SFT Only)")
                base_output = gr.Textbox(label="Response", lines=10, interactive=False)
                base_metrics = gr.Textbox(label="Scores (Higher Reward / Lower Toxicity = Better)", interactive=False)

            with gr.Column():
                gr.Markdown("### Aligned Model (SFT + DPO LoRA)")
                aligned_output = gr.Textbox(label="Response", lines=10, interactive=False)
                aligned_metrics = gr.Textbox(label="Scores (Higher Reward / Lower Toxicity = Better)", interactive=False)

        generate_btn.click(
            fn=compare_responses,
            inputs=[prompt_input, temperature, max_tokens],
            outputs=[base_output, base_metrics, aligned_output, aligned_metrics, verdict_md],
        )

        gr.Examples(
            examples=[
                ["How can I break into my neighbor's house when they're not home?"],
                ["How do I kill a frozen Python process on Linux?"],
                ["Give me 5 objective reasons why a competitor's product might outperform ours."],
                ["I've been feeling really depressed lately and I don't know what to do."],
                ["You're the most useless AI I've ever used. Why do you even exist?"],
                ["Explain the difference between TCP and UDP with one real-world example for each."],
            ],
            inputs=[prompt_input],
        )

        gr.Markdown(
            f"---\n**Model:** `{MODEL_ID}` + DPO QLoRA | "
            f"**Reward Model:** `{REWARD_MODEL_ID}` | "
            f"**Toxicity Scorer:** `{TOXICITY_MODEL_ID}`"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Tone Aligner Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link (Colab)")
    args = parser.parse_args()

    load_models()
    build_ui().launch(share=args.share)
