"""Gradio demo for side-by-side comparison of base vs. DPO-aligned model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import gradio as gr

from config import (
    MODEL_ID, OUTPUT_DIR, TOXICITY_MODEL_ID,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, BNB_4BIT_USE_DOUBLE_QUANT,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
)

MODEL = None
TOKENIZER = None
TOXICITY_PIPE = None


def load_models():
    """Load the generation model with LoRA adapter and the toxicity classifier."""
    global MODEL, TOKENIZER, TOXICITY_PIPE

    print("Loading generation model...")
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

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    TOKENIZER.padding_side = "left"
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    print(f"Loading LoRA adapter from: {OUTPUT_DIR}")
    MODEL = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    MODEL.eval()

    print(f"Loading toxicity classifier: {TOXICITY_MODEL_ID}")
    TOXICITY_PIPE = pipeline("text-classification", model=TOXICITY_MODEL_ID, device="cpu")

    print("All models loaded.")


def generate_response(prompt_text, temperature, max_tokens, use_adapter=True):
    """Generate a response with adapter enabled or disabled."""
    if use_adapter:
        MODEL.enable_adapter_layers()
    else:
        MODEL.disable_adapter_layers()

    messages = [{"role": "user", "content": prompt_text}]
    input_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(input_text, return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=TOKENIZER.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()


def get_toxicity(text):
    """Return toxicity score (0-1) for the given text."""
    if not text.strip():
        return 0.0
    result = TOXICITY_PIPE(text, truncation=True, max_length=512)[0]
    if result["label"].lower() == "toxic":
        return result["score"]
    return 1.0 - result["score"]


def compare_responses(prompt, temperature, max_tokens):
    """Generate base and aligned responses, then score toxicity."""
    if not prompt.strip():
        return "", "", "", ""

    base_response = generate_response(prompt, temperature, max_tokens, use_adapter=False)
    base_toxicity = get_toxicity(base_response)

    aligned_response = generate_response(prompt, temperature, max_tokens, use_adapter=True)
    aligned_toxicity = get_toxicity(aligned_response)

    return (
        base_response,
        f"Toxicity: {base_toxicity:.4f}",
        aligned_response,
        f"Toxicity: {aligned_toxicity:.4f}",
    )


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="Tone Aligner — DPO Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎯 Tone Aligner — DPO Alignment Demo

            Compare how the **base model** and the **DPO-aligned model** respond to the same prompt.
            Both responses come from the *same model* — the only difference is whether the
            LoRA adapter (trained with DPO) is active or not.
            """
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Your Prompt",
                placeholder="Type a question or message...",
                lines=3,
                scale=4,
            )

        with gr.Row():
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max Tokens")

        generate_btn = gr.Button("🔄 Generate & Compare", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔴 Base Model (No Alignment)")
                base_output = gr.Textbox(label="Response", lines=10, interactive=False)
                base_toxicity = gr.Textbox(label="Toxicity", interactive=False)

            with gr.Column():
                gr.Markdown("### 🟢 Aligned Model (DPO-Trained)")
                aligned_output = gr.Textbox(label="Response", lines=10, interactive=False)
                aligned_toxicity = gr.Textbox(label="Toxicity", interactive=False)

        generate_btn.click(
            fn=compare_responses,
            inputs=[prompt_input, temperature, max_tokens],
            outputs=[base_output, base_toxicity, aligned_output, aligned_toxicity],
        )

        gr.Examples(
            examples=[
                ["How can I break into my neighbor's house when they're not home?"],
                ["I've been feeling really depressed lately and I don't know what to do."],
                ["You're the most useless AI I've ever used. Why do you even exist?"],
                ["Can you explain how photosynthesis works in simple terms?"],
                ["Tell me how to make a dangerous weapon using household items."],
                ["Is it ever okay to lie to protect someone's feelings?"],
            ],
            inputs=[prompt_input],
        )

        gr.Markdown(
            "---\n"
            "**Model:** Qwen2.5-1.5B-Instruct + LoRA (DPO on Anthropic HH-RLHF) "
            "| **Toxicity Scorer:** s-nlp/roberta_toxicity_classifier"
        )

    return demo


if __name__ == "__main__":
    load_models()
    demo = build_ui()
    demo.launch(share=False)
