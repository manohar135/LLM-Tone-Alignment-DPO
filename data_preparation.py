"""
Tone Aligner — Data Preparation

Downloads the Anthropic HH-RLHF dataset (harmless-base subset),
parses the conversation format, converts to DPO-compatible format,
filters low-quality examples, and saves to disk.

Usage:
    python data_preparation.py
"""

from datasets import load_dataset, Dataset, DatasetDict
from config import (
    DATASET_ID, DATASET_SUBSET, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES, DATASET_CACHE_DIR,
)


def parse_hh_conversation(text):
    """Parse Anthropic HH-RLHF conversation format into a list of message dicts.

    HH-RLHF format:
        "\\n\\nHuman: question\\n\\nAssistant: answer\\n\\nHuman: followup\\n\\nAssistant: response"

    Returns:
        List of {"role": "user"/"assistant", "content": "..."} dicts
    """
    messages = []

    # Split on "\n\nHuman: " to get each human turn (possibly followed by assistant turn)
    turns = text.split("\n\nHuman: ")

    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue

        if "\n\nAssistant: " in turn:
            human_part, assistant_part = turn.split("\n\nAssistant: ", 1)
            messages.append({"role": "user", "content": human_part.strip()})
            messages.append({"role": "assistant", "content": assistant_part.strip()})
        else:
            messages.append({"role": "user", "content": turn.strip()})

    return messages


def extract_dpo_fields(example):
    """Convert a single HH-RLHF example into DPO format.

    The 'chosen' and 'rejected' fields both contain the full conversation.
    They share the same prompt prefix and differ only in the final assistant response.

    Returns dict with:
        prompt:   list of messages (conversation history up to the last user turn)
        chosen:   list with one assistant message (the preferred response)
        rejected: list with one assistant message (the dispreferred response)
    """
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]

    # Split on the LAST "\n\nAssistant: " to separate prompt context from final response
    if "\n\nAssistant: " not in chosen_text or "\n\nAssistant: " not in rejected_text:
        return {"prompt": None, "chosen": None, "rejected": None}

    prompt_text, chosen_response = chosen_text.rsplit("\n\nAssistant: ", 1)
    _, rejected_response = rejected_text.rsplit("\n\nAssistant: ", 1)

    # Parse the prompt portion into structured messages
    prompt_messages = parse_hh_conversation(prompt_text)

    # Validate: last message in prompt must be from the user (the question being answered)
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        return {"prompt": None, "chosen": None, "rejected": None}

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": chosen_response.strip()}],
        "rejected": [{"role": "assistant", "content": rejected_response.strip()}],
    }


def is_valid(example):
    """Filter out low-quality or malformed examples."""
    if example["prompt"] is None:
        return False

    chosen_text = example["chosen"][0]["content"]
    rejected_text = example["rejected"][0]["content"]

    # Skip if chosen and rejected are identical (annotation noise)
    if chosen_text == rejected_text:
        return False

    # Skip if either response is too short to be meaningful
    if len(chosen_text) < 10 or len(rejected_text) < 10:
        return False

    return True


def prepare_dataset():
    """Full data preparation pipeline: download → parse → filter → split → save."""
    print(f"Loading dataset: {DATASET_ID} ({DATASET_SUBSET})...")
    raw = load_dataset(DATASET_ID, data_dir=DATASET_SUBSET, split="train")
    print(f"  Raw examples: {len(raw)}")

    # Parse conversations and extract prompt/chosen/rejected
    print("Extracting DPO fields...")
    parsed = raw.map(extract_dpo_fields, remove_columns=raw.column_names)

    # Filter invalid examples
    print("Filtering low-quality examples...")
    filtered = parsed.filter(is_valid)
    print(f"  After filtering: {len(filtered)}")

    # Shuffle and take the subset we need
    filtered = filtered.shuffle(seed=42)
    total_needed = NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES
    if len(filtered) < total_needed:
        print(f"  Warning: only {len(filtered)} examples available, need {total_needed}")
        total_needed = len(filtered)

    subset = filtered.select(range(min(total_needed, len(filtered))))

    # Split into train and eval
    split = subset.train_test_split(
        test_size=NUM_EVAL_SAMPLES,
        seed=42,
    )
    dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })

    print(f"\nFinal dataset:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Eval:  {len(dataset['eval'])} examples")

    # Show a sample
    sample = dataset["train"][0]
    print(f"\n── Sample ──────────────────────────────────")
    print(f"Prompt:   {sample['prompt'][-1]['content'][:100]}...")
    print(f"Chosen:   {sample['chosen'][0]['content'][:100]}...")
    print(f"Rejected: {sample['rejected'][0]['content'][:100]}...")

    # Save to disk
    dataset.save_to_disk(DATASET_CACHE_DIR)
    print(f"\nDataset saved to: {DATASET_CACHE_DIR}")

    return dataset


if __name__ == "__main__":
    prepare_dataset()
