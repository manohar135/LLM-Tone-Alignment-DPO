"""Download and preprocess the Anthropic HH-RLHF dataset for DPO training."""

from datasets import load_dataset, Dataset, DatasetDict
from config import DATASET_ID, DATASET_SUBSET, NUM_TRAIN_SAMPLES, NUM_EVAL_SAMPLES, DATASET_CACHE_DIR


def parse_hh_conversation(text):
    """Parse HH-RLHF conversation format into a list of message dicts."""
    messages = []
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
    """Split an HH-RLHF example into prompt, chosen, and rejected messages."""
    chosen_text = example["chosen"]
    rejected_text = example["rejected"]

    if "\n\nAssistant: " not in chosen_text or "\n\nAssistant: " not in rejected_text:
        return {"prompt": None, "chosen": None, "rejected": None}

    # Split on the last assistant turn to separate prompt from final response
    prompt_text, chosen_response = chosen_text.rsplit("\n\nAssistant: ", 1)
    _, rejected_response = rejected_text.rsplit("\n\nAssistant: ", 1)

    prompt_messages = parse_hh_conversation(prompt_text)

    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        return {"prompt": None, "chosen": None, "rejected": None}

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": chosen_response.strip()}],
        "rejected": [{"role": "assistant", "content": rejected_response.strip()}],
    }


def is_valid(example):
    """Filter out malformed or low-quality examples."""
    if example["prompt"] is None:
        return False

    chosen_text = example["chosen"][0]["content"]
    rejected_text = example["rejected"][0]["content"]

    if chosen_text == rejected_text:
        return False
    if len(chosen_text) < 10 or len(rejected_text) < 10:
        return False

    return True


def prepare_dataset():
    """Full pipeline: download, parse, filter, split, and save to disk."""
    print(f"Loading dataset: {DATASET_ID} ({DATASET_SUBSET})...")
    raw = load_dataset(DATASET_ID, data_dir=DATASET_SUBSET, split="train")
    print(f"  Raw examples: {len(raw)}")

    print("Extracting DPO fields...")
    parsed = raw.map(extract_dpo_fields, remove_columns=raw.column_names)

    print("Filtering low-quality examples...")
    filtered = parsed.filter(is_valid)
    print(f"  After filtering: {len(filtered)}")

    filtered = filtered.shuffle(seed=42)
    total_needed = NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES

    if len(filtered) < total_needed:
        print(f"  Warning: only {len(filtered)} examples available, need {total_needed}")
        total_needed = len(filtered)

    subset = filtered.select(range(total_needed))

    split = subset.train_test_split(test_size=NUM_EVAL_SAMPLES, seed=42)
    dataset = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })

    print(f"\nFinal dataset:")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Eval:  {len(dataset['eval'])} examples")

    sample = dataset["train"][0]
    print(f"\nSample:")
    print(f"  Prompt:   {sample['prompt'][-1]['content'][:100]}...")
    print(f"  Chosen:   {sample['chosen'][0]['content'][:100]}...")
    print(f"  Rejected: {sample['rejected'][0]['content'][:100]}...")

    dataset.save_to_disk(DATASET_CACHE_DIR)
    print(f"\nDataset saved to: {DATASET_CACHE_DIR}")

    return dataset


if __name__ == "__main__":
    prepare_dataset()
