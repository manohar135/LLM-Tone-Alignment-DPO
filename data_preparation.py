"""Build a balanced, leak-free preference dataset for DPO training.

Combines two complementary sources (50/50 split):
  1. PKU-Alignment/PKU-SafeRLHF: teaches clear safety boundaries (safe vs unsafe)
     and helpful safe responses (prevents over-refusal).
  2. argilla/ultrafeedback-binarized-preferences-cleaned: teaches polite tone,
     clarity, and helpfulness.
"""

import argparse
import json
import os
import random
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

from config import (
    DATASET_CACHE_DIR,
    NUM_EVAL_SAMPLES,
    NUM_TRAIN_SAMPLES,
    PKU_DATASET_ID,
    PKU_RATIO,
    SANITY_EVAL_SAMPLES,
    SANITY_TRAIN_SAMPLES,
    SEED,
    UF_DATASET_ID,
)

MIN_RESPONSE_CHARS = 20
MAX_RESPONSE_CHARS = 3000
MAX_PROMPT_CHARS = 2000


def _clean(text):
    return (text or "").strip()


def _is_valid(prompt_msgs, chosen_text, rejected_text):
    """Filter out empty, identical, malformed, or overly long pairs."""
    if not prompt_msgs or prompt_msgs[-1]["role"] != "user":
        return False
    prompt_len = sum(len(m["content"]) for m in prompt_msgs)
    if prompt_len < 5 or prompt_len > MAX_PROMPT_CHARS:
        return False
    if chosen_text == rejected_text:
        return False
    if not (MIN_RESPONSE_CHARS <= len(chosen_text) <= MAX_RESPONSE_CHARS):
        return False
    if not (MIN_RESPONSE_CHARS <= len(rejected_text) <= MAX_RESPONSE_CHARS):
        return False
    return True


def _deduplicate(records):
    seen = set()
    unique = []
    for r in records:
        key = (
            r["prompt"][-1]["content"],
            r["chosen"][0]["content"],
            r["rejected"][0]["content"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def load_pku_saferlhf(train_n, eval_n, seed=SEED):
    """Load safety + safe-helpfulness pairs from official train/test splits."""
    print(f"\nLoading {PKU_DATASET_ID}...")
    rng = random.Random(seed)

    def _extract(split_name, target_n):
        raw = load_dataset(PKU_DATASET_ID, split=split_name)
        safety_pairs = []
        helpful_pairs = []

        for ex in raw:
            prompt = _clean(ex.get("prompt"))
            r0 = _clean(ex.get("response_0"))
            r1 = _clean(ex.get("response_1"))
            safe0 = bool(ex.get("is_response_0_safe", False))
            safe1 = bool(ex.get("is_response_1_safe", False))
            better_id = ex.get("better_response_id", -1)

            prompt_msgs = [{"role": "user", "content": prompt}]

            # One safe, one unsafe -> strong safety preference pair
            if safe0 != safe1:
                chosen, rejected = (r0, r1) if safe0 else (r1, r0)
                if _is_valid(prompt_msgs, chosen, rejected):
                    safety_pairs.append({
                        "prompt": prompt_msgs,
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                        "_source": "pku_safety",
                    })
            # Both safe -> prefer the more helpful response (reduces over-refusal)
            elif safe0 and safe1 and better_id in (0, 1):
                chosen, rejected = (r0, r1) if better_id == 0 else (r1, r0)
                if _is_valid(prompt_msgs, chosen, rejected):
                    helpful_pairs.append({
                        "prompt": prompt_msgs,
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                        "_source": "pku_safe_helpful",
                    })
            # Both unsafe -> skip (never teach the model an unsafe chosen answer)

        safety_pairs = _deduplicate(safety_pairs)
        helpful_pairs = _deduplicate(helpful_pairs)
        rng.shuffle(safety_pairs)
        rng.shuffle(helpful_pairs)

        # 70% clear safety-contrast pairs + 30% safe-helpful pairs
        safety_quota = int(target_n * 0.7)
        helpful_quota = target_n - safety_quota
        selected = safety_pairs[:safety_quota] + helpful_pairs[:helpful_quota]

        if len(selected) < target_n:
            rest = safety_pairs[safety_quota:] + helpful_pairs[helpful_quota:]
            selected.extend(rest[: target_n - len(selected)])

        rng.shuffle(selected)
        print(f"  PKU [{split_name}]: selected {len(selected)} pairs")
        return selected

    return _extract("train", train_n), _extract("test", eval_n)


def _extract_uf_messages(msg_list, fallback_prompt):
    """Extract (prompt_messages, final_assistant_text) from UltraFeedback format."""
    if not isinstance(msg_list, list) or not msg_list:
        return None, None

    cleaned = [
        {"role": m.get("role"), "content": _clean(m.get("content"))}
        for m in msg_list
        if m.get("role") in ("user", "assistant", "system") and _clean(m.get("content"))
    ]
    if not cleaned or cleaned[-1]["role"] != "assistant":
        return None, None

    prefix = cleaned[:-1] or [{"role": "user", "content": _clean(fallback_prompt)}]
    return prefix, cleaned[-1]["content"]


def load_ultrafeedback(train_n, eval_n, seed=SEED):
    """Load high-rated tone & helpfulness pairs with prompt-level train/eval split."""
    print(f"\nLoading {UF_DATASET_ID}...")
    raw = load_dataset(UF_DATASET_ID, split="train")
    rng = random.Random(seed)

    records = []
    for ex in raw:
        c_rating = ex.get("chosen-rating")
        r_rating = ex.get("rejected-rating")
        if c_rating is not None and r_rating is not None:
            if c_rating < 4.0 or c_rating <= r_rating:
                continue

        fallback = ex.get("prompt", "")
        c_prefix, chosen = _extract_uf_messages(ex.get("chosen"), fallback)
        r_prefix, rejected = _extract_uf_messages(ex.get("rejected"), fallback)

        if c_prefix is None or c_prefix != r_prefix:
            continue
        if not _is_valid(c_prefix, chosen, rejected):
            continue

        records.append({
            "prompt": c_prefix,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "_source": "ultrafeedback",
        })

    records = _deduplicate(records)

    # Group by prompt before splitting so no prompt leaks between train and eval
    by_prompt = {}
    for r in records:
        by_prompt.setdefault(r["prompt"][-1]["content"], []).append(r)

    keys = list(by_prompt.keys())
    rng.shuffle(keys)

    eval_records, train_records = [], []
    for k in keys:
        if len(eval_records) < eval_n:
            eval_records.extend(by_prompt[k])
        else:
            train_records.extend(by_prompt[k])

    eval_records = eval_records[:eval_n]
    rng.shuffle(train_records)
    train_records = train_records[:train_n]

    print(f"  UltraFeedback: selected {len(train_records)} train, {len(eval_records)} eval")
    return train_records, eval_records


def _strip_internal(records):
    return [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
    ]


def prepare_dataset(sanity=False):
    """Download, filter, balance, and save the DPO preference dataset."""
    train_n = SANITY_TRAIN_SAMPLES if sanity else NUM_TRAIN_SAMPLES
    eval_n = SANITY_EVAL_SAMPLES if sanity else NUM_EVAL_SAMPLES
    mode = "sanity" if sanity else "full"

    print(f"Building {mode} dataset: {train_n} train / {eval_n} eval (50% PKU-SafeRLHF + 50% UltraFeedback)")

    pku_train_n = int(train_n * PKU_RATIO)
    pku_eval_n = int(eval_n * PKU_RATIO)

    pku_train, pku_eval = load_pku_saferlhf(pku_train_n, pku_eval_n)
    uf_train, uf_eval = load_ultrafeedback(train_n - pku_train_n, eval_n - pku_eval_n)

    rng = random.Random(SEED)
    train_records = pku_train + uf_train
    eval_records = pku_eval + uf_eval
    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    dataset = DatasetDict({
        "train": Dataset.from_list(_strip_internal(train_records)),
        "eval": Dataset.from_list(_strip_internal(eval_records)),
    })

    os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
    dataset.save_to_disk(DATASET_CACHE_DIR)

    stats = {
        "mode": mode,
        "train_count": len(dataset["train"]),
        "eval_count": len(dataset["eval"]),
        "train_sources": dict(Counter(r["_source"] for r in train_records)),
        "eval_sources": dict(Counter(r["_source"] for r in eval_records)),
    }
    with open(os.path.join(DATASET_CACHE_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved dataset to: {DATASET_CACHE_DIR}")
    print(f"  Train: {stats['train_count']} -> {stats['train_sources']}")
    print(f"  Eval:  {stats['eval_count']}  -> {stats['eval_sources']}")

    sample = dataset["train"][0]
    print("\nSample example:")
    print(f"  Prompt:   {sample['prompt'][-1]['content'][:110]}...")
    print(f"  Chosen:   {sample['chosen'][0]['content'][:110]}...")
    print(f"  Rejected: {sample['rejected'][0]['content'][:110]}...")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare balanced DPO dataset.")
    parser.add_argument("--sanity", action="store_true", help="Build smaller 1.6K/200 sanity dataset")
    args = parser.parse_args()
    prepare_dataset(sanity=args.sanity)
