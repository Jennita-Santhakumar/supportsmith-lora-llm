"""Download and prepare a customer-support instruction dataset for SFT.

Dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (HuggingFace) - ~27k
real customer-support instruction/response pairs across intents like refunds, cancellations,
account/password issues, shipping, complaints, etc. Exactly the "customer-support / helpful
assistant tone" domain called for in the plan. We take a random subset (~2.6k) for a fast QLoRA
run on a 6GB laptop GPU, with a held-out val/test split that never touches training.

Output: JSONL files (train.jsonl, val.jsonl, test.jsonl) under data/, each line
{"instruction": ..., "response": ..., "category": ..., "intent": ...}. Also formats a `text`
field using the Qwen2.5 chat template for direct use by trl.SFTTrainer.
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful customer support assistant."

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def format_example(tokenizer, instruction: str, response: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=2600, help="total examples to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    random.seed(args.seed)

    print(f"Downloading {DATASET_NAME} from HuggingFace ...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Full dataset size: {len(ds)}")

    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_total, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    n = len(ds)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val

    train_rows = ds.select(range(0, n_train))
    val_rows = ds.select(range(n_train, n_train + n_val))
    test_rows = ds.select(range(n_train + n_val, n))

    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    for name, rows in splits.items():
        path = os.path.join(DATA_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                instruction = row["instruction"].strip()
                response = row["response"].strip()
                record = {
                    "instruction": instruction,
                    "response": response,
                    "category": row.get("category"),
                    "intent": row.get("intent"),
                    "text": format_example(tokenizer, instruction, response),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(rows)} examples to {path}")

    print(
        f"\nSplit sizes -> train={n_train}, val={n_val}, test={n_test} "
        f"(held out, no leakage: test/val rows are disjoint from train by dataset index range)."
    )


if __name__ == "__main__":
    main()
