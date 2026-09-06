"""Real evaluation: tuned (base + LoRA adapter, 4-bit) vs untuned base model, on the held-out
test split. Computes exact-match, F1, ROUGE-L, and latency for both, via real `.generate()`
calls - no fabricated numbers. No GPT-4 baseline call (pipeline standard: avoid paid-API cost;
the untuned base model is the real, measurable comparison point).
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "model-artifacts", "qwen2.5-1.5b-support-lora")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SYSTEM_PROMPT = "You are a helpful customer support assistant."


def load_test_set(limit: int):
    path = os.path.join(DATA_DIR, "test.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(tokenizer):
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config(), device_map="auto")


def load_tuned(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config(), device_map="auto")
    return PeftModel.from_pretrained(base, ADAPTER_DIR)


def generate(model, tokenizer, instruction: str, max_new_tokens: int = 128):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_ms = (time.time() - start) * 1000
    text = tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()
    return text, elapsed_ms


def exact_match(pred: str, ref: str) -> float:
    return float(pred.strip().lower() == ref.strip().lower())


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in common.items():
        overlap += min(c, ref_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_model(model, tokenizer, test_rows, scorer, label: str):
    ems, f1s, rouges, latencies = [], [], [], []
    for i, row in enumerate(test_rows):
        pred, latency_ms = generate(model, tokenizer, row["instruction"])
        ref = row["response"]
        ems.append(exact_match(pred, ref))
        f1s.append(token_f1(pred, ref))
        rouges.append(scorer.score(ref, pred)["rougeL"].fmeasure)
        latencies.append(latency_ms)
        if (i + 1) % 20 == 0:
            print(f"[{label}] {i + 1}/{len(test_rows)} evaluated")

    n = len(test_rows)
    return {
        "n_examples": n,
        "exact_match": sum(ems) / n,
        "f1": sum(f1s) / n,
        "rouge_l": sum(rouges) / n,
        "avg_latency_ms": sum(latencies) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="number of test examples to evaluate (0 = all)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible; evaluation requires the GPU used for training.")
    if not os.path.isdir(ADAPTER_DIR):
        raise SystemExit(f"No trained adapter at {ADAPTER_DIR}; run scripts/train.py first.")

    test_rows = load_test_set(args.limit)
    print(f"Evaluating on {len(test_rows)} held-out test examples (never seen during training).")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    print("\n=== Evaluating UNTUNED base model ===")
    base_model = load_base(tokenizer)
    base_results = evaluate_model(base_model, tokenizer, test_rows, scorer, "base")
    del base_model
    torch.cuda.empty_cache()

    print("\n=== Evaluating FINE-TUNED (QLoRA) model ===")
    tuned_model = load_tuned(tokenizer)
    tuned_results = evaluate_model(tuned_model, tokenizer, test_rows, scorer, "tuned")

    report = {"base_model": base_results, "fine_tuned_model": tuned_results}
    print("\n=== RESULTS (real, measured) ===")
    print(json.dumps(report, indent=2))

    out_path = os.path.join(os.path.dirname(__file__), "..", "model-artifacts", "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
