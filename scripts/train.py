"""Real QLoRA SFT fine-tune of Qwen2.5-1.5B-Instruct on the customer-support dataset.

Targets a 6GB laptop GPU (RTX 4050): 4-bit NF4 quantization (bitsandbytes) + LoRA adapters via
`peft`, trained with `trl.SFTTrainer`. Writes real progress (loss, step, lr) to the
`fine_tune_metrics` Postgres table as training proceeds, and a `training_jobs` row for the run.

Run: `python scripts/train.py` (from the scripts/ venv, after scripts/prepare_data.py).
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from db import complete_job, create_job, ensure_schema, fail_job, get_engine, log_metric

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model-artifacts", "qwen2.5-1.5b-support-lora")

# Qwen2's attention + MLP projection module names (verified via model.named_modules()) -
# different from the PRD sketch's Mistral-style q_proj/v_proj-only list.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class DBMetricsCallback(TrainerCallback):
    def __init__(self, engine, job_id):
        self.engine = engine
        self.job_id = job_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            log_metric(
                self.engine,
                self.job_id,
                epoch=int(state.epoch or 0),
                step=state.global_step,
                loss=logs.get("loss"),
                val_accuracy=logs.get("eval_accuracy"),
                learning_rate=logs.get("learning_rate"),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        log_metric(
            self.engine,
            self.job_id,
            epoch=int(state.epoch or 0),
            step=state.global_step,
            loss=metrics.get("eval_loss"),
            val_accuracy=metrics.get("eval_accuracy"),
            learning_rate=None,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1, help="override epochs, useful for quick smoke runs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. This training run requires the RTX 4050 GPU on the host.")

    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "val.jsonl")
    if not os.path.exists(train_path):
        raise SystemExit(f"{train_path} not found - run scripts/prepare_data.py first.")

    train_ds = load_dataset("json", data_files=train_path, split="train")
    val_ds = load_dataset("json", data_files=val_path, split="train")
    print(f"Train examples: {len(train_ds)}, val examples: {len(val_ds)}")

    engine = get_engine()
    ensure_schema(engine)
    job_id = create_job(engine, model_name="qwen2.5-1.5b-support-lora", base_model=BASE_MODEL, dataset_size=len(train_ds))
    print(f"Created training_jobs row: {job_id}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.config.use_cache = False

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        sft_config = SFTConfig(
            output_dir=os.path.join(os.path.dirname(__file__), "..", "model-artifacts", "checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="no",
            bf16=True,
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[DBMetricsCallback(engine, job_id)],
        )

        start = time.time()
        train_result = trainer.train()
        wall_clock_s = time.time() - start
        print(f"Training finished in {wall_clock_s:.1f}s ({wall_clock_s / 60:.1f} min)")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        final_loss = train_result.training_loss
        complete_job(engine, job_id, accuracy=None)

        with open(os.path.join(OUTPUT_DIR, "train_summary.json"), "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "base_model": BASE_MODEL,
                    "train_examples": len(train_ds),
                    "val_examples": len(val_ds),
                    "epochs": args.epochs,
                    "max_steps": args.max_steps,
                    "final_training_loss": final_loss,
                    "wall_clock_seconds": wall_clock_s,
                },
                f,
                indent=2,
            )
        print(f"Adapter saved to {OUTPUT_DIR}")

    except Exception:
        fail_job(engine, job_id)
        raise


if __name__ == "__main__":
    main()
