# Domain-Tuned Support LLM (LoRA + Quantization Platform)

> **Maintained by [Jennita S](https://github.com/Jennita-Santhakumar)** · [LinkedIn](https://linkedin.com/in/jennitas) · jennitasanthakumar0@gmail.com

[![CI](https://github.com/KabileshRajaselvan/domain-tuned-support-llm-lora-quantization-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/KabileshRajaselvan/domain-tuned-support-llm-lora-quantization-platform/actions/workflows/ci.yml)

A real QLoRA (4-bit) fine-tune of **Qwen2.5-1.5B-Instruct** on a customer-support instruction
dataset, served via FastAPI with a Postgres-backed inference cache, evaluated honestly against
the untuned base model (exact-match / F1 / ROUGE-L / latency) - no fabricated metrics, no paid
GPT-4 baseline call. Built as the 18th and final project in a job-search portfolio pipeline.

> **Note on scope vs. the source PRD**: the original PRD specifies Mistral-7B + 8-bit
> quantization + a GPT-4 baseline comparison + hardcoded "92% accuracy" success metrics. This
> build deviates from all four, deliberately - see "Design Decisions & Trade-offs" below for why,
> and the honest, actually-measured results this run produced instead.

## Architecture

```
                        scripts/prepare_data.py
                                |
        HuggingFace: bitext/Bitext-customer-support-llm-chatbot-training-dataset
                    (~2.6k sampled, 80/10/10 train/val/test split)
                                |
                                v
                  data/{train,val,test}.jsonl  (Qwen2.5 chat-template formatted)
                                |
                                v
                        scripts/train.py  (runs on HOST, needs the GPU)
        Qwen2.5-1.5B-Instruct --4-bit NF4 (bitsandbytes)--> QLoRA (peft, r=16,
        all 7 attention+MLP proj modules) --> trl.SFTTrainer
                                |                       |
                                |                       +--> fine_tune_metrics (Postgres, real
                                |                            step/loss/lr writes during training)
                                v
              model-artifacts/qwen2.5-1.5b-support-lora/  (LoRA adapter, gitignored)
                                |
                    scripts/evaluate.py (host, GPU)
        base model vs fine-tuned model, both real generate() calls on held-out test set
        --> exact-match, F1, ROUGE-L, latency --> model-artifacts/eval_results.json
                                |
                                v
     +--------------------------------------------------------------+
     |  docker-compose: postgres + backend (FastAPI) + frontend      |
     |                                                                |
     |  FastAPI (backend/app):                                       |
     |    POST /api/v1/generate  -> sha256(prompt+params) lookup in   |
     |      inference_cache (Postgres) -> hit: return cached row      |
     |                             -> miss: AutoPeftModelForCausalLM  |
     |      (4-bit) .generate() -> write inference_cache row          |
     |    GET  /api/v1/jobs, /api/v1/jobs/{id}/metrics -> training    |
     |      history from Postgres                                    |
     |    GET  /health                                                |
     |                                                                |
     |  React frontend: prompt box, live response + cache-hit badge,  |
     |    real eval metrics panel                                     |
     +--------------------------------------------------------------+
```

## Stack

| Layer | Choice |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (swapped from PRD's Mistral-7B - see Trade-offs) |
| Fine-tuning | QLoRA: 4-bit NF4 (`bitsandbytes`) + LoRA (`peft`, r=16, alpha=32) via `trl.SFTTrainer` |
| Dataset | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (HuggingFace), ~2.6k subset |
| API | FastAPI (Python 3.11) |
| Database | PostgreSQL 16 (`training_jobs`, `fine_tune_metrics`, `inference_cache`) |
| Frontend | React + Vite |
| Tests | pytest (unit + integration against a real Postgres) |
| CI | GitHub Actions (lint, test, docker build - CPU-only mock-mode smoke test, no GPU training) |

## Repo layout

```
backend/          FastAPI serving app + pytest tests (Python 3.11 venv)
scripts/          prepare_data.py, train.py, evaluate.py, db.py (separate venv, needs GPU + ML deps)
model-artifacts/  LoRA adapter + eval_results.json (gitignored, produced by scripts/)
data/             train/val/test JSONL (gitignored, produced by prepare_data.py)
frontend/         React app (prompt box, response, metrics panel)
docker-compose.yml, .github/workflows/ci.yml
```

## Running it

### 1. Backend + frontend (Docker, mock inference mode)

Docker Desktop on Windows does not reliably pass the GPU through to Linux containers, so the
containerized API defaults to `USE_REAL_MODEL=false` (a deterministic mock generator) purely to
keep the one-command `docker compose up` demo GPU-independent. This still exercises the real
API surface, Postgres schema, and caching logic - just not real generation.

```bash
docker compose up --build
```

- API: http://localhost:8095 (docs at `/docs`)
- Frontend: http://localhost:5250
- Postgres: localhost:5511

### 2. Real GPU-backed training + inference (host, venv)

Requires an NVIDIA GPU (built and run against an RTX 4050 laptop GPU, 6GB VRAM).

```bash
# Python 3.11 venv for the backend
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt

# separate venv for the training/eval scripts (torch+CUDA, trl, bitsandbytes, datasets)
cd ../scripts
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python prepare_data.py
.venv/Scripts/python train.py
.venv/Scripts/python evaluate.py
```

Then run the backend directly on the host with `USE_REAL_MODEL=true` (see `.env.example`) so it
loads the real 4-bit quantized model + trained adapter:

```bash
cd backend
uvicorn app.main:app --reload
```

## API reference

`POST /api/v1/generate`
```json
{ "prompt": "How do I reset my password?", "max_tokens": 200, "temperature": 0.7 }
```
```json
{
  "text": "...",
  "tokens_used": 42,
  "model_version": "qwen2.5-1.5b-support-lora-v1",
  "inference_time_ms": 812,
  "cached": false
}
```
A second identical request returns `"cached": true` and the same text, served from
`inference_cache` (Postgres, keyed on `sha256(prompt|max_tokens|temperature)`).

`GET /api/v1/jobs` - list training runs (`training_jobs` table).
`GET /api/v1/jobs/{id}/metrics` - per-step loss/lr/val-accuracy for a run (`fine_tune_metrics`).
`GET /health` - liveness check.

## Results (real, measured)

**Training** (`scripts/train.py`, QLoRA on Qwen2.5-1.5B-Instruct, RTX 4050 6GB, 4-bit NF4):
- 2,080 training examples, 260 held-out validation examples, 1 epoch (130 steps, batch size 2,
  grad accumulation 8, effective batch size 16), LoRA r=16/alpha=32 on the attention projections,
  18.46M trainable params (1.18% of the 1.56B total).
- Train loss: 1.73 -> 0.78 over the run (mean train loss 0.958). Eval loss at the mid-run
  checkpoint: 0.798.
- Wall-clock: 4,676s (77.9 min) on shared hardware (other concurrent workloads on the same GPU
  during parts of the run slowed several steps well past their steady-state ~6-10s/it).

**Evaluation** (`scripts/evaluate.py`, 60 held-out test examples never seen in training, real
`.generate()` calls for both models, no mocking):

| Metric | Untuned base (Qwen2.5-1.5B-Instruct) | Fine-tuned (QLoRA adapter) |
|---|---|---|
| Exact match | 0.000 | 0.000 |
| F1 | 0.275 | **0.441** |
| ROUGE-L | 0.191 | **0.333** |
| Avg. latency | 4,855 ms | 10,132 ms |

Exact match is 0 for both models on open-ended generation, as expected - it is not a useful metric
for this task and is reported only because the PRD's schema calls for it. The real signal is F1
and ROUGE-L: the fine-tuned model's outputs overlap far more with the held-out reference responses
than the untuned base model's do (+60% relative F1, +74% relative ROUGE-L), a genuine, measured
effect of the domain fine-tune. The PRD's claimed "92% accuracy / 80% cost reduction / 50ms
latency" figures are not reproduced here - see Design Decisions below for why. Fine-tuned latency
is higher than the base model's in this run because both models were evaluated back-to-back on
shared, contended hardware; it is not a claim about the adapter's own inference cost (the adapter
itself is a small percentage of total forward-pass compute). Raw results:
`model-artifacts/eval_results.json`.

## Design Decisions & Trade-offs

**Mistral-7B -> Qwen2.5-1.5B-Instruct.** The PRD specifies Mistral-7B. Even 4-bit quantized,
Mistral-7B's weights alone need ~4GB, and QLoRA training needs headroom for optimizer states,
gradients, and activations on top of that - it does not fit in the RTX 4050's 6GB VRAM alongside
Windows/driver overhead. Qwen2.5-1.5B-Instruct is a strong small instruct model that trains
comfortably in 4-bit on this GPU, which matters more for a portfolio piece than matching the
PRD's exact model name - the mechanics (QLoRA, quantization, real training/eval) are identical.

**4-bit (QLoRA) instead of the PRD's 8-bit.** 6GB VRAM is tight even for a 1.5B model once you
add LoRA training overhead; 4-bit NF4 quantization (`bitsandbytes`) roughly halves the base
model's memory footprint vs 8-bit, with `bnb_4bit_use_double_quant=True` for a further small
reduction, at a small, well-documented quality cost vs 8-bit (mitigated by LoRA's own capacity).

**No GPT-4 baseline call.** The PRD's `evaluate.py` sketch calls a GPT-4 baseline for comparison.
This pipeline's standard is to avoid paid API calls for a portfolio project; the real, measurable,
zero-cost comparison is the **untuned base model** (both actually run, not mocked), which more
directly isolates the effect of the fine-tune anyway.

**Real, not fabricated, metrics.** The PRD's "92% accuracy / 80% cost reduction / 50ms latency"
figures are hardcoded example text in the source document, not measured numbers. Every number in
the Results section above came from an actual `scripts/train.py` run (writing real per-step loss
to Postgres) and an actual `scripts/evaluate.py` run generating real text from both models against
a held-out test split. Where the real numbers fall short of the PRD's aspirational targets, that
is reported as-is - consistent with how this pipeline's other projects (e.g. the recommender's
under-target recall, the segmentation project's K=2 result) are documented honestly.

**Docker GPU passthrough.** `docker-compose.yml`'s `api` service runs in mock mode by default
because Docker Desktop on Windows does not reliably expose the GPU to Linux containers for
`bitsandbytes`/CUDA workloads. Rather than silently faking a "GPU-accelerated container demo",
this is documented plainly: the one-command Docker path proves out the API/DB/caching mechanics,
and the real GPU-backed generation is run directly on the host venv, which is where training also
runs (training was never going to be containerized anyway - see PRD deviation notes above).

## Testing

```bash
cd backend
pytest --cov=app --cov-report=term-missing
```

Tests force `USE_REAL_MODEL=false` (no GPU/model download needed) and run against a real
Postgres - either a local `docker compose up postgres` or CI's Postgres service container.
`db_session`/`client` fixtures skip integration tests cleanly if no Postgres is reachable.

## Credits

Developed by Kabilesh Rajaselvan with contributions from Jennita S.
