"""FastAPI serving app: /health, /api/v1/generate, /api/v1/jobs."""

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Base, engine, get_db
from app.inference import get_generator, input_hash
from app.models import FineTuneMetric, InferenceCache, TrainingJob
from app.schemas import GenerateRequest, GenerateResponse, JobOut, MetricOut

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Domain-Tuned Support LLM Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, db: Session = Depends(get_db)):
    key = input_hash(req.prompt, req.max_tokens, req.temperature)

    cached = db.execute(select(InferenceCache).where(InferenceCache.input_hash == key)).scalar_one_or_none()
    if cached is not None:
        return GenerateResponse(
            text=cached.output,
            tokens_used=len(cached.output.split()),
            model_version=cached.model_version,
            inference_time_ms=cached.inference_time_ms,
            cached=True,
        )

    generator = get_generator()
    try:
        result = generator.generate(req.prompt, req.max_tokens, req.temperature)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = InferenceCache(
        input_hash=key,
        prompt=req.prompt,
        output=result.text,
        model_version=settings.model_version,
        inference_time_ms=result.inference_time_ms,
    )
    db.add(row)
    db.commit()

    return GenerateResponse(
        text=result.text,
        tokens_used=result.tokens_used,
        model_version=settings.model_version,
        inference_time_ms=result.inference_time_ms,
        cached=False,
    )


@app.get("/api/v1/jobs", response_model=list[JobOut])
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.execute(select(TrainingJob).order_by(TrainingJob.created_at.desc())).scalars().all()
    return jobs


@app.get("/api/v1/jobs/{job_id}/metrics", response_model=list[MetricOut])
def job_metrics(job_id: str, db: Session = Depends(get_db)):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="job not found") from None

    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    metrics = (
        db.execute(select(FineTuneMetric).where(FineTuneMetric.job_id == job_id).order_by(FineTuneMetric.step))
        .scalars()
        .all()
    )
    return metrics
