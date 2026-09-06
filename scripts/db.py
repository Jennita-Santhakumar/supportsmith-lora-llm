"""Minimal DB helper shared by scripts/train.py so training can write real progress rows to
Postgres without depending on the backend package (scripts run in a separate venv)."""

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://llm:llm@localhost:5511/llm")

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY,
    model_name VARCHAR(255),
    base_model VARCHAR(255),
    dataset_size INT,
    status VARCHAR(50),
    accuracy FLOAT,
    created_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS fine_tune_metrics (
    id UUID PRIMARY KEY,
    job_id UUID REFERENCES training_jobs(id),
    epoch INT,
    step INT DEFAULT 0,
    loss FLOAT,
    val_accuracy FLOAT,
    learning_rate FLOAT,
    timestamp TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS inference_cache (
    id UUID PRIMARY KEY,
    input_hash VARCHAR(255) UNIQUE,
    prompt TEXT,
    output TEXT,
    model_version VARCHAR(50),
    inference_time_ms INT,
    created_at TIMESTAMPTZ
);
"""


def get_engine():
    return create_engine(DATABASE_URL)


def ensure_schema(engine):
    with engine.begin() as conn:
        for stmt in SCHEMA.strip().split(";\n\n"):
            if stmt.strip():
                conn.execute(text(stmt))


def create_job(engine, model_name: str, base_model: str, dataset_size: int) -> str:
    job_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO training_jobs (id, model_name, base_model, dataset_size, status, created_at) "
                "VALUES (:id, :model_name, :base_model, :dataset_size, 'running', :created_at)"
            ),
            {
                "id": job_id,
                "model_name": model_name,
                "base_model": base_model,
                "dataset_size": dataset_size,
                "created_at": datetime.now(timezone.utc),
            },
        )
    return job_id


def log_metric(engine, job_id: str, epoch: int, step: int, loss, val_accuracy, learning_rate):
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO fine_tune_metrics (id, job_id, epoch, step, loss, val_accuracy, learning_rate, timestamp) "
                "VALUES (:id, :job_id, :epoch, :step, :loss, :val_accuracy, :lr, :ts)"
            ),
            {
                "id": str(uuid.uuid4()),
                "job_id": job_id,
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "val_accuracy": val_accuracy,
                "lr": learning_rate,
                "ts": datetime.now(timezone.utc),
            },
        )


def complete_job(engine, job_id: str, accuracy):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE training_jobs SET status = 'completed', accuracy = :accuracy WHERE id = :id"),
            {"accuracy": accuracy, "id": job_id},
        )


def fail_job(engine, job_id: str):
    with engine.begin() as conn:
        conn.execute(text("UPDATE training_jobs SET status = 'failed' WHERE id = :id"), {"id": job_id})
