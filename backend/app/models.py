"""SQLAlchemy models matching the PRD's schema (training_jobs, fine_tune_metrics,
inference_cache), with minor additive columns needed for real (non-mocked) use."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    model_name: Mapped[str] = mapped_column(String(255))
    base_model: Mapped[str] = mapped_column(String(255))
    dataset_size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    metrics: Mapped[list["FineTuneMetric"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class FineTuneMetric(Base):
    __tablename__ = "fine_tune_metrics"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("training_jobs.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    step: Mapped[int] = mapped_column(Integer, default=0)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    job: Mapped["TrainingJob"] = relationship(back_populates="metrics")


class InferenceCache(Base):
    __tablename__ = "inference_cache"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    input_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(50))
    inference_time_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
