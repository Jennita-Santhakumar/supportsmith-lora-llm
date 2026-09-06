"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    text: str
    tokens_used: int
    model_version: str
    inference_time_ms: int
    cached: bool


class JobOut(BaseModel):
    id: str
    model_name: str
    base_model: str
    dataset_size: int
    status: str
    accuracy: float | None
    created_at: datetime

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class MetricOut(BaseModel):
    epoch: int
    step: int
    loss: float | None
    val_accuracy: float | None
    learning_rate: float | None
    timestamp: datetime

    model_config = {"from_attributes": True}
