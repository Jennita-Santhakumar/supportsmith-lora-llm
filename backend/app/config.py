"""App settings, loaded from environment variables (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    database_url: str = "postgresql+psycopg://llm:llm@localhost:5432/llm"
    cors_origins: str = "http://localhost:5250"

    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_dir: str = "../model-artifacts/qwen2.5-1.5b-support-lora"
    model_version: str = "qwen2.5-1.5b-support-lora-v1"

    # When true (default), loads the base model + LoRA adapter in 4-bit (bitsandbytes) on CUDA.
    # Set to false to run in "mock" mode (used by CI / tests / any machine without a CUDA GPU) -
    # a deterministic canned-response generator is used instead so the API surface, caching and
    # DB plumbing can be exercised without a GPU or the multi-GB model download.
    use_real_model: bool = True

    max_new_tokens_default: int = 256


@lru_cache
def get_settings() -> Settings:
    return Settings()
