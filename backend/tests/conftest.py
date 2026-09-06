"""Shared test fixtures.

Tests never load the real LLM (that would need a GPU and multi-GB downloads in CI). Instead
USE_REAL_MODEL is forced to false so app.inference.get_generator() returns the deterministic
MockGenerator - this exercises the real API/DB/caching code paths without mocking those out.
Postgres is real: tests read DATABASE_URL and skip automatically if it isn't reachable, so
`pytest` runs clean locally with nothing running, while CI (which starts a real Postgres service
container) runs everything.
"""

import os

os.environ["USE_REAL_MODEL"] = "false"
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://llm:llm@localhost:5511/llm_test")

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture(scope="session")
def database_url() -> str:
    return os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
def db_engine(database_url: str):
    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        pytest.skip(f"No reachable Postgres at {database_url}; skipping integration tests.")

    from app import models  # noqa: F401  (registers models on Base.metadata)
    from app.db import Base

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    from sqlalchemy.orm import sessionmaker

    session_factory = sessionmaker(bind=db_engine, future=True)
    session = session_factory()
    yield session
    session.rollback()
    for table in ("fine_tune_metrics", "inference_cache", "training_jobs"):
        session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
    session.commit()
    session.close()


@pytest.fixture
def client(db_session):
    from fastapi.testclient import TestClient

    from app.db import get_db
    from app.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
