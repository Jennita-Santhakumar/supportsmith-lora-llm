from app.inference import input_hash


def test_generate_returns_text(client):
    resp = client.post("/api/v1/generate", json={"prompt": "How do I reset my password?", "max_tokens": 64})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"]
    assert body["cached"] is False
    assert body["model_version"]
    assert body["inference_time_ms"] >= 0


def test_generate_caches_on_retry(client):
    payload = {"prompt": "How do I cancel my subscription?", "max_tokens": 64, "temperature": 0.7}

    first = client.post("/api/v1/generate", json=payload).json()
    assert first["cached"] is False

    second = client.post("/api/v1/generate", json=payload).json()
    assert second["cached"] is True
    assert second["text"] == first["text"]


def test_generate_validates_empty_prompt(client):
    resp = client.post("/api/v1/generate", json={"prompt": ""})
    assert resp.status_code == 422


def test_input_hash_is_deterministic():
    a = input_hash("hello", 64, 0.7)
    b = input_hash("hello", 64, 0.7)
    c = input_hash("hello", 64, 0.9)
    assert a == b
    assert a != c


def test_jobs_endpoint_empty(client):
    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200
    assert resp.json() == []
