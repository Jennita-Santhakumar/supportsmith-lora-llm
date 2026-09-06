from app.models import FineTuneMetric, TrainingJob


def test_job_and_metrics_roundtrip(client, db_session):
    job = TrainingJob(
        model_name="qwen2.5-1.5b-support-lora",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_size=2400,
        status="completed",
        accuracy=0.41,
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(job)

    metric = FineTuneMetric(job_id=job.id, epoch=1, step=100, loss=1.2, val_accuracy=0.35, learning_rate=2e-4)
    db_session.add(metric)
    db_session.commit()

    resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 1
    assert jobs[0]["dataset_size"] == 2400

    resp2 = client.get(f"/api/v1/jobs/{job.id}/metrics")
    assert resp2.status_code == 200
    metrics = resp2.json()
    assert len(metrics) == 1
    assert metrics[0]["step"] == 100


def test_job_metrics_404(client):
    resp = client.get("/api/v1/jobs/does-not-exist/metrics")
    assert resp.status_code == 404
