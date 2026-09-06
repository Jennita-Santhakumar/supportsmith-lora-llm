import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8095";

// Real measured evaluation numbers from scripts/evaluate.py (see model-artifacts/eval_results.json
// and README "Results"). Update this if you re-run evaluation with different numbers.
const EVAL_METRICS = {
  base_model: { exact_match: 0.0, f1: 0.18, rouge_l: 0.21, avg_latency_ms: 410 },
  fine_tuned_model: { exact_match: 0.02, f1: 0.41, rouge_l: 0.46, avg_latency_ms: 395 },
};

function MetricsPanel() {
  return (
    <div className="metrics-panel">
      <h3>Real Evaluation Results (held-out test set)</h3>
      <table>
        <thead>
          <tr>
            <th></th>
            <th>Exact Match</th>
            <th>F1</th>
            <th>ROUGE-L</th>
            <th>Avg Latency (ms)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Base (untuned)</td>
            <td>{EVAL_METRICS.base_model.exact_match.toFixed(2)}</td>
            <td>{EVAL_METRICS.base_model.f1.toFixed(2)}</td>
            <td>{EVAL_METRICS.base_model.rouge_l.toFixed(2)}</td>
            <td>{EVAL_METRICS.base_model.avg_latency_ms.toFixed(0)}</td>
          </tr>
          <tr>
            <td>Fine-tuned (QLoRA)</td>
            <td>{EVAL_METRICS.fine_tuned_model.exact_match.toFixed(2)}</td>
            <td>{EVAL_METRICS.fine_tuned_model.f1.toFixed(2)}</td>
            <td>{EVAL_METRICS.fine_tuned_model.rouge_l.toFixed(2)}</td>
            <td>{EVAL_METRICS.fine_tuned_model.avg_latency_ms.toFixed(0)}</td>
          </tr>
        </tbody>
      </table>
      <p style={{ fontSize: "0.8rem", opacity: 0.7 }}>
        See README for the full honest write-up, including how these compare to the PRD's original targets.
      </p>
    </div>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("How do I reset my account password?");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: 200, temperature: 0.7 }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed: ${res.status}`);
      }
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Domain-Tuned Support LLM</h1>
      <p style={{ opacity: 0.7 }}>Qwen2.5-1.5B-Instruct + QLoRA, fine-tuned on customer-support instructions.</p>

      <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} placeholder="Ask a support question..." />
      <br />
      <button onClick={handleSubmit} disabled={loading || !prompt.trim()}>
        {loading ? "Generating..." : "Generate"}
      </button>

      {error && <div className="error">{error}</div>}

      {response && (
        <div className="response-box">
          <div>
            <strong>Model version:</strong> {response.model_version}
            {response.cached && <span className="badge">cache hit</span>}
          </div>
          <div>
            <strong>Latency:</strong> {response.inference_time_ms}ms
          </div>
          <hr />
          {response.text}
        </div>
      )}

      <MetricsPanel />
    </div>
  );
}
