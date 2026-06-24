import { useEffect, useState } from "react";
import { getHealth, postPredict } from "./api.js";

const DEFAULT_FORM = {
  open: 65000,
  high: 66000,
  low: 64000,
  close: 65500,
  volume: 32000000000,
  ma_14: 64800,
  rsi: 58.5,
  macd: 120.3,
  daily_return: 0.012,
};

const FIELDS = [
  ["open", "Open"],
  ["high", "High"],
  ["low", "Low"],
  ["close", "Close"],
  ["volume", "Volume"],
  ["ma_14", "MA 14"],
  ["rsi", "RSI"],
  ["macd", "MACD"],
  ["daily_return", "Daily Return"],
];

export default function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [modelVersion, setModelVersion] = useState("v2");
  const [useLatest, setUseLatest] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  const onChange = (key, value) =>
    setForm((f) => ({ ...f, [key]: value }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = useLatest
        ? { use_latest: true, model_version: modelVersion }
        : {
            ...Object.fromEntries(
              Object.entries(form).map(([k, v]) => [k, Number(v)])
            ),
            model_version: modelVersion,
          };
      const data = await postPredict(payload);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <h1>BTC Price Forecasting</h1>
        <p className="subtitle">LSTM (V1) vs GRU (V2) · MLOps Deployment</p>
        <HealthBadge health={health} />
      </header>

      <main className="grid">
        <form className="card" onSubmit={submit}>
          <h2>Prediction Input</h2>

          <div className="row">
            <label className="model-toggle">
              Model version
              <select
                value={modelVersion}
                onChange={(e) => setModelVersion(e.target.value)}
              >
                <option value="v2">V2 — GRU (improved)</option>
                <option value="v1">V1 — LSTM (baseline)</option>
              </select>
            </label>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={useLatest}
                onChange={(e) => setUseLatest(e.target.checked)}
              />
              Use latest live data (Yahoo Finance)
            </label>
          </div>

          {!useLatest && (
            <div className="fields">
              {FIELDS.map(([key, label]) => (
                <label key={key} className="field">
                  <span>{label}</span>
                  <input
                    type="number"
                    step="any"
                    value={form[key]}
                    onChange={(e) => onChange(key, e.target.value)}
                  />
                </label>
              ))}
            </div>
          )}

          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Predicting…" : "Predict next price"}
          </button>
          {error && <p className="error">⚠️ {error}</p>}
        </form>

        <ResultCard result={result} />
      </main>

      <footer className="disclaimer">
        ⚠️ Academic project for educational purposes only. <strong>Not financial
        advice.</strong> Cryptocurrency prices are highly volatile and
        unpredictable.
      </footer>
    </div>
  );
}

function HealthBadge({ health }) {
  if (!health) return <span className="badge">checking…</span>;
  const ok = health.status === "ok";
  return (
    <div className={`badge ${ok ? "ok" : "down"}`}>
      API: {health.status}
      {ok && (
        <>
          {" · "}V2 model: {health.v2_model_exists ? "✓" : "✗"}
          {" · "}scaler: {health.scaler_exists ? "✓" : "✗"}
        </>
      )}
    </div>
  );
}

function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="card placeholder">
        <h2>Result</h2>
        <p>Submit the form to see a prediction.</p>
      </div>
    );
  }
  const up = result.trend === "Up";
  return (
    <div className="card result">
      <h2>Result</h2>
      <div className="price">${result.predicted_price.toLocaleString()}</div>
      <div className={`trend ${up ? "up" : "down"}`}>
        {up ? "▲" : "▼"} {result.trend} · {result.pct_change}%
      </div>
      <ul className="meta">
        <li><span>Last close</span><b>${result.last_close.toLocaleString()}</b></li>
        <li><span>Change</span><b>${result.change.toLocaleString()}</b></li>
        <li><span>Signal</span><b>{result.confidence_signal}</b></li>
        <li><span>Model</span><b>{result.model_version}</b></li>
      </ul>
      <p className="note">{result.message}</p>
    </div>
  );
}
