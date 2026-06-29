import React, { useEffect, useState } from "react";
import {
  getCorrelation,
  getDiagnostics,
  getHealth,
  getMetrics,
  getPriceHistory,
  getRuns,
  postPredict,
} from "./api.js";
import { Icon } from "./components.jsx";
import Forecast from "./tabs/Forecast.jsx";
import Comparison from "./tabs/Comparison.jsx";
import Diagnostics from "./tabs/Diagnostics.jsx";
import Experiments from "./tabs/Experiments.jsx";

const TABS = [
  ["forecast", "Forecast"],
  ["comparison", "Model Comparison"],
  ["diagnostics", "Training Diagnostics"],
  ["experiments", "Experiments"],
];

const DOCS_URL = `${import.meta.env.VITE_API_BASE || "https://obaiz-btc-forecasting-api.hf.space"}/docs`;

export default function App() {
  const [tab, setTab] = useState("forecast");
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [diagnostics, setDiagnostics] = useState(null);
  const [correlation, setCorrelation] = useState(null);
  const [priceHistory, setPriceHistory] = useState(null);
  const [runs, setRuns] = useState(null);

  const [result, setResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "down" }));
    getMetrics().then(setMetrics).catch(() => {});
    getDiagnostics().then(setDiagnostics).catch(() => {});
    getCorrelation().then(setCorrelation).catch(() => {});
    getPriceHistory().then(setPriceHistory).catch(() => {});
    getRuns().then((d) => setRuns(d.runs || [])).catch(() => setRuns([]));
  }, []);

  const onPredict = async (payload) => {
    setPredicting(true);
    setPredictError(null);
    try {
      setResult(await postPredict(payload));
    } catch (e) {
      setPredictError(e.message);
      setResult(null);
    } finally {
      setPredicting(false);
    }
  };

  const healthy = health && health.status === "ok";

  return (
    <div className="min-h-screen pb-28">
      {/* Top nav */}
      <nav className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-margin-mobile md:px-margin-desktop h-16 bg-surface/80 backdrop-blur-md border-b border-outline-variant">
        <div className="flex items-center gap-3">
          <Icon name="monitoring" className="text-primary text-2xl" />
          <span className="text-headline-md font-headline font-bold text-primary">QuantForecaster</span>
        </div>
        <a
          href={DOCS_URL}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 text-on-surface-variant hover:text-primary transition-colors text-label-md"
        >
          <Icon name="api" className="text-base" /> API Docs
        </a>
      </nav>

      <main className="max-w-container-max mx-auto mt-24 px-margin-mobile md:px-margin-desktop">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-stack-md">
          <div>
            <h1 className="text-display-lg font-headline gradient-text">BTC Price Forecasting</h1>
            <p className="text-on-surface-variant text-body-lg">LSTM (V1) vs GRU (V2) · MLOps Deployment</p>
          </div>
          <div className="flex items-center gap-2 bg-surface-container-high px-4 py-2 rounded-full border border-outline-variant">
            <span className={`w-2 h-2 rounded-full ${healthy ? "bg-green-500 animate-pulse" : "bg-error"}`} />
            <span className="font-mono text-label-md text-on-surface-variant">
              {health
                ? `API: ${health.status} · V2 ${health.v2_model_exists ? "✓" : "✗"} · scaler ${health.scaler_exists ? "✓" : "✗"}`
                : "connecting…"}
            </span>
          </div>
        </header>

        {/* Tabs */}
        <div className="flex items-center gap-8 border-b border-outline-variant mb-gutter overflow-x-auto whitespace-nowrap scrollbar-hide">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`pb-4 font-bold tracking-wide text-label-md uppercase transition-all ${tab === id ? "tab-active" : "tab-inactive"}`}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === "forecast" && (
          <Forecast
            onPredict={onPredict}
            result={result}
            predicting={predicting}
            predictError={predictError}
            priceHistory={priceHistory}
            health={health}
          />
        )}
        {tab === "comparison" && <Comparison metrics={metrics} correlation={correlation} />}
        {tab === "diagnostics" && <Diagnostics diagnostics={diagnostics} />}
        {tab === "experiments" && <Experiments runs={runs} />}
      </main>

      {/* Disclaimer */}
      <footer className="fixed bottom-0 left-0 w-full z-50 flex items-center justify-center gap-2 py-2 px-4 text-center bg-error-container text-on-error-container font-mono text-label-sm">
        <Icon name="warning" className="text-sm" />
        <span>Academic project for educational purposes only. Not financial advice. Crypto prices are highly volatile.</span>
      </footer>
    </div>
  );
}
