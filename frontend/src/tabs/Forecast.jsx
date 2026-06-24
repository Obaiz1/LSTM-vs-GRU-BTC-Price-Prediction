import React, { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, ErrorBox, Icon, Legend, SectionTitle, Spinner, usd } from "../components.jsx";

const FIELDS = [
  ["open", "Open ($)", 64821.5],
  ["high", "High ($)", 66120.0],
  ["low", "Low ($)", 63900.2],
  ["close", "Close ($)", 65400.1],
  ["volume", "Volume (24h)", 32000000000],
  ["ma_14", "MA 14", 65100.8],
  ["rsi", "RSI", 58.4],
  ["macd", "MACD", 142.1],
  ["daily_return", "Daily Return", 0.0085],
];

export default function Forecast({ onPredict, result, predicting, predictError, priceHistory, health }) {
  const [version, setVersion] = useState("v2");
  const [useLatest, setUseLatest] = useState(false);
  const [values, setValues] = useState(Object.fromEntries(FIELDS.map(([k, , v]) => [k, v])));

  const submit = () => {
    const payload = { model_version: version, use_latest: useLatest };
    if (!useLatest) FIELDS.forEach(([k]) => (payload[k] = Number(values[k])));
    onPredict(payload);
  };

  const chartData = useMemo(() => {
    if (!priceHistory?.dates) return [];
    const base = priceHistory.dates.map((d, i) => ({
      date: d,
      actual: priceHistory.close[i],
    }));
    if (result?.predicted_price != null && base.length) {
      base[base.length - 1] = { ...base[base.length - 1], predicted: base[base.length - 1].actual };
      base.push({ date: "T+1", predicted: result.predicted_price });
    }
    return base;
  }, [priceHistory, result]);

  const up = result?.trend === "Up";

  return (
    <div className="flex flex-col gap-gutter">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter">
        {/* Prediction Input */}
        <Card className="lg:col-span-8 p-stack-lg flex flex-col gap-stack-md">
          <div className="flex items-center justify-between">
            <h2 className="font-headline text-headline-md">Prediction Input</h2>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <span className="text-label-md text-on-surface-variant">Use latest live data</span>
              <input type="checkbox" className="sr-only peer" checked={useLatest} onChange={(e) => setUseLatest(e.target.checked)} />
              <span className="w-10 h-5 bg-outline-variant peer-checked:bg-primary-container rounded-full relative transition-colors after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4 after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-5" />
            </label>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-stack-md">
            <div className="flex flex-col gap-base">
              <label className="text-label-md text-on-surface-variant">Architecture</label>
              <select value={version} onChange={(e) => setVersion(e.target.value)} className="bg-surface-container-low border border-outline-variant rounded-lg px-4 py-2 text-on-surface focus:ring-2 focus:ring-primary outline-none">
                <option value="v2">V2 — GRU (improved)</option>
                <option value="v1">V1 — LSTM (baseline)</option>
              </select>
            </div>
            <div className="flex flex-col gap-base">
              <label className="text-label-md text-on-surface-variant">Time Horizon</label>
              <select className="bg-surface-container-low border border-outline-variant rounded-lg px-4 py-2 text-on-surface outline-none" disabled>
                <option>Next Day (T+1)</option>
              </select>
            </div>
          </div>

          <div className={`grid grid-cols-2 md:grid-cols-3 gap-stack-md mt-2 transition-opacity ${useLatest ? "opacity-40 pointer-events-none" : ""}`}>
            {FIELDS.map(([key, label]) => (
              <div key={key} className="flex flex-col gap-base">
                <label className="text-label-sm text-on-surface-variant">{label}</label>
                <input
                  type="number"
                  step="any"
                  value={values[key]}
                  onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
                  className="bg-surface-container-low border border-outline-variant rounded-lg px-4 py-2 text-on-surface focus:ring-2 focus:ring-primary outline-none"
                />
              </div>
            ))}
          </div>

          {useLatest && (
            <p className="text-label-sm text-on-surface-variant italic">
              ℹ️ Fetches the latest 60 days from Yahoo Finance — manual fields are ignored.
            </p>
          )}
          <p className="text-label-sm text-on-surface-variant/70 italic">
            ℹ️ The model needs a 60-day sequence; a single row is appended to the most recent 59 live days.
          </p>

          <button
            onClick={submit}
            disabled={predicting}
            className="gradient-btn mt-2 py-4 rounded-xl font-bold text-white shadow-lg flex items-center justify-center gap-2 active:scale-[0.99] transition-transform"
          >
            {predicting ? (
              <span className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Icon name="analytics" />
            )}
            {predicting ? "Predicting…" : "Predict next price"}
          </button>
        </Card>

        {/* Result */}
        <Card className="lg:col-span-4 p-stack-lg relative overflow-hidden">
          <div className="absolute -right-12 -top-12 w-32 h-32 bg-primary/20 blur-3xl rounded-full" />
          <h2 className="text-label-md text-on-surface-variant uppercase tracking-widest mb-4">Latest Prediction</h2>
          {predictError && <ErrorBox message={predictError} />}
          {!predictError && !result && (
            <p className="text-on-surface-variant text-label-md py-8">Run a prediction to see results.</p>
          )}
          {result && (
            <>
              <div className="text-display-lg font-headline text-primary leading-tight">{usd(result.predicted_price)}</div>
              <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full font-label-md self-start mt-2 ${up ? "bg-green-500/15 text-green-400" : "bg-error-container/20 text-error"}`}>
                <Icon name={up ? "trending_up" : "trending_down"} className="text-sm" />
                {up ? "▲ Up" : "▼ Down"} · {result.pct_change?.toFixed(2)}%
              </span>
              <div className="mt-8 flex flex-col gap-4 border-t border-outline-variant pt-6">
                <Row label="Last close" value={usd(result.last_close)} />
                <Row label="Expected change" value={usd(result.change)} accent={up ? "text-green-400" : "text-error"} />
                <Row label="Signal" value={<span className="bg-surface-container-high px-2 py-0.5 rounded">{result.confidence_signal}</span>} />
                <Row label="Model" value={<span className="font-mono text-primary">{result.model_version}</span>} />
              </div>
              <p className="mt-6 text-label-sm text-on-surface-variant italic opacity-60">{result.message}</p>
            </>
          )}
        </Card>
      </div>

      {/* Price History & Forecast */}
      <Card className="p-stack-lg">
        <SectionTitle
          title="Price History & Forecast"
          subtitle="~90-day window · predicted next close"
          right={<Legend items={[{ label: "Actual", color: "#a4c8ff" }, { label: "Predicted", color: "#cabeff" }]} />}
        />
        <div className="h-[360px] w-full chart-grid rounded-xl border border-outline-variant/30 p-2">
          {!priceHistory ? (
            <Spinner label="Loading price history…" />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                <defs>
                  <linearGradient id="actualFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a4c8ff" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="#a4c8ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" opacity={0.3} />
                <XAxis dataKey="date" tick={{ fill: "#938ea1", fontSize: 10 }} minTickGap={40} />
                <YAxis tick={{ fill: "#938ea1", fontSize: 10 }} domain={["auto", "auto"]} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} width={48} />
                <Tooltip formatter={(v) => usd(v)} labelStyle={{ color: "#c9c4d8" }} />
                <Area type="monotone" dataKey="actual" stroke="#a4c8ff" strokeWidth={2} fill="url(#actualFill)" connectNulls />
                <Line type="monotone" dataKey="predicted" stroke="#cabeff" strokeWidth={3} strokeDasharray="6 4" dot={false} connectNulls />
                {result?.predicted_price != null && (
                  <ReferenceDot x="T+1" y={result.predicted_price} r={6} fill="#cabeff" stroke="#fff" />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      </Card>
    </div>
  );
}

function Row({ label, value, accent = "" }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-on-surface-variant">{label}</span>
      <span className={`font-mono ${accent}`}>{value}</span>
    </div>
  );
}
