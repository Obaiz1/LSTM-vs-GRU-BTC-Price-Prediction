import React, { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Legend, SectionTitle, Spinner, usd } from "../components.jsx";

export default function Diagnostics({ diagnostics }) {
  const [model, setModel] = useState("v2");
  const d = diagnostics?.[model];

  const lossData = d?.history
    ? d.history.loss.map((l, i) => ({ epoch: i + 1, train: l, val: d.history.val_loss[i] }))
    : [];
  const evalData = d?.eval
    ? d.eval.y_true.map((t, i) => ({ idx: d.eval.dates?.[i] || i, actual: t, predicted: d.eval.y_pred[i] }))
    : [];

  return (
    <div className="flex flex-col gap-gutter">
      <div className="flex items-center gap-3">
        <span className="text-label-md text-on-surface-variant">Model:</span>
        {[["v2", "V2 GRU"], ["v1", "V1 LSTM"]].map(([k, label]) => (
          <button
            key={k}
            onClick={() => setModel(k)}
            className={`px-4 py-1.5 rounded-full text-label-md font-bold border transition-colors ${
              model === k ? "bg-primary-container text-on-primary border-transparent" : "border-outline-variant text-on-surface-variant hover:text-primary"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-gutter">
        {/* Loss curves */}
        <Card className="p-stack-lg">
          <SectionTitle
            title="Training vs Validation Loss"
            subtitle={d?.history ? `MSE across ${d.history.epochs} epochs` : "—"}
            right={<Legend items={[{ label: "Train", color: "#a4c8ff" }, { label: "Validation", color: "#cabeff" }]} />}
          />
          <div className="h-[340px] chart-grid rounded-xl border border-outline-variant/20 p-2">
            {!diagnostics ? (
              <Spinner label="Loading…" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" opacity={0.3} />
                  <XAxis dataKey="epoch" tick={{ fill: "#938ea1", fontSize: 10 }} />
                  <YAxis tick={{ fill: "#938ea1", fontSize: 10 }} width={60} tickFormatter={(v) => v.toExponential(0)} />
                  <Tooltip formatter={(v) => Number(v).toExponential(2)} />
                  <Line type="monotone" dataKey="train" stroke="#a4c8ff" strokeWidth={1.6} dot={false} />
                  <Line type="monotone" dataKey="val" stroke="#cabeff" strokeWidth={2.4} dot={false} />
                  {d?.history?.early_stop_epoch && (
                    <ReferenceLine x={d.history.early_stop_epoch} stroke="#ffb4ab" strokeDasharray="4 4" label={{ value: `best @ ${d.history.early_stop_epoch}`, fill: "#ffb4ab", fontSize: 10, position: "top" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>

        {/* Prediction vs Actual */}
        <Card className="p-stack-lg">
          <SectionTitle
            title="Prediction vs Actual (Test Set)"
            subtitle={d?.eval ? `${d.eval.n_test} hold-out days` : "—"}
            right={<Legend items={[{ label: "Actual", color: "#a4c8ff" }, { label: "Predicted", color: "#cabeff" }]} />}
          />
          <div className="h-[340px] chart-grid rounded-xl border border-outline-variant/20 p-2">
            {!diagnostics ? (
              <Spinner label="Loading…" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={evalData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" opacity={0.3} />
                  <XAxis dataKey="idx" tick={{ fill: "#938ea1", fontSize: 10 }} minTickGap={50} />
                  <YAxis tick={{ fill: "#938ea1", fontSize: 10 }} width={48} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} domain={["auto", "auto"]} />
                  <Tooltip formatter={(v) => usd(v)} />
                  <Line type="monotone" dataKey="actual" stroke="#a4c8ff" strokeWidth={1.6} dot={false} />
                  <Line type="monotone" dataKey="predicted" stroke="#cabeff" strokeWidth={2.2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
