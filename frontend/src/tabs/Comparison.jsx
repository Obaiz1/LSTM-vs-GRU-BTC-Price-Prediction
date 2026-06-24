import React from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card, Gauge, Heatmap, Icon, Legend, SectionTitle, Spinner, num } from "../components.jsx";

export default function Comparison({ metrics, correlation }) {
  const m1 = metrics?.v1?.metrics;
  const m2 = metrics?.v2?.metrics;

  const barData = m1 && m2
    ? [
        { name: "MAE", v1: m1.MAE, v2: m2.MAE },
        { name: "RMSE", v1: m1.RMSE, v2: m2.RMSE },
      ]
    : [];

  return (
    <div className="flex flex-col gap-gutter">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-gutter">
        {/* Bars */}
        <Card className="lg:col-span-8 p-stack-lg">
          <SectionTitle
            title="Error Comparison (MAE & RMSE)"
            subtitle="Lower is better — USD error on the hold-out test set"
            right={<Legend items={[{ label: "V1 LSTM", color: "#a4c8ff" }, { label: "V2 GRU", color: "#947dff" }]} />}
          />
          <div className="h-[320px]">
            {!metrics ? (
              <Spinner label="Loading metrics…" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} margin={{ top: 20, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" opacity={0.3} vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#c9c4d8", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#938ea1", fontSize: 10 }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} width={48} />
                  <Tooltip formatter={(v) => `$${num(v)}`} cursor={{ fill: "#ffffff08" }} />
                  <Bar dataKey="v1" name="V1 LSTM" fill="#a4c8ff" radius={[6, 6, 0, 0]} maxBarSize={70}>
                    <LabelList dataKey="v1" position="top" fill="#c9c4d8" fontSize={11} formatter={(v) => num(v, 0)} />
                  </Bar>
                  <Bar dataKey="v2" name="V2 GRU" fill="#947dff" radius={[6, 6, 0, 0]} maxBarSize={70}>
                    <LabelList dataKey="v2" position="top" fill="#cabeff" fontSize={11} formatter={(v) => num(v, 0)} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
          {metrics?.verdict && (
            <div className="mt-4 flex items-center gap-2 bg-primary/5 border border-primary/20 rounded-xl px-4 py-3 text-label-md text-on-surface-variant">
              <Icon name="verified" className="text-primary text-base" />
              {metrics.verdict}
            </div>
          )}
        </Card>

        {/* R2 gauges */}
        <Card className="lg:col-span-4 p-stack-lg flex flex-col justify-center items-center gap-stack-lg">
          <h3 className="text-label-md uppercase tracking-widest text-on-surface-variant text-center">R² Score (Accuracy)</h3>
          <div className="flex justify-around w-full">
            <Gauge value={m1?.R2} label="V1 LSTM" color="#a4c8ff" />
            <Gauge value={m2?.R2} label="V2 GRU" color="#cabeff" />
          </div>
          <p className="text-label-sm text-center italic text-on-surface-variant/60 px-4">
            Closer to 1.0 = more variance explained.
          </p>
          {m1 && m2 && (
            <div className="w-full grid grid-cols-2 gap-3 text-center">
              <Stat label="V1 Dir. Acc" value={`${(m1.directional_accuracy * 100).toFixed(1)}%`} />
              <Stat label="V2 Dir. Acc" value={`${(m2.directional_accuracy * 100).toFixed(1)}%`} />
            </div>
          )}
        </Card>
      </div>

      {/* Metrics table */}
      {m1 && m2 && (
        <Card className="p-stack-lg overflow-x-auto">
          <SectionTitle title="Metrics Table" subtitle="Best value per row highlighted" />
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-outline-variant text-label-md text-on-surface-variant uppercase">
                <th className="py-3">Metric</th>
                <th className="py-3">V1 LSTM</th>
                <th className="py-3 text-primary">V2 GRU</th>
              </tr>
            </thead>
            <tbody className="font-mono text-body-md">
              <MetricRow name="MAE (USD)" a={m1.MAE} b={m2.MAE} lowerBetter fmt={(v) => `$${num(v)}`} />
              <MetricRow name="RMSE (USD)" a={m1.RMSE} b={m2.RMSE} lowerBetter fmt={(v) => `$${num(v)}`} />
              <MetricRow name="R²" a={m1.R2} b={m2.R2} fmt={(v) => v.toFixed(4)} />
              <MetricRow name="Directional Acc." a={m1.directional_accuracy} b={m2.directional_accuracy} fmt={(v) => `${(v * 100).toFixed(1)}%`} />
            </tbody>
          </table>
        </Card>
      )}

      {/* Heatmap */}
      <Card className="p-stack-lg">
        <SectionTitle
          title="Feature Correlation Matrix"
          subtitle="Absolute correlation · purple = stronger"
          right={
            <div className="flex items-center gap-3">
              <span className="text-[10px] text-on-surface-variant">0.0</span>
              <div className="w-32 h-2 rounded-full" style={{ background: "linear-gradient(to right, rgba(202,190,255,0.08), #cabeff)" }} />
              <span className="text-[10px] text-on-surface-variant">1.0</span>
            </div>
          }
        />
        {!correlation ? <Spinner label="Loading correlation…" /> : <Heatmap labels={correlation.labels} matrix={correlation.matrix} />}
      </Card>

      {/* Architecture deep dive */}
      <Card className="p-stack-lg">
        <h3 className="font-headline text-headline-md mb-8">Architecture Deep Dive: LSTM vs GRU</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-gutter mb-8">
          <ArchCard
            tag="LSTM" title="Long Short-Term Memory" subtitle="More expressive, heavier" color="#a4c8ff" tint="bg-secondary/10 border-secondary/20"
            points={[["account_tree", "3 gates: Forget, Input, Output regulate information flow."], ["memory", "Maintains a separate Cell State (cₜ) as long-term memory."]]}
            gates={["Forget", "Input", "Output"]}
          />
          <ArchCard
            tag="GRU" title="Gated Recurrent Unit" subtitle="Efficient & faster — our champion" color="#cabeff" tint="bg-primary/5 border-primary/20"
            points={[["dynamic_feed", "Simplified to 2 gates: Reset and Update."], ["speed", "Combines cell + hidden into a single Hidden State (hₜ)."]]}
            gates={["Reset", "Update"]}
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-outline-variant text-label-md text-on-surface-variant uppercase">
                <th className="py-3">Property</th>
                <th className="py-3">LSTM (V1)</th>
                <th className="py-3 text-primary">GRU (V2 Champion)</th>
              </tr>
            </thead>
            <tbody className="text-body-md">
              <ArchRow p="Gate count" a="3 gates" b="2 gates" />
              <ArchRow p="Internal memory" a="Cell + Hidden state" b="Hidden state only" />
              <ArchRow p="Parameters" a="More" b="Fewer / efficient" />
              <ArchRow p="Training" a="Slower per epoch" b="Faster convergence" />
              <ArchRow p="Best for" a="Long dependencies" b="Smaller / volatile series" />
              <ArchRow p="Final R²" a={m1 ? m1.R2.toFixed(2) : "—"} b={m2 ? m2.R2.toFixed(2) : "—"} />
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="bg-surface-container-low rounded-lg py-2 border border-outline-variant/20">
      <div className="text-[10px] text-on-surface-variant uppercase">{label}</div>
      <div className="font-mono font-bold">{value}</div>
    </div>
  );
}

function MetricRow({ name, a, b, lowerBetter = false, fmt }) {
  const aWins = lowerBetter ? a < b : a > b;
  const cls = "py-3";
  const win = "text-primary font-bold";
  return (
    <tr className="border-b border-outline-variant/20">
      <td className={cls}>{name}</td>
      <td className={`${cls} ${aWins ? win : "text-on-surface-variant"}`}>{fmt(a)}</td>
      <td className={`${cls} ${!aWins ? win : "text-on-surface-variant"}`}>{fmt(b)}</td>
    </tr>
  );
}

function ArchCard({ tag, title, subtitle, color, tint, points, gates }) {
  return (
    <div className={`p-6 rounded-xl border ${tint}`}>
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-lg flex items-center justify-center font-bold" style={{ background: `${color}22`, color }}>{tag}</div>
        <div>
          <h4 className="font-bold text-lg leading-tight">{title}</h4>
          <span className="text-label-sm" style={{ color }}>{subtitle}</span>
        </div>
      </div>
      <div className="space-y-4 mb-6">
        {points.map(([icon, text]) => (
          <div key={text} className="flex items-start gap-3">
            <Icon name={icon} className="text-base" style={{ color }} />
            <p className="text-body-md text-on-surface-variant">{text}</p>
          </div>
        ))}
      </div>
      <div className="flex gap-3 flex-wrap">
        {gates.map((g) => (
          <div key={g} className="px-3 h-8 rounded flex items-center justify-center text-[11px]" style={{ border: `1px solid ${color}`, color }}>{g}</div>
        ))}
      </div>
    </div>
  );
}

function ArchRow({ p, a, b }) {
  return (
    <tr className="border-b border-outline-variant/20">
      <td className="py-3 font-medium">{p}</td>
      <td className="py-3 text-on-surface-variant">{a}</td>
      <td className="py-3 text-primary font-bold">{b}</td>
    </tr>
  );
}
