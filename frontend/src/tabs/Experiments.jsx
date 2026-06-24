import React from "react";
import { Card, Icon, Spinner, num } from "../components.jsx";

const fmtTime = (s) => (s == null ? "—" : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

export default function Experiments({ runs }) {
  if (!runs) return <Card className="p-stack-lg"><Spinner label="Loading experiment runs…" /></Card>;

  return (
    <div className="flex flex-col gap-gutter">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="font-headline text-headline-md">Experiment History</h2>
          <p className="text-on-surface-variant text-body-md">MLflow tracking results (experiment: BTC_Price_Forecasting)</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-gutter">
        {runs.map((r, i) => (
          <RunCard key={r.name} run={r} champion={i === 0} />
        ))}
        <div className="glass-card rounded-2xl p-6 border-2 border-dashed border-outline-variant flex flex-col items-center justify-center text-on-surface-variant gap-3 min-h-[260px]">
          <Icon name="science" className="text-5xl opacity-60" />
          <div className="text-center">
            <span className="font-bold text-lg block">Reproduce</span>
            <span className="text-label-sm">python training/train_v2_gru.py</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function RunCard({ run, champion }) {
  const p = run.params || {};
  const m = run.metrics || {};
  return (
    <div className={`glass-card rounded-2xl p-6 relative overflow-hidden ${champion ? "border-l-4 border-primary" : "border-l-4 border-outline"}`}>
      {champion && (
        <div className="absolute top-0 right-0 p-2 opacity-10">
          <Icon name="verified" className="text-6xl" />
        </div>
      )}
      <div className="flex justify-between items-start mb-4">
        <div>
          <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${champion ? "text-primary bg-primary/20" : "text-on-surface-variant bg-surface-container-highest"}`}>
            {champion ? "Champion" : "Baseline"}
          </span>
          <h4 className="text-headline-md mt-1 font-headline">{run.name}</h4>
        </div>
        <span className="text-label-sm text-on-surface-variant opacity-60">{p.architecture?.includes("GRU") ? "GRU" : "LSTM"}</span>
      </div>

      <div className="space-y-4">
        <div className="bg-surface-container-low p-3 rounded-lg border border-outline-variant/20">
          <p className="text-[10px] text-on-surface-variant uppercase font-bold mb-2">Hyperparameters</p>
          <div className="grid grid-cols-2 gap-y-2 text-label-md font-mono">
            <KV k="Lookback" v={`${p.lookback ?? "—"}d`} />
            <KV k="LR" v={p.learning_rate ?? "—"} />
            <KV k="Epochs" v={p.epochs ?? "—"} />
            <KV k="Optimizer" v={p.optimizer ?? "—"} />
          </div>
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Metric k="R²" v={m.R2?.toFixed(2)} hi={champion} />
          <Metric k="RMSE" v={m.RMSE ? `$${num(m.RMSE, 0)}` : "—"} hi={champion} />
          <Metric k="Val Loss" v={run.val_loss ? run.val_loss.toExponential(1) : "—"} hi={champion} />
        </div>
        <div className="flex justify-between items-center pt-2 border-t border-outline-variant/20 text-label-sm text-on-surface-variant">
          <span>Train time: <b className="text-on-surface">{fmtTime(run.train_time_sec)}</b></span>
          <span className="font-mono text-[10px] truncate max-w-[120px]" title={p.architecture}>{p.architecture}</span>
        </div>
      </div>
    </div>
  );
}

const KV = ({ k, v }) => (
  <div className="flex flex-col">
    <span className="text-[10px] opacity-60">{k}</span>
    <span>{v}</span>
  </div>
);

const Metric = ({ k, v, hi }) => (
  <div className="flex flex-col">
    <span className="text-[10px] text-on-surface-variant uppercase font-bold">{k}</span>
    <span className={`font-mono font-bold ${hi ? "text-primary" : ""}`}>{v ?? "—"}</span>
  </div>
);
