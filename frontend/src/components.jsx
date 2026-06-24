// Shared UI primitives + small formatting helpers for the dashboard.
import React from "react";

export const usd = (n) =>
  n == null || isNaN(n)
    ? "—"
    : n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });

export const num = (n, d = 2) =>
  n == null || isNaN(n) ? "—" : Number(n).toLocaleString("en-US", { maximumFractionDigits: d });

export function Icon({ name, className = "", ...rest }) {
  return (
    <span className={`material-symbols-outlined ${className}`} {...rest}>
      {name}
    </span>
  );
}

export function Card({ children, className = "" }) {
  return <section className={`glass-card rounded-2xl ${className}`}>{children}</section>;
}

export function SectionTitle({ title, subtitle, right }) {
  return (
    <div className="flex items-start justify-between mb-6 gap-4">
      <div>
        <h3 className="font-headline text-headline-md text-on-surface">{title}</h3>
        {subtitle && <p className="text-label-md text-on-surface-variant mt-0.5">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="flex gap-4 flex-wrap">
      {items.map((it) => (
        <div key={it.label} className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-full" style={{ background: it.color }} />
          <span className="text-label-sm text-on-surface-variant">{it.label}</span>
        </div>
      ))}
    </div>
  );
}

export function Spinner({ label }) {
  return (
    <div className="flex items-center justify-center gap-3 py-10 text-on-surface-variant">
      <span className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      {label && <span className="text-label-md">{label}</span>}
    </div>
  );
}

export function ErrorBox({ message }) {
  return (
    <div className="flex items-center gap-2 bg-error-container/20 text-error border border-error/30 rounded-xl px-4 py-3 text-label-md">
      <Icon name="error" className="text-base" />
      <span>{message}</span>
    </div>
  );
}

// Radial R² gauge (SVG). value in [0,1].
export function Gauge({ value, label, color = "#cabeff" }) {
  const r = 40;
  const circ = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value || 0));
  const offset = circ * (1 - pct);
  return (
    <div className="text-center">
      <div className="relative w-28 h-28 mb-3">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={r} fill="transparent" stroke="#36333e" strokeWidth="8" />
          <circle
            cx="50"
            cy="50"
            r={r}
            fill="transparent"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.8s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center font-bold text-xl font-mono">
          {value == null ? "—" : value.toFixed(2)}
        </div>
      </div>
      <span className="text-label-sm uppercase font-bold" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

// Correlation heatmap from {labels, matrix}.
export function Heatmap({ labels, matrix }) {
  const n = labels.length;
  const short = labels.map((l) => l.replace("Daily_Return", "Ret").replace("Volume", "Vol").replace("MA_14", "MA14"));
  return (
    <div className="grid grid-cols-[auto_1fr] gap-3 items-stretch">
      <div className="flex flex-col justify-between py-1 text-[10px] text-on-surface-variant uppercase font-bold gap-1 text-right">
        {short.map((l) => (
          <span key={l}>{l}</span>
        ))}
      </div>
      <div>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
          {matrix.flatMap((row, i) =>
            row.map((v, j) => {
              const a = Math.abs(v);
              return (
                <div
                  key={`${i}-${j}`}
                  title={`${labels[i]} × ${labels[j]} = ${v.toFixed(2)}`}
                  className="aspect-square rounded-sm flex items-center justify-center"
                  style={{ background: `rgba(202, 190, 255, ${0.08 + a * 0.92})` }}
                >
                  <span className="text-[8px] text-on-primary font-bold opacity-80 hidden md:block">
                    {v.toFixed(1)}
                  </span>
                </div>
              );
            })
          )}
        </div>
        <div className="grid mt-2 text-[10px] text-on-surface-variant uppercase font-bold text-center" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
          {short.map((l) => (
            <span key={l}>{l}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
