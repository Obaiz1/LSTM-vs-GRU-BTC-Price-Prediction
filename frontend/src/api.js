// Central API client. In dev, Vite proxies "/api" -> http://localhost:8000.
// In production (Vercel), calls the HuggingFace Space backend directly.
const API_BASE = import.meta.env.VITE_API_BASE || "https://obaiz-btc-forecasting-api.hf.space";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export const getRoot = () => request("/");
export const getHealth = () => request("/health");
export const postPredict = (payload) =>
  request("/predict", { method: "POST", body: JSON.stringify(payload) });

// Analytics endpoints (dashboard charts)
export const getMetrics = () => request("/metrics");
export const getDiagnostics = () => request("/diagnostics");
export const getCorrelation = () => request("/correlation");
export const getPriceHistory = () => request("/price-history");
export const getRuns = () => request("/runs");
