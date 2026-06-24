// Central API client. In dev, Vite proxies "/api" -> http://localhost:8000.
// Override with VITE_API_BASE (e.g. the Minikube NodePort URL) at build time.
const API_BASE = import.meta.env.VITE_API_BASE || "/api";

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

export const getHealth = () => request("/health");
export const getRoot = () => request("/");
export const postPredict = (payload) =>
  request("/predict", { method: "POST", body: JSON.stringify(payload) });
