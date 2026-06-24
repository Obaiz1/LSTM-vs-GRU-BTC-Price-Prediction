import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api -> FastAPI on :8000 so there are no CORS surprises
// during local development. In production set VITE_API_BASE at build time.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
