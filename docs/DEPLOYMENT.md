# Deployment — Frontend on Vercel + Backend on Hugging Face Spaces

The project is two pieces:

| Piece | Tech | Where it goes | Why |
|-------|------|---------------|-----|
| **Frontend** (`frontend/`) | React + Vite static site | **Vercel** | free static hosting, instant |
| **Backend** (`deployment/api.py`) | FastAPI + TensorFlow (~1 GB) | **Hugging Face Spaces** | 16 GB RAM free — fits TensorFlow (Render/most free tiers OOM) |

The frontend calls the backend via the `VITE_API_BASE` environment variable.
CORS is already open (`allow_origins=["*"]`) so cross-origin calls work.

---

## Step 1 — Deploy the backend to a Hugging Face Space

1. Create a **write** token at <https://huggingface.co/settings/tokens>.
2. Run the deploy script (uploads the repo so the Space builds the `Dockerfile`):

   ```bash
   pip install huggingface_hub
   # bash
   export HF_TOKEN=hf_xxx
   python deploy/huggingface/deploy_space.py --owner <hf-username> --name btc-forecasting-api
   # PowerShell
   $env:HF_TOKEN="hf_xxx"; python deploy/huggingface/deploy_space.py --owner <hf-username> --name btc-forecasting-api
   ```

3. The Space builds for ~5–10 min (installing TensorFlow). When green, your API is at:

   ```
   https://<hf-username>-btc-forecasting-api.hf.space
   ```

   Verify: open `…/docs` and `…/health` in the browser.

> Alternative (no script): create a **Docker** Space in the HF UI, then upload the
> repo files + copy `deploy/huggingface/README.md` as the Space's `README.md`.

---

## Step 2 — Deploy the frontend to Vercel

1. Import the GitHub repo at <https://vercel.com/new>.
2. The repo's root `vercel.json` already builds `frontend/` → `frontend/dist`,
   so you do **not** need to change the Root Directory.
3. Add an **Environment Variable** (Project → Settings → Environment Variables):

   ```
   VITE_API_BASE = https://<hf-username>-btc-forecasting-api.hf.space
   ```

4. **Deploy**. Your dashboard is live at `https://<project>.vercel.app`.

> Changing `VITE_API_BASE` later requires a **redeploy** (Vite inlines env vars at
> build time).

---

## Local development

```bash
# backend
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
# frontend (proxies /api -> :8000)
cd frontend && npm install && npm run dev
```

Academic project — **not financial advice**.
