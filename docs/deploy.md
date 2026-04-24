# Deployment Guide — Deep Hedging Lab

## Architecture

```
Vercel (Next.js frontend)  ──HTTPS──▶  Railway (FastAPI backend)
```

The frontend is a static Next.js app deployed on Vercel.
The backend is a FastAPI service deployed on Railway, running uvicorn.

---

## Required environment variables

### Backend (Railway)

| Variable | Description | Example |
|---|---|---|
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `https://deep-hedging-lab.vercel.app` |
| `PORT` | Bind port (set automatically by Railway) | _(auto)_ |

### Frontend (Vercel)

| Variable | Description | Example |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Full URL of the Railway backend | `https://deep-hedging-api.up.railway.app` |

---

## Backend deployment (Railway)

### 1. Create a Railway project

1. Go to [railway.app](https://railway.app) and create a new project.
2. Connect your GitHub repo (or push via Railway CLI).
3. Railway auto-detects Python via `pyproject.toml` / `requirements.txt`.

### 2. Set environment variables

In the Railway service dashboard → **Variables**:

```
ALLOWED_ORIGINS=https://<your-vercel-project>.vercel.app
```

Railway sets `PORT` automatically — do not set it manually.

### 3. Deploy

Railway uses `railway.toml` at the repo root:

```toml
[deploy]
startCommand = "uvicorn services.sim.main:app --host 0.0.0.0 --port $PORT"
```

The first deploy installs `requirements.txt` (includes `fastapi`, `uvicorn`, `torch`, etc.)
and then starts uvicorn.

> **Note on build time:** `torch` is a large dependency (~800 MB). The first build takes
> several minutes. Subsequent deploys are cached by Nixpacks.

### 4. Verify

Visit `https://<your-railway-service>.up.railway.app/` — you should see:

```json
{"status": "ok", "service": "deep-hedging-lab-api", "version": "0.1.0"}
```

---

## Frontend deployment (Vercel)

### 1. Import project

1. Go to [vercel.com](https://vercel.com) → **Add New Project**.
2. Import the GitHub repo.
3. Set **Root Directory** to `apps/web`.
4. Framework auto-detected as **Next.js**.

### 2. Set environment variables

In Vercel → **Settings** → **Environment Variables**:

```
NEXT_PUBLIC_API_URL = https://<your-railway-service>.up.railway.app
```

Add this for **Production**, **Preview**, and **Development** environments.

### 3. Deploy

Click **Deploy**. Vercel runs `npm run build` automatically.

---

## Local development

```bash
# Install Python dependencies
make install

# Install frontend dependencies
make install-web

# Terminal 1 — backend (hot-reload)
make run-backend

# Terminal 2 — frontend
make run-frontend
```

The frontend at `http://localhost:3000` hits the backend at `http://localhost:8000` by default
(hardcoded fallback in `apps/web/lib/api.ts`).

To override locally, create `apps/web/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## CORS

The backend reads `ALLOWED_ORIGINS` at startup:

```python
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
```

To allow both your Vercel domain and localhost:

```
ALLOWED_ORIGINS=https://deep-hedging-lab.vercel.app,http://localhost:3000
```

---

## Health check

`GET /` returns `{"status": "ok", ...}` — used by Railway's liveness probe.

---

## Notes

- The backend is stateless within a single process: run state is stored in an in-memory
  dict (`RUNS`). Restarting the process loses pending/completed run results. This is
  acceptable for a demo deployment; for persistence, replace with a database or Redis.
- The backend runs synchronous experiment jobs in a background thread. Railway's free tier
  (512 MB RAM, shared CPU) may be tight for large experiments; the presets are sized to
  complete within 60–90 seconds.
