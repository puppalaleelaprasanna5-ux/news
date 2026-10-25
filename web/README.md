# NewsCred web application

This React 19 + TypeScript application is a Vite single-page app using
TanStack Router. The page has no server loaders, server functions, or other
SSR-only behavior, so production uses a static client build. FastAPI serves
that build from `web/dist` alongside `/health` and `/analyze`.

## Local development (Windows)

Install Node.js 20 or newer, then from the repository root:

```powershell
cd web
npm ci
npm run dev
```

In a second terminal, start the existing backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:5173>. Vite proxies `/health` and `/analyze` to port
8000; no frontend `.env` file is required for local development. The backend
allows CORS only from the local development origins, and production uses
same-origin requests.

## Frontend checks

From `web`:

```powershell
npm test
npx tsc --noEmit
npm run build
```

The response parser preserves classifier probabilities and their uncalibrated
status, per-claim evidence interpretations and limitations, source metadata,
service warnings, and the API's separate final/credibility assessments. It
does not turn similarity into support or contradiction, and identifies
fictional sample documents when the API marks them as synthetic.
