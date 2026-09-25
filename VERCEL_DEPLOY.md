# Public jury dashboard on Vercel

This deployment is a **read-only aggregate evidence snapshot** of the frozen
V3-A evaluation. It is a live, shareable web page, not a live client-inference
service. The official prediction runner and local FastAPI remain unchanged.
No client transactions, TEST labels, submission rows, or model files are shipped.

## Refresh the measured snapshot

With the project Python environment and the recorded V3 artifacts available:

```powershell
& 'C:\Users\jagui\miniforge3\envs\tx-forecasting\python.exe' scripts/export_public_dashboard.py --artifact-root 'C:\Users\jagui\Transaction-Activity-Forecasting'
& 'C:\Users\jagui\miniforge3\envs\tx-forecasting\python.exe' scripts/export_public_dashboard.py --artifact-root 'C:\Users\jagui\Transaction-Activity-Forecasting' --check
```

The exporter uses `outputs/metrics/ubs_v3/valid_results.json` and
`importance_A.csv`, validates the selected model and 1,000-client VALID
cohort, and writes only allowlisted aggregate fields to
`frontend/publicSnapshot.js`. The source report SHA-256 is shown in the UI.
Commit the refreshed snapshot only after reviewing its diff.

## Deploy

1. Import `santidelatorre/Transaction-Activity-Forecasting` into Vercel.
2. Set **Root Directory** to `frontend`, **Framework Preset** to Vite,
   and **Production Branch** to `feature/jaime-unified-v2` until the
   team's PR is merged. Do not deploy an old `main` dashboard.
3. Keep the build command `npm run build` and output directory `dist`.
   No secrets or local artifact-root environment variable are needed.
4. Deploy, then open `/`, `/api/v1/health`, `/api/v1/dashboard`, and
   `/api/v1/experiments?limit=7&offset=0` on the Vercel domain.
5. Check that the page displays V3-A Macro-F1 and the 90-day horizon and that
   the browser console has no failing API requests. Only then share the URL.

The Vercel functions serve only aggregate dashboard, experiment and health
responses. They intentionally do **not** expose the local API's client,
overview, results or submission endpoints. Local development still uses the
FastAPI backend via the Vite proxy.

## Checks

```powershell
cd frontend
npm ci
npm run build
node --test tests/public-api.test.js
```

A Git-connected Vercel project redeploys when this branch changes. Refreshing
metrics requires a new verified export and commit; the page does not re-run
the predictor at request time.

For a disposable preview when the Windows CLI cannot run its internal build,
run `node build-prebuilt.mjs` from `frontend` after `npm run build`, followed by
`vercel deploy --temporary --prebuilt --yes`. The script packages the same
allowlisted snapshot and API handlers as Node.js functions. Anonymous preview
links expire after the interval printed by Vercel; they are **not** suitable
as final submission URLs until claimed by a Vercel account.
