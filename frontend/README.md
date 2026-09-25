# Recurring Insights dashboard

The React dashboard is a read-only jury view of the frozen V3-A predictor and
the recorded V1–V4 evaluation history. It does not fit a model, alter a
prediction, read TEST labels, or treat model scores as probabilities.
The bundled UBS mark identifies the challenge sponsor; this remains an
independent prototype, not an official UBS product. SVG source:
<https://en.wikipedia.org/wiki/File:UBS_Logo.svg> (UBS trademark).

From the repository root, in PowerShell with the project Python environment:

```powershell
conda activate tx-forecasting
# Only needed when the ignored V3 evaluation files live in another worktree:
$env:RECURRING_ARTIFACT_ROOT='C:\Users\jagui\Transaction-Activity-Forecasting'
python -m uvicorn transaction_forecasting.api.main:app --app-dir src --host 127.0.0.1 --port 8001
```

In a second terminal:

```powershell
cd frontend
npm ci
$env:RECURRING_API_TARGET='http://127.0.0.1:8001'
npm run dev -- --port 5173
```

## Vercel

Import this repository on Vercel and leave the Root Directory as the repository
root. `vercel.json` installs and builds `frontend/`. Do not point Vercel at
`frontend/` alone: `/api/v1/*` is a small Node API in the repo root that serves
the committed dashboard snapshot (`api/data/dashboard.json`). No Python runtime
and no gitignored datasets are required. The free plan is enough.

Refresh the snapshot after a real V3 artifact appears locally:

```powershell
$env:PYTHONPATH='src'
python -c "import json; from pathlib import Path; from transaction_forecasting.api.dashboard import snapshot; Path('api/data/dashboard.json').write_text(json.dumps(snapshot(Path('.').resolve()), indent=2)+'\n', encoding='utf-8')"
```

Open <http://127.0.0.1:5173/>. Healthcheck:
`http://127.0.0.1:8001/api/v1/health`. For a static build, run
`npm run build`; FastAPI serves `frontend/dist` after restart.

## Evidence contract

- `GET /api/v1/dashboard` reads
  `outputs/metrics/ubs_v3/valid_results.json` and
  `importance_A.csv` from `RECURRING_ARTIFACT_ROOT`.
- `reports/dashboard_evidence.json` is a small, source-traced extract of the
  V1/V2 final report and the V4 synthesis decision. Its V4 source commit and
  Git blob are recorded in the file. It is data for the dashboard, not a new
  evaluation. V3 variant scores come from the live evaluation artifact.
- The version line joins official VALID points only when the V3 report's V2
  control, cohort size, and input fingerprints match the V2 report. V4 is a
  decision marker with **no new score**.
  Source-report order is not experiment chronology.
- V4 clean TRAIN OOF and reused VALID scores have separate dot-chart columns.
  Author-reported scores use open markers; independently reproduced scores
  use filled markers. The simulated Ginestar stress score is outside the
  clean-model chart. Missing VALID evaluations remain missing.
- `GET /api/v1/experiments` provides the paginated V3 report arms followed
  by any existing SQLite experiment rows. It never creates a database.
- The selected V3-A uses 75% CatBoost and 25% periodicity-heuristic **weights**.
  The one V4 equal-average experiment uses fixed 50/50 weights; it was not
  evaluated on VALID. Weights are not component scores.
- VALID was reused in model research. There is no TEST performance claim.

Checks, with both servers running:

```powershell
python -m pytest -q tests/test_dashboard.py
cd frontend
npm run build
npm run test:ui
```

Browser tests verify API-to-screen scores, missing and incomparable results,
pagination, responsive overflow, and retry behavior. They save ignored
desktop/mobile screenshots under `frontend/test-results/`.
