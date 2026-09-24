# V2 Submission Report

Generated on 2026-09-24 from branch `feature/v2-integration` at source commit
`db1e2dce41e06cf744f81f46827bb2fce9f870d3` plus the submission-readiness
working-tree changes described by this report.

## Model

- Selected model: fixed 75% history CatBoost / 25% periodicity-heuristic blend.
- CatBoost: multiclass, 300 iterations, depth 4, learning rate 0.05, balanced
  class weights, CPU, four threads.
- Features: 146 label-free client-history aggregates covering transaction volume,
  amounts, currencies, MCC/type/direction counts and shares, recency/activity,
  calendar patterns, gaps and unsupervised recurrence/periodicity statistics.
- The supervised merchant-description mapping is isolated to the held-out
  heuristic path. CatBoost receives no target-derived `family_*` columns.
- Seed: 42.
- Model selection: TRAIN fit and official VALID macro-F1. After freezing the
  recipe, the final model was refit on TRAIN+VALID (3,000 clients) before TEST
  inference. TEST was not used for model selection.
- Optional unlabeled pretraining data was not used because the frozen V2 recipe
  is not designed to consume it.

## Validation

Official validation split: 1,000 clients. Primary metric is fixed-eight-class
macro-F1.

- Macro-F1: **0.391549456**
- Accuracy: **0.424000**

| class | support | predicted | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cloud | 89 | 71 | 0.507042 | 0.404494 | 0.450000 |
| gym | 121 | 154 | 0.435065 | 0.553719 | 0.487273 |
| insurance | 99 | 148 | 0.358108 | 0.535354 | 0.429150 |
| mobile | 104 | 139 | 0.395683 | 0.528846 | 0.452675 |
| music | 93 | 60 | 0.216667 | 0.139785 | 0.169935 |
| software | 104 | 142 | 0.323944 | 0.442308 | 0.373984 |
| streaming | 97 | 63 | 0.317460 | 0.206186 | 0.250000 |
| none | 293 | 223 | 0.600897 | 0.457338 | 0.519380 |

Validation label distribution: cloud 89, gym 121, insurance 99, mobile 104,
music 93, software 104, streaming 97, none 293.

Confusion matrix (rows are actual labels, columns are predicted labels; label
order: cloud, gym, insurance, mobile, music, software, streaming, none):

| actual | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

Machine-readable metrics and validation predictions are in
`outputs/metrics/ubs_v2_submission/` (ignored generated artifacts).

## Comparison

| metric | V1 | V2 | absolute difference |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.271024266 | 0.391549456 | +0.120525190 |
| Accuracy | 0.266000 | 0.424000 | +0.158000 |

V2 improves F1 for cloud (+0.161599), gym (+0.147352), insurance
(+0.105340), mobile (+0.088458), software (+0.102627), streaming (+0.002688)
and none (+0.425197). Music decreases by 0.069059. The recommendation is to
submit V2 because it has the higher official-validation macro-F1 by 0.120525.
V1 remains intact as a fallback; no `submission_recommended.csv` is needed
because `submission_v2.csv` is already the recommended artifact.

## Test submission

- Clients: 1,000.
- CSV: `outputs/submission_v2.csv`.
- SHA-256: `da1314cfdea23ffea7756ff82db4e7d4efe77c6fcaf33cb0b66a689795f59adc`.
- Prediction distribution: cloud 92, gym 164, insurance 164, mobile 172,
  music 56, software 175, streaming 45, none 132.

## Contract validation

- Correct columns and order: PASS.
- Correct row count (1,000): PASS.
- Exact sample ID set and exact sample order: PASS.
- Unique, non-null client IDs: PASS.
- Allowed labels only: PASS.
- Non-null and non-blank predictions: PASS.
- Submission validator: **PASS**.

The contract was checked against the UBS challenge specification and against
`data/raw/ubs_2026/sample_submission.csv`. The local data directory reflects
the repository's established ignored-data layout; its files match the official
package filenames.

## Reproduction

From the repository root with the project environment installed:

```powershell
python scripts/run_ubs_v2.py
python scripts/validate_submission.py
```

The first command performs TRAIN-only fitting, VALID evaluation/selection,
TRAIN+VALID final refit, TEST inference and writes `outputs/submission_v2.csv`.
The second command independently validates the upload artifact. On the current
machine, `.venv/Scripts/python.exe` is the verified Python 3.12 runtime.

## Manual submission checklist

- [ ] Team name preparado
- [ ] Repository link preparado: `https://github.com/santidelatorre/Transaction-Activity-Forecasting`
- [x] `outputs/submission_v2.csv` generado
- [x] Submission validator PASS
- [ ] Archivo revisado
- [ ] Subir CSV al formulario oficial antes del milestone correspondiente

The team name was not found in the repository and must be supplied manually.
