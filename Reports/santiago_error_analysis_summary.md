# Error Analysis — Meeting Summary

## Baseline
Macro-F1: **0.2710243** (verified)
Accuracy: **0.2660**
Model: `recurrence_heuristic` · 1000 valid clients

## 3 weakest classes

1. `none` — F1 0.094 (recall 0.058)
2. `music` — F1 0.239
3. `streaming` — F1 0.247

## 3 biggest confusions

1. `none → cloud` — 89 errors (32% of none mistakes)
2. `none → mobile` — 43 errors
3. `streaming → mobile` — 26 errors (also `music → mobile` 21)

## 5 key findings

1. V1 is a family hunter: only 68 `none` predictions vs 293 true.
2. none→cloud is driven by `family_cloud_recurrence_score` (argmax cloud on 86.5% of those errors).
3. Wrong `none` clients often look *more* regular/stable than correct `none`.
4. Winner does **not** use text TF-IDF; digital classes collide (music/streaming↔mobile).
5. Global volume gaps are weak; class-conditional gaps are actionable.

## Who should investigate what

| team | task | priority |
|---|---|---|
| FEATURE ENGINEERING | none-exclusive family scores | CRITICAL |
| TEMPORAL / RECURRENCE | relative recency + interrupted streams (≥3 support) | CRITICAL |
| TEXT / MERCHANTS | mobile/music/streaming aliases | CRITICAL |
| MODELS / TUNING | none calibration after feature fixes | HIGH |
| EXPERIMENT TRACKING | log per-class F1 + top confusions always | HIGH |
| INTEGRATION / VALIDATION | freeze protocol; watch valid-tuning optimism | MEDIUM |

## Top 3 recommended experiments

1. **EA-001** — exclusive / none-penalized family scores
2. **EA-002** — relative recency + interruption flags
3. **EA-003** — alias map for digital subscriptions

## Main risk

Treating “has recurring payments” as proof of a target family (hurts `none`).

## Recommendation for V2

Fix **none exclusivity + temporal interruption + digital text aliases** before more model grids. Re-check Macro-F1 and none recall after each isolated change.
