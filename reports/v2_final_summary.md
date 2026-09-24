# V2 Final Summary

V1 Macro-F1: **0.2710243**

V2 Macro-F1: **0.3915495**

Improvement: **+0.1205252** (44.5% relative)

Accuracy: **0.2660 → 0.4240**; same 1,000 validation clients.

WHAT WE KEPT:

- Esteban's small CatBoost, trained on 146 history features without target encodings.
- Carles' periodicity heuristic in one fixed 75% CatBoost / 25% heuristic blend.
- Santiago's error diagnosis; Christian/Laura's encoding warning; Jaime's tracking and Laura's checks.

WHAT WE REJECTED:

- Temporal intervals/activity/horizon and Javier's merchant blend: lower Macro-F1.
- Original 218-feature ML recipe: own-label training influence, despite reproducing 0.3504931.
- Calibration on the safe model: falls from 0.3839505 to 0.3186625.

BIGGEST V2 IMPROVEMENTS:

1. Remove 72 supervised family columns from ML training.
2. Use the frozen small CatBoost; none F1 rises from 0.0942 to 0.5194.
3. Periodicity blend recovers streaming to 0.2500 and partly recovers music.

MAIN REMAINING PROBLEM:
Music F1 is 0.1699, below V1's 0.2390. Official validation has been reused for
selection; results are not an independent generalization estimate. The small
blend improvement is less certain than the large total improvement over V1.

VALIDATION:
79 tests; valid 1,000-row submission; two fresh fits produce identical prediction
and submission hashes. V1 source and original submission remain unchanged.
Seven reports read; six separate summaries exist (Esteban has no separate one).

READY FOR SUBMISSION: **YES**, after human review.

File: `outputs/predictions/submission_v2.csv`.
Run: `python scripts/run_ubs_v2.py` (local recovered runtime: `.venv/runtime_v2/python.exe`).
Full evidence: `reports/v2_final_report.md`.
Branch: `feature/v2-integration`. No PR or merge to main performed.
