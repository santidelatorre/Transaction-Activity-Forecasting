# V2 Final Report

## V1 baseline

Official V1 Macro-F1 = **0.2710243** (exact 0.2710242658492452), accuracy
0.2660, 1,000 validation clients. The original full V1 runner was reproduced
at the same score and returned an identical submission byte for byte.
Source reference `0199a8b`; original V1 runner, config, builder and models remain
unchanged. Fresh baseline outputs are in `outputs/metrics/v2_integration/baseline`.
The original submission SHA-256 remains
`b67d364e207d297b1a4d5898f0c37b6f62289cf56808882f5bdcfeb6c5795b0f`.

## Team reports inspected

Seven workstreams and seven complete available reports were read before code
integration. Six separate summaries exist; Esteban has no separate summary.

| member | actual remote branch | report and summary |
| --- | --- | --- |
| Esteban | origin/esteban-v2-models | Reports/esteban_mejor_modelo_0.35.md; no separate summary |
| Santiago | origin/featrue/santiago-v2-erroranalysis | reports/handoff/santiago_error_analysis.md; santiago_error_analysis_summary.md |
| Christian | origin/features/christian-v2-features | reports/handoff/christian_feature_engineering.md; christian_feature_engineering_summary.md |
| Carles | origin/features/ginestar-v2-temporal | reports/handoff/carles_temporal_recurrence.md; carles_temporal_recurrence_summary.md |
| Jaime | origin/features/jaime-v2-experiment-tracking | reports/handoff/jaime_experiment_tracking.md; jaime_experiment_tracking_summary.md |
| Javier | origin/features/javier-v2-text | reports/handoff/javier_text_merchants.md; javier_text_merchants_summary.md |
| Laura | origin/features/laura-v2-integracion | reports/handoff/laura_integration_validation.md; laura_integration_validation_summary.md |

Santiago's actual name differs from the request. No branch was renamed or
rewritten. All seven logs, three-dot diffs, source files and recommended commits
were inspected. Esteban's duplicate Santiago analysis was not imported.
See `v2_integration_plan.md` for hypotheses, disagreements, dependencies,
recommendations, risks and integration priorities.

## Candidates evaluated

| candidate | Macro-F1 | accuracy | decision |
| --- | --- | --- | --- |
| baseline | 0.271024266 | 0.2660 | BASELINE |
| tracking | 0.271024266 | 0.2660 | KEEP_TOOLING |
| quality_gate | 0.271024266 | 0.2660 | KEEP_TOOLING |
| temporal_intervals | 0.267817007 | 0.2620 | REJECT |
| temporal_periodicity | 0.272636283 | 0.2650 | KEEP_PROVISIONAL |
| temporal_activity | 0.264064548 | 0.2610 | REJECT |
| temporal_horizon | 0.261599760 | 0.2580 | REJECT |
| merchant_blend | 0.243527493 | 0.3000 | REJECT |
| catboost_v1_raw | 0.192653263 | 0.2920 | REJECT_SELF_LABEL |
| catboost_v1_calibrated | 0.350493131 | 0.3400 | REJECT_SELF_LABEL |
| catboost_history_raw | 0.383950488 | 0.4240 | KEEP_PROVISIONAL |
| catboost_history_calibrated | 0.318662517 | 0.3250 | REJECT |
| catboost_history_temporal_blend | 0.391549456 | 0.4240 | KEEP_PROVISIONAL |
| final_v2 | 0.391549456 | 0.4240 | KEEP_FINAL |
| quality_compatibility | 0.391549456 | 0.4240 | KEEP_TOOLING |

Full progression, including deltas against the previous stable/control state,
is in `v2_integration_results.md`. Each experiment uses the same official
2,000/1,000 client split and fixed-eight-label evaluator. Temporal blocks are
tested separately with V1 bias -1 and temperature 1; rejected blocks are never
stacked. Their opt-in reproduction code is retained as experimental tooling.
No whole feature branch was merged.

## Candidates accepted

- Esteban's fixed CatBoost recipe, adapted to **146 history features without
  target-derived family columns**. Standalone Macro-F1 0.383950488.
- Carles' periodicity reweighting as the heuristic component of one fixed
  75/25 blend. Combined official Macro-F1 0.391549456; contribution beyond raw
  CatBoost is 0.007598968 on this validation set.
- Santiago's diagnosis and Christian/Laura's training-encoding warning inform
  the removal of 72 supervised columns from ML. This removal is integration
  work, not a feature implementation claimed on Christian's branch.
- Jaime's SQLite logger and Laura's tests/provenance/submission review tools.
  These are neutral infrastructure, not predictive gains.

## Candidates rejected

- Temporal intervals, activity and horizon: all lower official Macro-F1 with
  fixed V1 calibration. They remain disabled in the final predictor.
- Javier's strongest fixed merchant blend: 0.243527493, reproducing its negative
  result despite higher accuracy. Text utilities remain research-only.
- Esteban's original raw and calibrated 218-feature recipes: training rows
  contain their own-label-derived family evidence. The reported calibrated
  0.350493131 was reproduced, then rejected for this issue.
- Applying Esteban's frozen calibration to safe history CatBoost: 0.318662517,
  below its uncalibrated 0.383950488. Final V2 does not use this calibration.
- Broad model grids, combined temporal weights, annual-period claims and copied
  analysis scripts: no sufficient comparable evidence or unnecessary scope.
- Christian's proposed concentration/amount/activity features: deferred, not
  empirically disproven. The branch has no implementation or measured ablation.

## Why rejected approaches failed

The temporal blocks mostly reduce useful family evidence while leaving other
heuristic terms unchanged. This is an observed score regression; it does not
prove a general causal explanation for the dataset. Merchant features raise
none detection and accuracy but damage the eight-class objective. The original
ML training path allows supervised description lift to see each training label;
its distributions differ from held-out inference. Removing that route produces
a much stronger raw CatBoost result. A negative none logit bias suppresses none,
contrary to the wording in Esteban's report, and harms the safe model.

## Final V2 architecture

`scripts/run_ubs_v2.py` is independent of the unchanged V1 runner.
`ubs.v2.IntegratedV2Model` fits a label-free history builder and small CatBoost.
A separate train-fitted description map supplies the heuristic only at
prediction time for excluded clients. `predict_components` refuses fit-client
overlap. The final probabilities are 0.75 history CatBoost + 0.25 periodicity
heuristic, in the official label order. No validation fitting or parameter
search occurs in this runner. After held-out scoring it refits the same recipe
on 3,000 train+valid clients for the 1,000 test clients.

## Features

146 numeric history features reuse V1 volume, amounts, currency-separated
moments, MCC/type/currency/direction counts and shares, activity windows,
calendar and unsupervised recurrence aggregates. No raw client ID is a feature.
All 72 `family_*` columns are absent from the CatBoost design matrix. Category
vocabulary and imputation are fitted on training clients only.

The separate heuristic uses 32 existing family inputs, with only the seven
positive-family recurrence scores reweighted by Carles' periodicity factor.
Unknown/unsupported periodicity is explicit; timestamps are strictly before
cutoff, and repeated timestamps are deduplicated for intervals. The mapping is
learned only from fitted clients and is never used to build ML training rows.

## Model

CatBoost multiclass: 300 iterations, depth 4, learning rate 0.05, balanced class
weights, seed 42, CPU and four threads. The heuristic retains V1 bias -1 and
temperature 1. Blend weight 0.25 was specified before that single interaction
was scored; no blend grid or class threshold search was performed.

## Validation

Official split, cutoff 2026-01-01, target `target_next_recurring_merchant`, eight
labels, Macro-F1 primary. Data loader verifies client separation, labels and
timestamps. Final held-out results come from the pre-refit model, never from
the final train+valid model. Every recorded candidate prediction artifact was
checked against its saved SHA-256.

Additional fixed-recipe audit: five stratified client folds inside train,
seed 42; all feature/mapping/model state rebuilt per fold. V1 control bias is
fixed -1, so these are **internal comparisons, not the official V1 protocol**.

| fold | V1 fixed bias | history CatBoost | blend | blend - history |
| --- | --- | --- | --- | --- |
| 1 | 0.320713 | 0.387150 | 0.386089 | -0.001061 |
| 2 | 0.328372 | 0.397995 | 0.414131 | +0.016135 |
| 3 | 0.383563 | 0.419547 | 0.423242 | +0.003696 |
| 4 | 0.372925 | 0.415319 | 0.426194 | +0.010875 |
| 5 | 0.348621 | 0.372650 | 0.387149 | +0.014499 |

Pooled internal OOF Macro-F1: V1 fixed = 0.351330,
history = 0.399767, blend = 0.408884.

Paired stratified bootstrap, 2,000 replicates, seed 42, official validation:
95% interval of blend minus V1 Macro-F1 = [+0.086887, +0.155749].
Blend minus raw history interval = [-0.006087, +0.022187].
Intervals are conditional on this reused split, not a correction for model
selection. The small incremental blend advantage must not be described as a
proven generalization gain if its interval crosses zero.

## V1 vs V2

| metric | V1 | V2 | delta |
| --- | ---: | ---: | ---: |
| Macro-F1 | 0.271024266 | 0.391549456 | +0.120525190 |
| Accuracy | 0.266000 | 0.424000 | +0.158000 |
| Validation clients | 1000 | 1000 | 0 |
| Submission clients | 1000 | 1000 | 0 |

## F1 per class

| class | V1 F1 | V2 F1 | delta |
| --- | --- | --- | --- |
| cloud | 0.288401 | 0.450000 | +0.161599 |
| gym | 0.339921 | 0.487273 | +0.147352 |
| insurance | 0.323810 | 0.429150 | +0.105340 |
| mobile | 0.364217 | 0.452675 | +0.088458 |
| music | 0.238994 | 0.169935 | -0.069059 |
| software | 0.271357 | 0.373984 | +0.102627 |
| streaming | 0.247312 | 0.250000 | +0.002688 |
| none | 0.094183 | 0.519380 | +0.425197 |

### V1 confusion matrix

| actual / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 46 | 5 | 5 | 15 | 4 | 3 | 7 | 4 |
| gym | 20 | 43 | 5 | 20 | 5 | 12 | 5 | 11 |
| insurance | 16 | 11 | 34 | 14 | 2 | 8 | 8 | 6 |
| mobile | 12 | 4 | 11 | 57 | 1 | 4 | 5 | 10 |
| music | 13 | 12 | 9 | 21 | 19 | 5 | 6 | 8 |
| software | 19 | 13 | 8 | 13 | 8 | 27 | 9 | 7 |
| streaming | 15 | 13 | 6 | 26 | 3 | 6 | 23 | 5 |
| none | 89 | 31 | 33 | 43 | 24 | 30 | 26 | 17 |

### V2 confusion matrix

| actual / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

## Remaining weaknesses

Music remains the weakest class, with F1 0.169935 versus V1 0.238994; this
regression is disclosed and accepted for the +0.120525 global Macro-F1 gain.
Streaming recovers to 0.250000 but remains weak. Accuracy is still 0.4240.
The synthetic description/MCC relationship and train-to-valid distribution
shift limit extrapolation. V1 still pools some nominal currencies; these
inherited history features were not expanded. All team work reused official
validation, so no untouched test generalization estimate is available.

## Leakage audit

- PASS: no post-cutoff observations; loader checks disjoint train/valid/test IDs.
- PASS: history builder accepts no labels; tests show equivalence under label
  changes and no family columns in ML training input.
- PASS: supervised heuristic mapping is fitted only on training clients and
  prediction rejects fitted-client overlap; no in-sample self-label inference.
- PASS: no IDs, hidden test labels, validation-fitted vocabulary/scaling or
  fabricated historical family labels enter the final predictor.
- PASS: final refit predictions are separated from validation metrics.
- LIMITATION: V1 bias, reported CatBoost recipe and candidate choice have prior
  official-validation selection history. This is selection optimism, not an
  assertion that validation is independent. No blanket proof of all leakage
  or production generalization is claimed.

## Tests

79 tests pass, including the original 37, logger, quality gate, temporal
cutoff/interval/client-fold checks, text utilities and V2 label isolation.
Ruff check, Ruff format and pre-commit across all files pass. The quality gate
passes all 79 tests and lint/format checks with no FAIL entries. Its overall
status is WARN because its default mode does not run predictive evaluation,
has no reviewed baseline gate report, and reserves manual review/peak-memory
checks. V2 metrics and submission evidence were verified separately by the
official CLI and the two fresh runs; no blanket gate PASS is claimed.
Gate evidence: `outputs/metrics/quality_gate/20260924T150029697840Z/report.json`.
The logger test was corrected to support temporary directories inside Git,
and two gate type checks were adapted to the pinned Ruff hook. These fixes
do not change any predictor; final saved predictions were rescored afterwards.
The local Python 3.12.6 runtime was recovered under ignored
`.venv/runtime_v2/`, reusing installed dependencies. Core versions:
{"catboost": "1.2.10", "numpy": "2.3.5", "pandas": "2.3.3", "scikit-learn": "1.9.1"}.

## Submission validation

Final file: `outputs/predictions/submission_v2.csv` (ignored; not uploaded).
1,000 rows, 1,000 unique IDs, exact sample ID set and order, exactly the two
official columns, allowed and non-null classes. Shared UBS and official
validators pass. Two independent fresh fits generate identical validation
predictions and identical submission bytes.

- Validation SHA-256: `57ca783d554b9aea6602f1046582dc18c93ddfe5d418ba50173ab1e36cb9313a`.
- Submission SHA-256: `da1314cfdea23ffea7756ff82db4e7d4efe77c6fcaf33cb0b66a689795f59adc`.
- Run source commit: `fe6fdf7bd01e79c9a53057541ee131854e6c54bb`.
- First-run elapsed seconds: 143.5; repeat: 127.9.

Input fingerprints (no client-level data is versioned):

| input | SHA-256 |
| --- | --- |
| train_transactions.jsonl | ba0902b33b1181921ee5b1f8f445693201c0968ca52d70a92e12cbb114e7d70a |
| train_labels.csv | cab0ec46064348f970ffafdd4f42a12d41a8db660e9db835e7235851b556ff9f |
| valid_transactions.jsonl | 2c30941d49ca20f7b082526407f24529c5a704205610373eff952f4f444a3622 |
| valid_labels.csv | 979bd9b69253084fc9e2b1ef473c07a628a8b18091169051ba304cfed0fb5642 |
| test_transactions.jsonl | cc6083d6e0d4f4787f0766d6eb22fd5d267194c9a71a04da81e60dc23f5463e4 |
| sample_submission.csv | 53a24cd23d680c2573f5b048dd69d1747f20adc45e308593d914ee13e8d00ed2 |

## Commits incorporated

| source_branch | source_commit | integration_commit | description |
| --- | --- | --- | --- |
| origin/features/jaime-v2-experiment-tracking | fe85154 | b5f368b | Logger foundation, clean cherry-pick |
| origin/features/jaime-v2-experiment-tracking | cd19c44 | d3e34c1 | Outcomes and concurrency tests, clean cherry-pick |
| origin/features/laura-v2-integracion | 614f769 | 0974f23 | Quality gate and tests, clean cherry-pick |
| origin/features/ginestar-v2-temporal | 6ba1311 | 68a2553 | Selected temporal modules/tests; horizon and optional combined guard fixes |
| origin/features/javier-v2-text | f598fc3, 7ef31a6 | 95166df | Selected utilities/tests and adapted fixed-blend reproduction; research only |
| origin/esteban-v2-models | 358d604 | a570a23 | Fixed recipe and minimal calibration adaptation; new label-free projection |
| origin/esteban-v2-models + temporal | 358d604, 6ba1311 | 6c49ebb | Model decisions and one fixed interaction |
| integration | — | fe6fdf7 | Frozen independent V2 runner, audit and split guards |
| Jaime + Laura integration | cd19c44, 614f769 | 4db8b96 | Temporary-directory test and pinned Ruff compatibility; predictor unchanged |
| reports from all seven | listed above | 6eafc9a, ca8dfcb | Initial plan and measured temporal decisions |

Santiago and Christian's reports informed design but no predictive code was
cherry-picked from their analysis/documentation-only branches. The documentation
commit containing this report is discoverable with `git log -- reports/v2_final_report.md`.

## Reproduction

From the repository root with a working supported Python environment:

```powershell
python scripts/run_ubs_baseline.py --config configs/ubs_v1.toml
python scripts/run_ubs_v2.py
python scripts/audit_ubs_v2.py
python -m pytest
```

On this machine use `.venv/runtime_v2/python.exe` instead of `python`, because
the old `.venv/Scripts/python.exe` points to a removed interpreter. V1 artifacts
are preserved during this integration by using the isolated baseline config in
`outputs/metrics/v2_integration/baseline.toml`. The audit expects the local
candidate ledger/predictions; individual trials can be reproduced with
`scripts/evaluate_v2_candidate.py --candidate NAME` in the order recorded.

## Final recommendation

Use the frozen V2 candidate for human review and submission: it beats the
official V1 baseline on the same split, passes tests and data contracts, has
no identified final-pipeline leakage violation, and reproduces identical
submission bytes. Preserve V1 as fallback. Review the music regression and
validation-selection limitation. Branch: `feature/v2-integration` only.
No PR, merge to main, branch rename, or teammate-history rewrite was performed.
