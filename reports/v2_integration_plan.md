# V2 Integration Plan

## Baseline

Macro-F1 V1 = 0.2710243; accuracy = 0.2660; 1,000 official validation clients.
Exact stored score: 0.2710242658492452. V1 source reference: `0199a8b`.
The existing `configs/ubs_v1.toml`, runner, feature builder, model module and
submission must remain independently executable and unmodified. Fresh baseline
runs use separate output paths. Stored metrics are evidence to reproduce, not a
substitute for execution.

## Git discovery and report completeness

Started on `feature/v2-integration`, tracking its remote, clean at `0199a8b`.
Fetched all remotes with pruning. Main and integration initially match.
Seven V2 workstreams exist. Santiago's actual remote is
`origin/featrue/santiago-v2-erroranalysis`, not the name in the task. No branch
will be renamed. Esteban's report is outside `reports/handoff/` and has no
separate summary. All seven available reports and six summaries were read
before source changes; the missing Esteban summary is explicitly recorded.

Other discovered remotes: `origin/main`, `origin/HEAD`,
`origin/feature/v2-integration`, and the older personal branches
`origin/dev/Christian`, `origin/dev/Esteban`, `origin/dev/carles`,
`origin/dev/jaime`, `origin/dev/javi`, `origin/dev/laura`, `origin/dev/santiago`.

## Team contributions

| team_member | branch | area | best_result | delta_vs_v1 | main_contribution | recommended_commits | risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Esteban | origin/esteban-v2-models | Models | Reported official valid 0.3505, accuracy 0.340 | About +0.0795, unverified | CatBoost 300/depth4/lr0.05 balanced plus temperature 2, none bias -1.5 | 358d604; report be26308 | About 42 variants and calibration selected on valid; training self-label encoding |
| Santiago | origin/featrue/santiago-v2-erroranalysis | Error analysis | Reproduced V1 0.2710243 | 0; analysis only | none failures, candidate lifecycle, MCC and OOF recommendations | e7b42c0, analysis only | Post-hoc validation diagnostics are not causal evidence |
| Christian | origin/features/christian-v2-features | Features | No experiments or code | N/A | Concentration, activity and within-currency hypotheses | b18480c, documentation only | No measured improvement; overlaps text and temporal |
| Carles (Ginestar) | origin/features/ginestar-v2-temporal | Temporal / recurrence | Combined internal fold mean 0.469750 vs internal V1 0.461885 | +0.007866 INTERNAL ONLY | Stream summaries, support, temporal score blocks and isolated folds | Review 6ba1311; analysis 1ceca7c | Wins only 3/5 folds; official diagnostic 0.153978 uses different calibration |
| Jaime | origin/features/jaime-v2-experiment-tracking | Experiment tracking | 40/40 concurrent thread writes | N/A | SQLite logger, metadata, class metrics and decisions | fe85154 then cd19c44 | Not wired into runner; no data hashes or process stress test |
| Javier | origin/features/javier-v2-text | Text / merchant proxy | 0.243527, accuracy 0.300 | -0.027497 | Text and leave-one-client-out merchant utilities, fixed blends | f598fc3 then 7ef31a6 | All predictors worse than V1; description distribution shift |
| Laura | origin/features/laura-v2-integracion | Integration / validation | 19 gate tests, no model score | N/A | Provenance, tests, lint, artifact and submission checks | 614f769 | Gate is V1-specific; WARN is not approval; no live evaluation |

### Reports inspected

- Esteban: `Reports/esteban_mejor_modelo_0.35.md`; no separate summary.
- Santiago: `reports/handoff/santiago_error_analysis.md` and `_summary.md`.
- Christian: `reports/handoff/christian_feature_engineering.md` and `_summary.md`.
- Carles: `reports/handoff/carles_temporal_recurrence.md` and `_summary.md`.
- Jaime: `reports/handoff/jaime_experiment_tracking.md` and `_summary.md`.
- Javier: `reports/handoff/javier_text_merchants.md` and `_summary.md`.
- Laura: `reports/handoff/laura_integration_validation.md` and `_summary.md`.

All named commits exist in the corresponding remote history. Branch logs and
three-dot diffs against origin/main were inspected. Esteban also contains
`Reports/santiago_error_analysis*.md` and a different analysis script: these
are duplicates and will not replace Santiago's canonical analysis.

## Agreements between reports

- Preserve V1, the provided client split, pre-2026-01-01 history and the official
  fixed-eight-label scorer; rank primarily by Macro-F1.
- none is the main V1 weakness (F1 0.094183; 17/293 detected). Improving its
  accuracy alone can harm the seven families.
- V1 computes 218 numeric features but the selected heuristic uses only 32.
- Global text is weaker than recurrence; text on active streams is a future
  hypothesis, not implemented evidence of an improvement.
- Train-supervised description lift applied to its own training rows is unsafe
  for ML training; held-out validation mapping must be fitted on train only.
- Reusing validation for selection makes the published winner optimistic;
  internal client folds and frozen recipes are useful additional checks.

## Contradictions

- Esteban says negative none bias emits more none. His code adds the negative
  value to the none logit, which suppresses it. Audit the claim against code.
- Carles' 0.47 is internal training OOF, whereas Esteban/Javier use official
  validation. These numbers cannot be ranked as competing official scores.
- Christian, Jaime and Laura could not reproduce V1 in their environments;
  Santiago, Carles and Javier did. This is an environment difference.
- Some handoffs say no commit exists; remote logs now identify their later
  documentation commits. Remote history is authoritative for commit identity.
- The successful calibration recipe may conceal in-sample target-encoding
  mismatch. A raw model/control and a safe training-feature variant are needed.

## Dependencies

1. Restore compatible Python and reproduce V1 before predictive changes.
2. Tracking and review checks precede candidate runs; they are neutral tooling.
3. Temporal reweighting requires the V1 builder's train-only description mapping.
4. Text blend reproduction requires the saved V1 predictions and matching IDs.
5. Model evaluation follows controlled feature evaluation. CatBoost already
   exists in V1; only the selected recipe/calibration needs adaptation.
6. Any final train+valid refit occurs after validation selection; its predictions
   must never be scored as held-out validation predictions.

## Potential duplicates

- Christian's description concentration overlaps Javier's merchant statistics.
- Christian's recent activity overlaps Carles' temporal summaries.
- Esteban's copied error-analysis script conflicts with Santiago's longer one.
- Jaime's logger and Laura's provenance overlap partly but serve different jobs.
- Avoid importing broad model grids, reports/figures, and historical proxy
  evaluation when only a small predictive component is needed.

## Leakage risks

- Reject future timestamps, overlapping client partitions, IDs as predictors,
  validation/test-fitted vocabularies, and label-derived historical targets.
- Keep V1 unchanged for comparison. For a new fitted ML predictor, use
  client-excluded/cross-fitted supervised features or omit target-derived columns.
- Esteban's reported calibration was selected on official valid, as was V1.
  Reproduce the frozen reported recipe as exploratory evidence, label the
  selection risk, and audit before promotion. No new blind grid on valid.
- Carles' mappings inferred from full train are not transaction ground truth.
- Pooled nominal currency amounts are inherited V1 behavior; do not expand them.
- Record data/code hashes and package versions; final predictions must repeat.

## Integration candidates

| classification | contribution | decision criterion |
| --- | --- | --- |
| MUST TEST | V1 reproduction, shared scoring and submission checks | Match stored V1 and original submission hash |
| MUST TEST | Jaime tracking | Tests pass; predictions unchanged; log all official candidate runs |
| MUST TEST | Carles intervals / periodicity / activity / horizon separately with fixed V1 bias | Same official split and scorer; no accumulated regressions |
| MUST TEST | Esteban frozen CatBoost raw then reported calibration | Same inputs; verify improvement; audit supervised training features |
| MUST TEST | Safe supervised-feature training or removal control for any accepted ML recipe | No own-label contribution; measure effect separately |
| SHOULD TEST | Laura checks, adapted only where V2 runner differs | Tests and clear provenance; no fabricated gate PASS |
| SHOULD TEST | Javier frozen strongest blend | Verify negative result before rejecting production activation |
| SHOULD TEST | Best measured temporal block plus best safe model | Only after individual measurements and with explicit hypothesis |
| OPTIONAL | Christian concentration / active-day hypotheses | No implementation available; only if evidence warrants new code |
| OPTIONAL | Santiago diagnostic script and all plots | Use findings; no need to import 1,792-line analysis into prediction path |
| REJECT | Full branch merges, broad model grid, copied analysis scripts | Unnecessary scope / duplicate code |
| REJECT | Annual periodicity as established gain, historical proxy scores as official F1 | Unsupported or incomparable evidence |
| REJECT | Any model using future/test information or own labels as training features | Leakage blocker |

## Proposed integration order

1. Freeze source and hashes of existing V1 artifacts; reproduce into isolated
   ignored outputs; run baseline tests. Repair local runtime without changing V1.
2. Integrate reviewed tracking and quality evidence utilities separately. Commit
   each contribution and rerun tests plus official baseline prediction scoring.
3. Integrate a switchable temporal module and runner. Measure intervals,
   periodicity, activity, horizon independently with frozen V1 calibration;
   reject regressions immediately, preserving every result row. Test a combined
   block only if individual evidence gives a reason.
4. Reproduce Javier's strongest fixed blend in isolation; retain experimental
   utility only if needed, never activate a negative predictor by default.
5. Evaluate Esteban's frozen CatBoost recipe on the established feature table,
   raw then calibrated, with explicit own-label audit. Adapt only useful code.
6. Establish a safe fitted-model feature path and measure its effect. Evaluate
   only one justified temporal/model interaction, if there is a winning block.
7. Freeze the best demonstrated recipe; repeat validation and final test refit,
   validate submission schema/IDs/order/classes and deterministic hashes.
8. Run all tests, lint, format, pre-commit and import checks; write full results,
   final report and short summary; commit on integration branch only.

Small official gains remain exploratory on the reused 1,000-client validation
set. If no safe and reproducible candidate beats 0.2710243, V1 remains the
validated recommendation and the final report must say so.
