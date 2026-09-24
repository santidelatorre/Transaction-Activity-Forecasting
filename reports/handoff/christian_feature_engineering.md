# Team Handoff — Feature Engineering

> **Status: BLOCKED — no raw UBS data or working Python runtime is available in this workspace.** No new features were implemented or scored. This report records the inspected V1, the available repository profile, and the exact work needed to resume. Do not treat candidate hypotheses below as measured results.

## 1. Scope

- Christian's workstream: client-level feature engineering, isolated feature ablations, and evidence for V2 integration.
- Inspected the V1 feature builder, data contract, evaluation runner, model wrappers, dataset profile, and available tests.
- Out of scope: broad model tuning, replacing V1 models, changing the competitive submission, and work owned by temporal/recurrence or text/merchant workstreams.
- No raw data exists under `data/raw/ubs_2026`; only `data/raw/.gitkeep` is present. The checked-in profile is historical documentation and cannot substitute for row-level class analysis or a new validation run.

## 2. Baseline

- Official reference supplied by the team: Macro-F1 **0.2710243**, accuracy **0.2660**, 1,000 validation clients, valid submission.
- Baseline verified in this run: **NO**. The repository has no saved V1 metrics/predictions, the configured raw data directory is absent, and the local Python 3.11 environment points to a missing interpreter.
- Existing protocol: separate train/validation/test files with client disjointness enforced by `ubs.data.load_ubs_data`; features are generated from transaction histories strictly before `2026-01-01`; `evaluate_predictions` computes Macro-F1 over all eight labels.
- The current runner tunes the recurrence heuristic on `x_valid, y_valid`, and selects the ensemble weight on validation. Those validation labels affect the reported validation result. Also, description lift is fit using train labels and then used to transform those same train clients; use out-of-fold encodings for train features before treating this target-derived feature as leakage-safe.

## 3. Existing V1 features

| feature_family | existing_features | information_captured | potential_gap |
|---|---|---|---|
| Volume | `n_transactions`, `n_in`, `n_out` | Total and directional transaction counts | Activity density per active day; explicit active-day count |
| Amounts | Global `sum/mean/median/std/min/max`; per-currency `sum/mean/median/std`; fee sum/mean/positive share | Basic level and dispersion, plus currency-separated moments | Per-currency quantiles, IQR, range, robust CV/skew; global amount moments mix currencies and need caution |
| Direction | `out_share`, in/out counts, direction counts/shares | Direction mix | Explicit in-share and directional recent-vs-history changes (in-share is derivable) |
| Types and MCC | Train-vocabulary count/share for each `type` and `mcc`; `n_unique_type`, `n_unique_mcc` | Per-level frequency and simple cardinality | Dominant share, top-k share, HHI and entropy |
| Description / merchant proxy | `n_unique_description`; ordered description document for train-only TF-IDF; repeated-description streams; train-fitted class description lift | Text identity, repetition, cadence, and class association | Client-level concentration summaries; raw merchant field is absent from the contract |
| Currency | `n_unique_currency`; train-vocabulary count/share; per-currency amount moments | Currency composition and basic within-currency moments | Currency-specific robust quantiles and dispersion ratios |
| Diversity | `n_unique_mcc/description/type/currency` | Number of distinct values | Ratios to volume and effective diversity/entropy |
| Calendar / activity | History span, time since last transaction, rates per 30 days, absolute counts/frequencies in 7/14/30/60/90/180-day windows, active months, monthly count mean/std/CV, weekday/weekend shares | Recency, absolute recent activity, monthly variation, calendar mix | Active days/weeks; recent-to-prior contrasts; inactivity-period counts; observed-period-normalized rates |
| Gaps / recurrence | Mean/median/std event gaps; repeated-description count, repeat volume, top/mean appearances, stream interval/amount CV, regularity/due scores, periodicity bins, family-level lift/recurrence signals | Repeated behavior and cadence | Cross-description/MCC concentration; avoid overlap with temporal/recurrence workstream |
| Other | MCC/type-specific stream consistency; fees | Fee behavior and repeat-stream consistency | Class-wise value of fees is unknown without the raw data |

## 4. Missing information in V1

V1 has counts, shares, simple unique counts, absolute recent windows, monthly variability, and recurrence features. It does not explicitly encode the shape of a client's description/MCC frequency distribution (dominant/top-k share, HHI, entropy), active-day density, or recent-vs-prior-history changes. It also lacks amount quantiles and robust dispersion per currency. These are gaps in representation, not evidence of predictive value.

## 5. Hypotheses investigated

No new hypothesis was tested against labels because the raw data is unavailable. The following are proposed, explicitly unscored hypotheses:

| hypothesis | motivation and available evidence | feature definition | leakage check |
|---|---|---|---|
| H-FE-01: concentration | The checked-in profile reports 100% of train clients have repeated descriptions (median repeat appearances 3). A client's dominant description/MCC share could separate a narrow recurring-service history from general spending. The profile does not give these metrics by class. | Per client, `top1_share=max(count(value))/N`, `top3_share=sum(three largest counts)/N`, `HHI=sum((count(value)/N)^2)`, and normalized Shannon entropy `-sum(p*log(p))/log(K)` for description and MCC. | Counts use only that client's pre-cutoff history; do not use labels or fit across validation/test. |
| H-FE-02: recent vs prior history | V1 already has absolute recent-window counts. The profile says 99.5% of clients' last event is within 30 days, so a contrast may be more useful than recency alone; this is not class evidence. | For each 30/60/90-day window, compare event rate in the recent window with the immediately preceding equal-length window; use smoothed ratios/log-ratios and explicit zero-history flags. | Define windows ending at the fixed cutoff; exclude all events at/after cutoff. Do not use valid/test labels to tune window selection. |
| H-FE-03: active-day intensity | V1 counts active months but not distinct calendar days. Similar transaction totals can represent concentrated bursts or activity spread across the year. | Distinct active days; transactions per active day; active days / observed days; count of zero-activity gaps between active days. | Use timestamps before cutoff only; define observed span from each client's first event to cutoff or report both span conventions. |
| H-FE-04: within-currency amount shape | The profile reports four currencies and a long-tailed amount distribution (train p01 3.41, median 73.08, p99 7,983.11). Quantile spread may capture billing regularity. | Per currency: P10/P25/P50/P75/P90, IQR, range, robust CV (IQR / `abs(P50)`), and optional skew only with adequate observations. Never pool nominal amounts across currencies. | Compute independently from each client's pre-cutoff transactions; no exchange rates or population-fitted statistics. |
| H-FE-05: effective diversity | V1's `n_unique_*` ignores frequency balance. Two clients with equal unique counts may distribute transactions very differently. | Description/MCC entropy and effective number `exp(entropy)`; compare against existing unique counts and shares. | Per-client history only; no target-derived encoding. |

## 6. Experiments performed

No model/feature experiments were performed. Every row below is a planned, **not-run** slot, not a scored experiment. Runtime, metrics, delta, and affected classes are therefore unavailable; no result category is assigned.

| experiment_id | feature_group | features_added | macro_f1 | delta_vs_v1 | accuracy | affected_classes | result |
|---|---|---|---:|---:|---:|---|---|
| FE-001 | Concentration | Description/MCC top-1 and top-3 shares; HHI; entropy | — | — | — | — | NOT RUN — data unavailable |
| FE-002 | Effective diversity | Description/MCC normalized entropy and effective counts | — | — | — | — | NOT RUN — data unavailable |
| FE-003 | Recent vs history | Equal-length recent/prior event-rate contrasts for 30/60/90 days | — | — | — | — | NOT RUN — data unavailable |
| FE-004 | Amount shape | Per-currency quantiles, IQR, robust CV | — | — | — | — | NOT RUN — data unavailable |
| FE-005 | Active days | Active days, transactions per active day, inactivity gaps | — | — | — | — | NOT RUN — data unavailable |

Protocol planned: preserve the provided train/valid client split, baseline preprocessing and chosen V1 model/configuration; add one feature group at a time; record seed 42, runtime, per-class F1, and confusion matrix. The exact V1 selected model/config must be recovered from its run artifact before scoring, because those artifacts are absent here.

## 7. Best individual features

None established. There are no ablation results, and feature importance alone would not establish incremental value. Candidate definitions from H-FE-01 through H-FE-05 remain unranked.

## 8. Best feature groups

None established. Concentration is the first proposed experiment, not a demonstrated winner.

## 9. What worked

No new feature group has a measured improvement. Do not claim an increase over Macro-F1 0.2710243.

## 10. What did NOT work

No new features were run, so none can honestly be labeled neutral or worse. Do not interpret missing results as evidence of no improvement. The global V1 amount aggregates across currencies are semantically risky because the repository profile explicitly says amounts are in original currency units; retain them only as baseline features for a controlled comparison, and do not add more pooled monetary features.

## 11. Class-specific findings

No new feature has class-specific evidence. Per-class findings for `streaming`, `music`, `software`, `cloud`, `mobile`, `insurance`, `gym`, and `none` are all **NOT AVAILABLE** until validation labels and transactions are restored and the experiment runner can execute.

## 12. Best combined feature set

None. No new feature group has been scored individually, so combining any would violate the requested ablation sequence.

## 13. Recommended features for V2

| feature | evidence | impact | priority | integration_status |
|---|---|---|---|---|
| Description/MCC concentration (H-FE-01) | Dataset-wide recurrence is common in the stored profile; class separation not yet measured | Unknown | HIGH experiment priority | NOT APPROVED — run FE-001 first |
| Recent-vs-prior activity (H-FE-02) | V1 has absolute windows; profile shows recent transactions are common | Unknown | HIGH experiment priority | NOT APPROVED — coordinate with temporal owner |
| Active-day density (H-FE-03) | V1 has active months but no active-day count | Unknown | MEDIUM | NOT APPROVED — run FE-005 |
| Per-currency amount shape (H-FE-04) | Four currencies and long-tailed amounts in stored profile | Unknown | MEDIUM | NOT APPROVED — run FE-004 |
| Effective diversity (H-FE-05) | V1 has unique counts but no distribution balance | Unknown | MEDIUM | NOT APPROVED — run FE-002 |

Priority describes experiment order, not proven predictive value.

## 14. Features NOT recommended

- Do not integrate any proposed feature solely from this report; none has a measured delta.
- Do not add global monetary percentiles or ratios over pooled currencies without conversion or a currency-invariant definition.
- Do not add description token/merchant normalization features without coordination with the text/merchant owner.
- Do not add window/cadence features already being studied by the temporal/recurrence owner without an explicit joint ablation.

## 15. Leakage and risks

- New-feature leakage: none introduced; no new feature code was written.
- V1 cutoff: `ClientFeatureBuilder.transform` rejects timestamps at/after `2026-01-01`; `read_transactions` also enforces the cutoff.
- V1 target-derived feature risk: `description_lift_` is learned from train labels, then used when transforming those same training clients. Training rows should receive out-of-fold lift encodings; validation/test may use a mapping fit on train only.
- V1 validation-selection risk: `RecurrenceHeuristic.tune(x_valid, y_valid)` and the validation-tuned ensemble weight use validation labels before reporting validation score. Report this as a protocol risk; do not silently compare a differently tuned model to the official V1 score.
- Currency risk: global V1 amount moments pool unlike currency units; new amount features here are specified within currency.
- Overfitting/redundancy: many candidate concentration/diversity measures are mathematically correlated. Ablate one component at a time and compare with existing unique counts/shares.
- Split dependence: one 1,000-client validation set can make small deltas noisy. Preserve the official split for comparability and add train-only cross-fitting for target-derived transforms.

## 16. Code generated

| path | purpose | integration_value |
|---|---|---|
| `reports/handoff/christian_feature_engineering.md` | Evidence-bounded handoff and blocked experiment plan | Analysis only; no feature implementation |
| `reports/handoff/christian_feature_engineering_summary.md` | Short meeting summary | Analysis only |

No source code, tests, plots, datasets, or experiment outputs were generated.

## 17. Commits worth reviewing

| commit_hash | description | recommendation |
|---|---|---|
| — | No commit created in this blocked run | — |

## 18. Dependencies and conflicts

- **Temporal/recurrence:** FE-002/003/005 may overlap active-day, recent-window, cadence, and inactivity work. Coordinate definitions before implementation.
- **Text/merchants:** description concentration uses transaction counts by exact description; distinguish this from token/merchant normalization and avoid duplicating their work.
- **Models:** no model tuning or model replacement was performed. Keep the baseline estimator/config fixed in feature ablations.
- **Error analysis:** per-class gains and regressions cannot be assessed without validation predictions and labels.
- **Integration:** only feature groups with controlled, reproducible Macro-F1 evidence should be proposed for cherry-pick.

## 19. Recommended next experiment

Restore the official train/validation files and a working Python 3.11 runtime, then run **FE-001**: add only per-client exact-description and MCC top-1/top-3 shares, HHI, and normalized entropy to the fixed V1 numeric matrix; report distributions by class, missingness/outliers, correlation with existing unique counts/shares, validation Macro-F1/accuracy/per-class F1, and component ablations. Fit all quantities per client from pre-cutoff history only. Resolve the target-encoding and validation-calibration protocol risks before treating the comparison as leakage-safe.

## 20. Executive summary for V2 integration AI

- Christian's scope is client-level feature engineering; no model tuning was done.
- Official baseline reference is Macro-F1 0.2710243 / accuracy 0.2660, supplied by the team; it was not reproducible in this workspace.
- V1 already has volume, direction, simple diversity, categorical shares, amount moments, recent windows, monthly statistics, event gaps, and description recurrence.
- Candidate gaps are frequency concentration, recent-vs-prior contrasts, active-day density, and within-currency amount quantiles.
- The repository profile is descriptive and lacks class-wise measurements of those candidates.
- No new feature was implemented or evaluated; there is no best group or measured delta.
- Do not integrate candidate features until controlled ablations run on the official split.
- Dataset files are absent and the local `.venv` Python points to a missing installation, so tests and baseline could not execute.
- Audit V1's in-sample train description-lift encoding and validation-tuned recurrence/ensemble before claiming leakage-safe scores.
- First experiment after restoring inputs: FE-001 description/MCC concentration, with one-component ablations.
