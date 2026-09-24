# Team Handoff — Error Analysis

## 1. Scope

Person 5/7 analyzed where official V1 fails on the 1,000 validation clients and translated
those errors into isolated experiments for the other workstreams. We reproduced V1, inspected
its real implementation, computed class/pair metrics, compared correct vs incorrect clients,
and added analysis-only diagnostics. We did **not** change V1, tune or replace models, add
production features, use test labels, or alter the competitive submission.

## 2. Baseline

- Macro-F1: **0.2710243**.
- Accuracy: **0.2660**.
- Evaluated clients: **1000**.
- Baseline reference: `0199a8b` on the active analysis branch; submission SHA-256
  `b67d364e207d297b1a4d5898f0c37b6f62289cf56808882f5bdcfeb6c5795b0f`.
- Protocol: fit supervised transforms on 2,000 train clients, score fixed eight-class Macro-F1
  on 1,000 disjoint validation clients with the official evaluator, keep test labels absent.

| class | precision | recall | f1 | support | predicted_count |
| --- | --- | --- | --- | --- | --- |
| cloud | 0.200 | 0.517 | 0.288 | 89 | 230 |
| gym | 0.326 | 0.355 | 0.340 | 121 | 132 |
| insurance | 0.306 | 0.343 | 0.324 | 99 | 111 |
| mobile | 0.273 | 0.548 | 0.364 | 104 | 209 |
| music | 0.288 | 0.204 | 0.239 | 93 | 66 |
| software | 0.284 | 0.260 | 0.271 | 104 | 95 |
| streaming | 0.258 | 0.237 | 0.247 | 97 | 89 |
| none | 0.250 | 0.058 | 0.094 | 293 | 68 |

## 3. Error profile

- Weakest: `none` (F1 0.094), `music` (0.239), `streaming` (0.247).
- Relatively easiest: `mobile` (0.364), `gym` (0.340), `insurance` (0.324).
- Strongly overpredicted: `cloud` (+141) and `mobile` (+105).
- Strongly underpredicted: `none` (-225) and `music` (-27).

| class | support | predicted_count | prediction_minus_support |
| --- | --- | --- | --- |
| none | 293 | 68 | -225 |
| music | 93 | 66 | -27 |
| software | 104 | 95 | -9 |
| streaming | 97 | 89 | -8 |
| gym | 121 | 132 | 11 |
| insurance | 99 | 111 | 12 |
| mobile | 104 | 209 | 105 |
| cloud | 89 | 230 | 141 |

## 4. Main confusion pairs

| real_class | predicted_class | errors | percentage | severity |
| --- | --- | --- | --- | --- |
| none | cloud | 89 | 32.2 | CRITICAL |
| none | mobile | 43 | 15.6 | HIGH |
| none | insurance | 33 | 12.0 | HIGH |
| none | gym | 31 | 11.2 | HIGH |
| none | software | 30 | 10.9 | HIGH |
| streaming | mobile | 26 | 35.1 | MEDIUM |
| none | streaming | 26 | 9.4 | MEDIUM |
| none | music | 24 | 8.7 | MEDIUM |
| music | mobile | 21 | 28.4 | MEDIUM |
| gym | cloud | 20 | 25.6 | MEDIUM |

## 5. Pair-by-pair diagnosis

**Pair 1: `cloud` vs `none` — "
            f"93 crossed errors (4/89)**

### Hypothesis

V1 confunde `none` con `cloud` porque cualquier evidencia recurrente y un lift positivo compiten sin una puerta explícita de 'stream cualificado'; además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones observadas aparecen en mcc_5732_count, mcc_5732_share.

### Evidence

| feature | available_to_selected_v1 | auc_separability | mutual_information | effect_size_a_minus_b | median_a | median_b |
| --- | --- | --- | --- | --- | --- | --- |
| mcc_5732_count | False | 0.820 | 0.173 | 1.384 | 15.000 | 7.000 |
| mcc_5732_share | False | 0.815 | 0.156 | 1.378 | 0.197 | 0.107 |
| family_cloud_descriptions | False | 0.767 | 0.138 | 1.274 | 1.000 | 0.000 |
| family_cloud_description_lift | True | 0.777 | 0.125 | 1.255 | 1.491 | 0.000 |
| family_cloud_occurrences | True | 0.767 | 0.115 | 1.225 | 3.000 | 0.000 |

Text coverage by client:

| term | coverage_a_pct | coverage_b_pct | coverage_difference_pp |
| --- | --- | --- | --- |
| cloud access | 83.1 | 17.7 | 65.4 |
| cloud | 93.3 | 38.6 | 54.7 |
| backup | 74.2 | 27.0 | 47.2 |
| cloud backup | 71.9 | 24.9 | 47.0 |

### Recommended owner

TEMPORAL / RECURRENCE

### Recommended experiment

Run **EA-002** and report Macro-F1, both class F1 values and directional error counts.


---

**Pair 2: `mobile` vs `none` — "
            f"53 crossed errors (10/43)**

### Hypothesis

V1 confunde `none` con `mobile` porque cualquier evidencia recurrente y un lift positivo compiten sin una puerta explícita de 'stream cualificado'; además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones observadas aparecen en mcc_4814_count, mcc_4814_share.

### Evidence

| feature | available_to_selected_v1 | auc_separability | mutual_information | effect_size_a_minus_b | median_a | median_b |
| --- | --- | --- | --- | --- | --- | --- |
| mcc_4814_count | False | 0.839 | 0.154 | 1.416 | 9.000 | 1.000 |
| mcc_4814_share | False | 0.823 | 0.136 | 1.250 | 0.102 | 0.021 |
| family_mobile_recurrence_score | True | 0.729 | 0.080 | 0.927 | 0.522 | 0.000 |
| family_mobile_descriptions | False | 0.707 | 0.079 | 0.837 | 1.000 | 0.000 |
| family_mobile_occurrences | True | 0.702 | 0.077 | 0.757 | 3.500 | 0.000 |

Text coverage by client:

| term | coverage_a_pct | coverage_b_pct | coverage_difference_pp |
| --- | --- | --- | --- |
| contract | 77.9 | 28.7 | 49.2 |
| phone contract | 74.0 | 26.3 | 47.8 |
| phone | 76.9 | 29.7 | 47.2 |
| bill | 69.2 | 31.4 | 37.8 |

### Recommended owner

FEATURE ENGINEERING

### Recommended experiment

Run **EA-003** and report Macro-F1, both class F1 values and directional error counts.


---

**Pair 3: `gym` vs `none` — "
            f"42 crossed errors (11/31)**

### Hypothesis

V1 confunde `none` con `gym` porque cualquier evidencia recurrente y un lift positivo compiten sin una puerta explícita de 'stream cualificado'; además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones observadas aparecen en mcc_7997_count, mcc_7997_share.

### Evidence

| feature | available_to_selected_v1 | auc_separability | mutual_information | effect_size_a_minus_b | median_a | median_b |
| --- | --- | --- | --- | --- | --- | --- |
| mcc_7997_count | False | 0.855 | 0.228 | 1.354 | 8.000 | 0.000 |
| mcc_7997_share | False | 0.846 | 0.198 | 1.352 | 0.100 | 0.000 |
| family_gym_recurrence_score | True | 0.684 | 0.097 | 0.727 | 0.386 | 0.000 |
| family_gym_typical_amount | False | 0.688 | 0.092 | 0.888 | 38.095 | 0.000 |
| family_gym_descriptions | False | 0.685 | 0.074 | 0.846 | 1.000 | 0.000 |

Text coverage by client:

| term | coverage_a_pct | coverage_b_pct | coverage_difference_pp |
| --- | --- | --- | --- |
| gym | 89.3 | 41.6 | 47.6 |
| fit club | 74.4 | 27.6 | 46.7 |
| fit | 76.0 | 30.0 | 46.0 |
| club | 76.0 | 30.4 | 45.7 |

### Recommended owner

FEATURE ENGINEERING

### Recommended experiment

Run **EA-003** and report Macro-F1, both class F1 values and directional error counts.


---

**Pair 4: `insurance` vs `none` — "
            f"39 crossed errors (6/33)**

### Hypothesis

V1 confunde `none` con `insurance` porque cualquier evidencia recurrente y un lift positivo compiten sin una puerta explícita de 'stream cualificado'; además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones observadas aparecen en mcc_6300_count, mcc_6300_share.

### Evidence

| feature | available_to_selected_v1 | auc_separability | mutual_information | effect_size_a_minus_b | median_a | median_b |
| --- | --- | --- | --- | --- | --- | --- |
| mcc_6300_count | False | 0.836 | 0.234 | 1.331 | 10.000 | 0.000 |
| mcc_6300_share | False | 0.824 | 0.177 | 1.154 | 0.115 | 0.000 |
| family_insurance_recurrence_score | True | 0.665 | 0.093 | 0.790 | 0.000 | 0.000 |
| family_insurance_amount_similarity | False | 0.661 | 0.069 | 0.719 | 0.000 | 0.000 |
| family_insurance_description_lift | True | 0.654 | 0.070 | 0.781 | 0.000 | 0.000 |

Text coverage by client:

| term | coverage_a_pct | coverage_b_pct | coverage_difference_pp |
| --- | --- | --- | --- |
| cover plan | 67.7 | 23.2 | 44.5 |
| cover | 87.9 | 48.1 | 39.8 |
| policy | 65.7 | 27.6 | 38.0 |
| insurance | 57.6 | 21.5 | 36.1 |

### Recommended owner

FEATURE ENGINEERING

### Recommended experiment

Run **EA-003** and report Macro-F1, both class F1 values and directional error counts.


---

**Pair 5: `software` vs `none` — "
            f"37 crossed errors (7/30)**

### Hypothesis

V1 confunde `none` con `software` porque cualquier evidencia recurrente y un lift positivo compiten sin una puerta explícita de 'stream cualificado'; además, `none_bias=-1.0` penaliza la abstención. Las mayores separaciones observadas aparecen en mcc_5734_count, mcc_5734_share.

### Evidence

| feature | available_to_selected_v1 | auc_separability | mutual_information | effect_size_a_minus_b | median_a | median_b |
| --- | --- | --- | --- | --- | --- | --- |
| mcc_5734_count | False | 0.769 | 0.096 | 0.944 | 9.000 | 4.000 |
| mcc_5734_share | False | 0.743 | 0.046 | 0.845 | 0.109 | 0.053 |
| family_software_typical_amount | False | 0.668 | 0.063 | 0.801 | 0.000 | 0.000 |
| family_software_description_lift | True | 0.661 | 0.061 | 0.825 | 0.000 | 0.000 |
| family_software_amount_similarity | False | 0.670 | 0.050 | 0.791 | 0.000 | 0.000 |

Text coverage by client:

| term | coverage_a_pct | coverage_b_pct | coverage_difference_pp |
| --- | --- | --- | --- |
| saas billing | 67.3 | 24.9 | 42.4 |
| saas | 69.2 | 28.3 | 40.9 |
| premium plan | 72.1 | 35.2 | 37.0 |
| software | 61.5 | 27.3 | 34.2 |

### Recommended owner

FEATURE ENGINEERING

### Recommended experiment

Run **EA-003** and report Macro-F1, both class F1 values and directional error counts.


## 6. Correct vs incorrect clients

V1 understands clients when the real family's recurrence score, description lift and amount
stability are clearly present. Effect sizes are large for correct `gym` (family score d=2.06),
`insurance` (family descriptions d=2.39), `music` (description lift d=2.03), and `streaming`
(family score d=2.51). It fails when evidence is absent, stale, split across candidates or when
historical family text exists for a future `none` client. Correct `none` clients are unusual:
they have explicit learned `none`-description evidence (median score 1.595 versus 0 in errors),
showing that V1 does not treat `none` as absence of a valid future candidate.

Strongest correct/error contrast by requested dimension (`d`: correct minus incorrect):

| class | amount | description/merchant | history_vs_recent | periodicity | recency | regularity | volume |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | family_cloud_amount_similarity (d=+0.91) | family_mobile_descriptions (d=-0.94) | transactions_last_180d (d=-1.04) | periodicity_monthly_count (d=-0.43) | family_streaming_recency (d=-0.46) | family_cloud_regularity (d=+0.86) | n_out (d=-0.88) |
| gym | family_gym_amount_similarity (d=+1.98) | family_gym_description_lift (d=+1.94) | transactions_last_180d (d=-0.29) | periodicity_weekly_count (d=-0.40) | family_gym_recency (d=+0.63) | family_gym_regularity (d=+1.59) | n_transactions (d=-0.22) |
| insurance | family_insurance_amount_similarity (d=+2.17) | family_insurance_descriptions (d=+2.39) | transactions_last_14d (d=-0.41) | periodicity_biweekly_count (d=+0.53) | family_insurance_recency (d=+0.75) | family_insurance_regularity (d=+2.10) | n_transactions (d=+0.35) |
| mobile | family_mobile_amount_similarity (d=+1.55) | family_mobile_description_lift (d=+1.76) | diag_recent_30_share (d=-0.41) | periodicity_quarterly_count (d=-0.28) | family_mobile_recency (d=+0.41) | family_mobile_regularity (d=+1.51) | n_in (d=+0.05) |
| music | family_music_amount_similarity (d=+1.94) | family_music_description_lift (d=+2.03) | frequency_last_7d (d=+0.56) | periodicity_monthly_count (d=-0.62) | family_music_recency (d=+1.09) | family_music_regularity (d=+1.77) | n_transactions (d=-0.42) |
| software | family_software_amount_similarity (d=+1.82) | family_software_description_lift (d=+1.81) | transactions_last_90d (d=-0.54) | periodicity_quarterly_count (d=-0.33) | family_software_recency (d=+1.30) | family_software_regularity (d=+1.70) | n_transactions (d=-0.39) |
| streaming | family_streaming_amount_similarity (d=+1.84) | family_streaming_descriptions (d=+2.06) | diag_recent_30_share (d=-0.34) | periodicity_monthly_count (d=+0.42) | family_none_recency (d=-0.52) | family_streaming_regularity (d=+1.60) | n_in (d=+0.41) |
| none | family_none_amount_similarity (d=+1.17) | family_none_descriptions (d=+1.18) | diag_recent_amount_log_ratio (d=-0.28) | periodicity_weekly_count (d=-0.26) | family_none_recency (d=+0.45) | family_none_regularity (d=+1.10) | out_share (d=+0.22) |

Full reproducible table: `outputs/metrics/v2_error_analysis/correct_vs_incorrect.csv`.

## 7. What V1 currently sees

The feature builder computes 218 numeric aggregates, but selected V1 is
`recurrence_heuristic` and decides with 32 values: for each of eight labels it uses only
`family_*_recurrence_score`, `family_*_occurrences`, `family_*_description_lift`, and
`family_*_regularity`. Logistic sees global word/bigram TF-IDF and CatBoost sees the numeric
table, but neither was selected. Therefore computed amount, MCC, recency, periodicity and
composition features are not part of the official V1 decision.

| family | status | feature_count |
| --- | --- | --- |
| amount | analysis-only diagnostic | 1 |
| amount | computed but not used by selected V1 | 36 |
| merchant / description | computed but not used by selected V1 | 4 |
| other | computed but not used by selected V1 | 20 |
| recurrence | computed but not used by selected V1 | 28 |
| recurrence | selected V1 decision | 24 |
| temporal | computed but not used by selected V1 | 35 |
| temporal / recurrence | analysis-only diagnostic | 6 |
| text | computed but not used by selected V1 | 8 |
| text | selected V1 decision | 8 |
| text / merchant-description | analysis-only diagnostic | 5 |
| volume / composition | computed but not used by selected V1 | 55 |

## 8. What V1 is missing

- A candidate-validity/abstention gate: `none` is incorrectly modeled as another positive family.
- Candidate identity: family sums hide which exact stream generated the evidence.
- Stream lifecycle: recent vs historical cadence, missed cycles, expected phase and inactive
  streams.
- Candidate-linked MCC, amount stability, recency and direction, despite being available in data.
- Merchant normalization and robust text tied to a recurring stream rather than a client text bag.
- Conflict/margin between top candidates and calibrated uncertainty.
- OOF evidence separating real improvement from tuning to official validation.

## 9. Main findings

| priority | finding | evidence | affected_classes | likely_owner | estimated_difficulty | overfitting_risk | recommended_experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRÍTICA | V1 casi nunca se abstiene correctamente en none | F1=0.094, recall=0.058; 276 FN de none | none vs todas | MODELS / TUNING + FEATURE ENGINEERING | media | medio | NONE-01: gate/umbral de abstención y margen top1-top2 |
| CRÍTICA | La mayor fuga es none hacia cloud | 89 errores none→cloud; par bidireccional=93 | none / cloud | TEMPORAL / RECURRENCIA | media | medio | TEMP-01: stream candidato, fase y estabilidad reciente |
| ALTA | MCC específicos separan los cinco pares principales pero V1 los ignora | Top univariante: MCC 5732/4814/7997/6300/5734 para familia vs none | cloud/mobile/gym/insurance/software vs none | FEATURE ENGINEERING + MODELS / TUNING | baja-media | medio-alto | FEAT-02: MCC del stream candidato con ablación y OOF |
| ALTA | El texto discrimina, pero también aparece en clientes none | `cloud access`: 83.1% en cloud y 17.7% en none | familias vs none | TEXT / MERCHANTS | baja-media | medio | TEXT-01: texto por stream + lifecycle; no bolsa global sola |
| ALTA | Falta modelar histórico frente a comportamiento reciente por stream | V1 agrega recurrencia completa; no incorpora drift/lifecycle específico del candidato | none / cloud / mobile | TEMPORAL / RECURRENCIA | media | bajo-medio | TEMP-02: actividad 30/90d vs historia y stream activo/inactivo |
| MEDIA | La V1 elegida descarta 186 agregados y el TF-IDF calculado | La heurística decide con 32 columnas de 218 (4 por familia) | todas; especialmente music/streaming | MODELS / TUNING | media | medio-alto | MODEL-01: ranking de candidatos con ablaciones por familia |
| MEDIA | Heurística y mejor ML aciertan subconjuntos distintos | 199 victorias solo heurística y 213 solo Logistic en validación | todas | MODELS / TUNING + EXPERIMENT TRACKING | media | alto si se calibra en un único split | MODEL-02: meta-regla OOF o stacking leakage-safe |
| ALTA | La selección/calibración depende de un único valid de 1.000 clientes | none_bias, temperatura, modelo y ensemble se eligen sobre el mismo split | todas | EXPERIMENT TRACKING + INTEGRATION / VALIDATION | media | reduce riesgo | VALID-01: OOF por cliente y estabilidad de F1 por clase |

## 10. Recommendations by team area

| team_area | finding | evidence | recommended_action | priority |
| --- | --- | --- | --- | --- |
| TEMPORAL / RECURRENCE | Full-history evidence does not identify ended streams. | All top five pairs are family-vs-none. | EA-002: candidate lifecycle and missed cycles. | CRITICAL |
| TEXT / MERCHANTS | Family phrases discriminate but leak into none histories. | cloud access coverage: 83.1% cloud vs 17.7% none. | EA-004: candidate-conditioned normalized text. | HIGH |
| FEATURE ENGINEERING | Strong MCC/candidate signals are not used by selected V1. | Family MCC is top univariate feature for five main pairs. | EA-003: candidate table and MCC ablation. | HIGH |
| MODELS / TUNING | V1 models none as a positive family instead of abstention. | 17/293 none correct; none_bias=-1.0. | EA-001: OOF-calibrated candidate gate. | CRITICAL |
| EXPERIMENT TRACKING | Selection and calibration share official validation. | Model, temperature and bias use the same 1,000 clients. | EA-005: fixed client-level OOF protocol. | HIGH |
| INTEGRATION / VALIDATION | Accuracy can hide rare-class regression. | none F1=0.094 despite accuracy=0.266. | EA-006: class/pair regression gates. | HIGH |

## 11. Experiments recommended

EXPERIMENT ID:
EA-001

OWNER:
MODELS / TUNING

PROBLEM:
V1 predicts `none` only 68 times although its support is 293.

EVIDENCE:
Only 17/293 `none` clients are correct; `none_bias=-1.0` suppresses abstention.

CHANGE TO TEST:
Add a candidate-validity gate using top-1 score, top-1/top-2 margin and negative evidence; calibrate only with train OOF predictions.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
F1 and recall of `none`; worst per-family F1 delta.

SUCCESS CRITERIA:
OOF Macro-F1 improves by at least 0.01 and `none` F1 by at least 0.05, with no positive-family F1 falling more than 0.03.

RISK:
Threshold overfitting to one split and excessive abstention.

---

EXPERIMENT ID:
EA-002

OWNER:
TEMPORAL / RECURRENCE

PROBLEM:
Historical family evidence remains active even when the stream may have ended.

EVIDENCE:
The five largest class pairs are family-vs-`none`; V1 sums full-history streams.

CHANGE TO TEST:
Add candidate-stream phase error, recent interval CV, expected-next-date residual, missed-cycle count and active/inactive status.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
F1 for `none`, `cloud`, and `mobile`; family-to-`none` confusion counts.

SUCCESS CRITERIA:
At least 10% relative reduction in the top two pair errors without lower Macro-F1.

RISK:
Cutoff leakage or encoding future transactions when constructing lifecycle labels.

---

EXPERIMENT ID:
EA-003

OWNER:
FEATURE ENGINEERING

PROBLEM:
The selected heuristic ignores strong category evidence already calculated.

EVIDENCE:
MCC 5732/4814/7997/6300/5734 is the strongest univariate separator for the five largest family-vs-`none` pairs.

CHANGE TO TEST:
Attach MCC count/share, amount stability and recency to each recurring candidate; ablate MCC alone, then the complete candidate table.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
F1 of the five affected families and `none`; pair error counts.

SUCCESS CRITERIA:
OOF Macro-F1 gain of at least 0.01 that remains positive without raw text.

RISK:
Synthetic-generator shortcut; MCC must be audited across folds and test-like data.

---

EXPERIMENT ID:
EA-004

OWNER:
TEXT / MERCHANTS

PROBLEM:
Exact descriptions are brittle, but global text also creates false positives.

EVIDENCE:
`cloud access` appears in 83.1% of cloud clients but also 17.7% of `none`; equivalent overlap exists for the other family phrases.

CHANGE TO TEST:
Normalize merchant-like descriptions and test character 3-5 grams only at candidate stream level, combined with stream recency/lifecycle.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
F1 of `music`, `streaming`, and `mobile`; text-driven `none` false positives.

SUCCESS CRITERIA:
Positive OOF Macro-F1 delta and fewer text-driven `none` false positives.

RISK:
Memorizing generator vocabulary or increasing false positives from historical text.

---

EXPERIMENT ID:
EA-005

OWNER:
EXPERIMENT TRACKING

PROBLEM:
Model, bias and ensemble are selected on the same official validation split.

EVIDENCE:
Current V1 tunes `none_bias`, temperature and model selection on 1,000 valid clients.

CHANGE TO TEST:
Create fixed stratified client folds inside train; log fold Macro-F1, per-class F1, confusion pairs, seed and exact feature set.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
Mean/std of per-class F1 and top-pair error counts across folds.

SUCCESS CRITERIA:
Every proposed V2 change has OOF deltas and variance before official validation.

RISK:
Incorrect fold-level fitting of text/lift features causing leakage.

---

EXPERIMENT ID:
EA-006

OWNER:
INTEGRATION / VALIDATION

PROBLEM:
V2 components may improve accuracy while regressing rare-class Macro-F1.

EVIDENCE:
V1 accuracy and Macro-F1 rank models differently; `none` dominates the error volume.

CHANGE TO TEST:
Add gates for official Macro-F1, every class F1, top confusion counts, ID/order validity and deterministic prediction hash.

PRIMARY METRIC:
Macro-F1.

SECONDARY METRIC:
Minimum class F1, submission validity and run-to-run identity.

SUCCESS CRITERIA:
Integration rejects any unexplained V1 regression and records intended trade-offs.

RISK:
Overly rigid gates blocking a justified class trade-off; allow reviewed exceptions.


## 12. What NOT to do

- Do not assume more global TF-IDF solves the problem; family phrases also occur in `none`
  histories.
- Do not tune gates, class thresholds or ensemble weights directly on the official validation again.
- Do not use raw `client_id`, test labels, post-cutoff rows, or validation-fitted description lifts.
- Do not aggregate monetary values across currencies without a documented conversion or
  per-currency split.
- Do not promote MCC without OOF/stability checks; it may be a synthetic-generator shortcut.
- Do not interpret mutual information/effect size as causal or integrate every diagnostic feature.
- Do not replace V1 before an isolated experiment passes class-level and pair-level regression
  gates.

## 13. Code generated

| path | purpose | integration_value |
| --- | --- | --- |
| scripts/analyze_v1_errors.py | Reproduce metrics and all diagnostics/reports. | ANALYSIS ONLY; reusable quality-gate inputs. |
| reports/handoff/santiago_error_analysis.md | Technical handoff for the V2 integration AI. | READ FIRST; no production code. |
| reports/handoff/santiago_error_analysis_summary.md | Under-two-minute team meeting summary. | MEETING AID. |
| reports/figures/error_analysis/*.png | Four compact error visualizations. | DOCUMENTATION ONLY. |

Generated CSV/JSON diagnostics under `outputs/metrics/v2_error_analysis/` remain ignored and
contain the complete tables. No dataset, client-level report, model artifact, or secret is tracked.

## 14. Commits worth reviewing

No commit was created in this environment. All files are **ANALYSIS ONLY** until reviewed. The
recommended future commit classification is `ANALYSIS ONLY`; cherry-pick production features
only from the owner experiments after independent validation.

## 15. Dependencies and conflicts

- Temporal/recurrence may independently create lifecycle features; reuse their canonical names.
- Text/merchant work may normalize descriptions; this analysis should consume, not duplicate, it.
- Feature engineering may create candidate tables; MCC must attach to that shared interface.
- Models/tuning owns the gate/ranker; avoid parallel threshold implementations.
- Tracking must fit description lifts/text inside each fold to prevent leakage.
- Integration should retain V1 artifacts and this report as regression references.

## 16. Recommended V2 priorities

1. **CRITICAL:** EA-001 candidate-validity gate for `none`, calibrated OOF.
2. **CRITICAL:** EA-002 stream lifecycle/phase features.
3. **HIGH:** EA-003 candidate-level MCC and recurrence evidence with shortcut audit.
4. **HIGH:** EA-005 client-level OOF protocol before official validation.
5. **HIGH:** EA-004 candidate-conditioned text/merchant normalization.
6. **HIGH:** EA-006 per-class and pair-level integration gates.

## 17. Executive summary for V2 integration AI

- V1 is reproducible at Macro-F1 0.2710243 and accuracy 0.2660.
- Its dominant failure is `none`: 17/293 correct, F1 0.094, 276 false negatives.
- `none→cloud` (89), `none→mobile` (43), and `none→insurance` (33) dominate errors.
- Cloud and mobile are overpredicted by 141 and 105 clients; none is underpredicted by 225.
- V1 selected a recurrence heuristic using only 32 of 218 computed numeric features.
- It models `none` as positive learned-description evidence, not absence of a valid candidate.
- Add an OOF-calibrated candidate-validity gate before choosing a family (EA-001).
- Add stream lifecycle, phase, missed-cycle and active/inactive signals (EA-002).
- Test candidate-linked MCC; it separates all five top pairs but may be a shortcut (EA-003).
- Condition normalized text on the candidate stream and lifecycle, not a global text bag (EA-004).
- Establish fixed client-level OOF tracking before touching official validation (EA-005).
- Preserve per-class F1 and confusion-pair gates during integration (EA-006).
- Do not use test labels, post-cutoff data, client IDs, or validation-fitted transforms.
- Do not assume global text or MCC generalizes without fold stability and ablation evidence.
- Keep V1 unchanged as the regression baseline until isolated experiments pass these gates.
