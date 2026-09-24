# Team Handoff — Error Analysis

## 1. Scope

Role: person **5/7 — Error Analysis / diagnóstico de V1** (executed on branch
`esteban-v2-models` and stored under `Reports/` per Esteban workspace choice).

Analyzed:

- Reproduction of the official V1 winning model (`recurrence_heuristic`)
- Confusion matrix, per-class precision/recall/F1/support
- Top confusion pairs and feature deltas vs correctly classified clients
- Correct vs incorrect clients overall and for the weakest classes
- What features the winning V1 path actually uses

Not analyzed / out of scope:

- Building V2 models or replacing the competitive submission
- Broad feature engineering implementation
- Hyperparameter search as a goal (diagnosis only)
- Unlabeled pretrain corpus

## 2. Baseline

Reproduced with `scripts/analyze_v1_errors.py` on `data/raw/ubs_2026`:

| metric | value |
|---|---:|
| Macro-F1 | **0.2710242658492452** |
| Accuracy | **0.2660** |
| Validation clients | **1000** |
| Winning model | `recurrence_heuristic` |
| Calibration on valid | `none_bias=-1.00`, `temperature=1.00` |

Per-class F1 (validation):

| class | precision | recall | f1 | support | predicted |
|---|---:|---:|---:|---:|---:|
| mobile | 0.273 | 0.548 | 0.364 | 104 | 209 |
| gym | 0.326 | 0.355 | 0.340 | 121 | 132 |
| insurance | 0.306 | 0.343 | 0.324 | 99 | 111 |
| cloud | 0.200 | 0.517 | 0.288 | 89 | 230 |
| software | 0.284 | 0.260 | 0.271 | 104 | 95 |
| streaming | 0.258 | 0.237 | 0.247 | 97 | 89 |
| music | 0.288 | 0.204 | 0.239 | 93 | 66 |
| none | 0.250 | 0.058 | 0.094 | 293 | 68 |

Protocol: train-fit description lifts inside `ClientFeatureBuilder`; heuristic
scores on valid; official 8-class macro-F1. Baseline reference commit on this
branch lineage: `0199a8b` (main V1) + local model-search commits not required
for this diagnosis.

Baseline verified: **YES** (exact Macro-F1 match).

## 3. Error profile

- **Worst F1:** `none` (0.094), `music` (0.239), `streaming` (0.247)
- **Best F1:** `mobile` (0.364), `gym` (0.340), `insurance` (0.324)
- **Overpredicted:** `cloud` (pred/support ≈ 2.58), `mobile` (≈ 2.01), `insurance` (≈ 1.12)
- **Underpredicted:** `none` (pred/support ≈ 0.23), `music` (≈ 0.71), `streaming` (≈ 0.92)

Interpretation: V1 behaves like a **family hunter**. It rarely emits `none`
(68 predictions vs 293 true) and floods `cloud`/`mobile`.

## 4. Main confusion pairs

| real_class | predicted_class | errors | percentage | hypothesis |
|---|---|---:|---:|---|
| none | cloud | 89 | 32.2% of none errors | Family lift + recurrence score prefers cloud streams that also appear in none clients |
| none | mobile | 43 | 15.6% | Same pattern for mobile candidate descriptions |
| none | insurance | 33 | 12.0% | Regular/stable streams mistaken for insurance family |
| none | gym | 31 | 11.2% | High regularity/stability scores without true gym target |
| none | software | 30 | 10.9% | Software-like description lift on non-target clients |
| streaming | mobile | 26 | 35.1% of streaming errors | Digital-subscription overlap; text not used by winner |
| music | mobile | 21 | 28.4% of music errors | Same digital-sub confusion |

Full table: `Reports/tables/confusion_pairs.csv`.

## 5. Pair-by-pair diagnosis

### none → cloud (89)

### Hypothesis
V1 ranks `family_cloud_recurrence_score` highest for many true-`none` clients
because description-lift associations from train fire on generic cloud-like
strings that also occur in none histories.

### Evidence
- Among these 89 errors, argmax of `family_*_recurrence_score` is **cloud in 86.5%**
  of clients (`Reports/v1_error_analysis.json`).
- vs correct none: errors have **fewer unique descriptions** (median 24 vs 32,
  effect size −0.96) and **higher amount_median** (89 vs 67).

### Recommended owner
TEXT / MERCHANTS (+ FEATURE ENGINEERING for none-interruption features)

### Recommended experiment
Train-only negative evidence: penalize family scores when candidate streams also
have high coverage among train `none`, or require exclusive lift.

### none → mobile / insurance / gym / software

### Hypothesis
Generic recurrence strength (regular/stable streams) is treated as positive
evidence for a target family, but Javi’s EDA already showed none clients also
have regular series.

### Evidence
- none→gym errors: higher `regular_stream_count` (median 8 vs 6) and
  `stable_amount_stream_count` (3 vs 2) than correct none.
- none→software/insurance: elevated `best_recurrence_score` vs correct none.
- Correct none clients are slightly **more recent** (`days_since_last` median
  1.5 vs 3.7 on wrong none) and have longer history.

### Recommended owner
TEMPORAL / RECURRENCE + FEATURE ENGINEERING

### Recommended experiment
Add relative recency (`days_since_last / median_interval`) and “interrupted
candidate” flags; do **not** infer none from lack of generic regularity.

### streaming → mobile (26) and music → mobile (21)

### Hypothesis
Digital subscriptions share cadence/amount patterns; the winning heuristic does
**not** use TF-IDF text, so mobile lift dominates ambiguous streams.

### Evidence
- streaming→mobile is 35% of streaming mistakes; music→mobile 28% of music mistakes.
- Winning model `text_in_winning_model=false` while TF-IDF exists only in logistic
  (weaker on macro-F1).

### Recommended owner
TEXT / MERCHANTS

### Recommended experiment
Alias normalization + exclusive mobile/music/streaming lexicons; evaluate family
score after alias merge.

## 6. Correct vs incorrect clients

Overall correct vs incorrect effect sizes are small (weekend_share, out_share,
history_days ~0.1), meaning **global volume alone does not explain errors**.

Class-conditional patterns matter more:

- **none correct vs incorrect:** correct none have longer `history_days` and
  more recent activity; incorrect none look like “active recurring payers”.
- **music correct vs incorrect:** incorrect music shows **higher**
  `best_recurrence_score` and more monthly periodicity — V1 sees recurrence but
  assigns the wrong family.
- **streaming correct vs incorrect:** incorrect streaming has higher stream
  regularity means — again, recurrence without the right merchant family.

Conclusion: V1 “understands” clients with a sharp family-aligned stream; it
fails on clients with **generic recurrence** or **ambiguous digital text**.

## 7. What V1 currently sees

Winning path (`RecurrenceHeuristic`) uses **family_* aggregates** derived from:

| family | what it captures | superficial? | missing |
|---|---|---|---|
| Volume | n_transactions, window counts | useful context | not decision-critical |
| Amounts | mean/std/median, currency splits | yes | currency-safe stability per stream already partial |
| Temporal | history_days, recency, DOW, monthly CV | yes | calendar due-dates; relative gap/interval |
| Recurrence | repeated streams, regularity, due_score, periodicity bins | core signal | support≥3 discipline; interruption |
| Family scores | description lift × recurrence (train-only lift) | **decisive** | exclusivity vs none; aliasing |
| Categorical MCC/type shares | bag-of-codes | weak | merchant identity |
| Text TF-IDF | available in logistic only | **unused by winner** | aliases, char n-grams for digital classes |

Numeric feature count on valid matrix: **218**.

## 8. What V1 is missing

Evidence-backed gaps:

1. **None-vs-family exclusivity** — family lifts fire on none clients (cloud 86.5%).
2. **Interrupted / overdue candidate streams** — relative recency not used as
   “should have paid already → maybe stopped → none”.
3. **Text discrimination for mobile/music/streaming** — winner ignores TF-IDF.
4. **Support-aware regularity** — high regularity on none→gym errors; need ≥3
   events (Javi).
5. **Calibration toward none** — only 68 none predictions vs 293 true (later
   model-search calibrator helps, but that is outside this diagnosis mandate).

## 9. Main findings

1. V1’s bottleneck is **`none` recall (0.058)**, not rare-family detection alone.
2. Largest single confusion is **none→cloud (89)** driven by family_cloud scores.
3. Overprediction of **cloud/mobile** is structural (2× support).
4. Recurrence strength without family exclusivity **hurts none**.
5. **music/streaming** confuse with **mobile**; text is underused by the winner.
6. Correct none clients look quieter/recent; wrong none look like subscribers.
7. Global correct-vs-incorrect feature gaps are weak; class-conditional gaps are
   actionable.
8. Logistic TF-IDF exists but loses to the heuristic on macro-F1 — text needs
   better targets (aliases), not just “add TF-IDF again”.

## 10. Recommendations by team area

| team_area | finding | evidence | recommended_action | priority |
|---|---|---|---|---|
| TEMPORAL / RECURRENCE | Regular streams ≠ target family | none→gym/software high regularity | relative recency + support≥3 | CRITICAL |
| TEXT / MERCHANTS | Digital class collisions | streaming/music→mobile; none→cloud lift | aliases + exclusive lexicons | CRITICAL |
| FEATURE ENGINEERING | Family scores lack none penalty | 86.5% cloud argmax on none→cloud | none-coverage penalty / exclusive lift | CRITICAL |
| MODELS / TUNING | none under-emitted | 68 vs 293 preds | calibration/`none_bias` after feature fixes | HIGH |
| EXPERIMENT TRACKING | Need class-wise logging | none F1 hides in accuracy | force per-class F1 in every row | HIGH |
| INTEGRATION / VALIDATION | Selection on valid for bias | heuristic tune on valid | document optimism; freeze protocol | MEDIUM |
| ERROR ANALYSIS | Done for V1 | this report | refresh after Features V2 lands | LOW |

## 11. Experiments recommended

### EXPERIMENT ID: EA-001

OWNER: FEATURE ENGINEERING (+ TEXT)

PROBLEM: none→cloud and generic family false positives.

EVIDENCE: 89 none→cloud; family score argmax cloud 86.5%.

CHANGE TO TEST: Add train-only feature
`family_score_exclusive = recurrence_score * lift * (1 - none_coverage)`
or subtract none-smoothed description prior.

PRIMARY METRIC: Macro-F1.

SECONDARY METRIC: none F1; cloud precision.

SUCCESS CRITERIA: Macro-F1 ≥ 0.29 and none F1 ≥ 0.15 without collapsing mobile F1.

RISK: Over-penalizing true cloud clients; must fit prior on train only.

### EXPERIMENT ID: EA-002

OWNER: TEMPORAL / RECURRENCE

PROBLEM: Regular none clients predicted as gym/software/insurance.

EVIDENCE: none→gym errors have higher regular/stable stream counts than correct none.

CHANGE TO TEST: Features `interval_count`, `relative_recency=days_since_last/median_interval`,
`interrupted_flag = relative_recency > 2` on candidate streams (Javi definition).

PRIMARY METRIC: Macro-F1.

SECONDARY METRIC: none F1; gym/software precision.

SUCCESS CRITERIA: +0.01 Macro-F1 or +0.05 none F1 with stable family F1.

RISK: Threshold leakage if tuned carelessly on valid; keep grid tiny.

### EXPERIMENT ID: EA-003

OWNER: TEXT / MERCHANTS

PROBLEM: streaming/music ↔ mobile.

EVIDENCE: 26 streaming→mobile; 21 music→mobile; winner uses no text.

CHANGE TO TEST: Alias map + per-class keyword lists; rebuild description keys before
streams; keep heuristic otherwise fixed.

PRIMARY METRIC: Macro-F1.

SECONDARY METRIC: music/streaming/mobile F1.

SUCCESS CRITERIA: music or streaming F1 +0.05 without none regression.

RISK: Train-overfit aliases; validate coverage drop train→valid (Javi warned).

### EXPERIMENT ID: EA-004

OWNER: MODELS / TUNING

PROBLEM: none under-prediction after features stabilize.

EVIDENCE: 68 none preds vs 293 true under V1 heuristic.

CHANGE TO TEST: Freeze features; tune only temperature/`none_bias` or calibrated
CatBoost (already explored elsewhere); report class F1.

PRIMARY METRIC: Macro-F1.

SECONDARY METRIC: none F1.

SUCCESS CRITERIA: Macro-F1 ≥ 0.30 with documented valid-tuning risk.

RISK: Selection optimism on valid (same class as V1).

### EXPERIMENT ID: EA-005

OWNER: EXPERIMENT TRACKING

PROBLEM: Accuracy can rise while macro-F1/`none` burn.

EVIDENCE: Several ML runs historically had higher accuracy than heuristic but
worse macro-F1.

CHANGE TO TEST: Require every logged experiment to store per-class F1 + top-3
confusion pairs.

PRIMARY METRIC: Macro-F1.

SECONDARY METRIC: completeness of logs.

SUCCESS CRITERIA: Template adopted by all feature/model PRs.

RISK: None (process).

## 12. What NOT to do

- Do **not** treat “has regular streams” as evidence against `none`.
- Do **not** assume adding raw TF-IDF again will beat the heuristic without
  aliasing / exclusivity.
- Do **not** use validation labels to learn description→family maps.
- Do **not** optimize accuracy as the primary KPI.
- Do **not** delete or rewrite `scripts/run_ubs_baseline.py`.
- Do **not** assume client-id numeric suffixes are features.
- Weak idea without evidence here: “short history fallback” — almost no ultra-short
  clients in this dataset (Javi EDA).

## 13. Code generated

| path | purpose | integration_value |
|---|---|---|
| `scripts/analyze_v1_errors.py` | Reproduce V1 heuristic + emit tables/figures/JSON | ANALYSIS / keep for V2 refresh |
| `Reports/v1_error_analysis.json` | Machine-readable diagnosis | ANALYSIS ONLY |
| `Reports/tables/*.csv` | Confusion pairs, per-class, comparisons | ANALYSIS ONLY |
| `Reports/figures/error_analysis/*.png` | Confusion/F1/pair/box plots | ANALYSIS ONLY |
| `Reports/santiago_error_analysis.md` | Full handoff | MUST READ for integrator |
| `Reports/santiago_error_analysis_summary.md` | Meeting summary | MUST READ for standup |

No V1 production modules were modified for this diagnosis.

## 14. Commits worth reviewing

| commit_hash | description | recommendation |
|---|---|---|
| `fcc7a6b` | analysis: add reproducible V1 error diagnostics and Reports handoff | CHERRY-PICK RECOMMENDED (script + Reports) |
| `a056ebd` | docs: record error-analysis commit hash in Reports handoff | ANALYSIS ONLY |

## 15. Dependencies and conflicts

- Overlaps Javi (recurrence support/recency) and Laura/text owners (aliases).
- Overlaps Esteban model-search calibrator work: useful **after** feature fixes,
  not a substitute for none exclusivity.
- Santiago error-analysis remote branch naming differs; this handoff lives on
  `esteban-v2-models` under `Reports/` as requested locally.
- Integration must avoid double-counting calibration gains and feature gains.

## 16. Recommended V2 priorities

1. Exclusive family scores / none coverage penalty (EA-001)
2. Relative recency + interrupted candidates (EA-002)
3. Mobile/music/streaming aliasing (EA-003)
4. Then none-oriented calibration (EA-004)
5. Enforce per-class logging (EA-005)

## 17. Executive summary for V2 integration AI

V1 (`recurrence_heuristic`, Macro-F1 0.271) fails mainly by **under-predicting
`none`** (recall 0.058) and **over-predicting cloud/mobile**. The single largest
error is **none→cloud (89)**; on those clients `family_cloud_recurrence_score`
wins 86.5% of the time. Regular/stable streams also push none→gym/software.
Secondary failure mode: **streaming/music→mobile** while the winning model
**ignores TF-IDF text**. V2 should prioritize (1) none-exclusive family scoring,
(2) support-aware relative recency/interruption features, (3) digital alias
disambiguation, then (4) calibration. Do not equate generic recurrence with a
target family. Do not replace the V1 baseline runner. Re-run
`scripts/analyze_v1_errors.py` after feature changes to verify none recall and
top confusion pairs move.
