# V4 merchant-text corruption — Ginestar

Branch `exp/v4-shift-corruption-ginestar`. Frozen reference `origin/baseline/v4-frozen` at `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`.
V3-A from `scripts/run_ubs_v3.py`; historical VALID Macro-F1 0.424111097737.
That score is provenance only, never a selection threshold. This experiment reads no VALID/TEST labels.

## Findings

The clean control reproduces frozen V3-A TRAIN OOF Macro-F1 to 1e-9 on all 2,000 clients. A falls from 0.459794 to 0.373384 under severe corruption: an absolute loss of 0.086410 and a relative loss of 18.79%, with losses in all five folds. V2 falls by 3.62% and the history classifier by 2.70%. V2 overtakes A only at the severe level; these results do not select or promote a replacement baseline.

Cloud, streaming and music lose the most F1. A's `none` predictions rise from 304 to 1,077 against a fixed true support of 597. The isolated opaque stream mask leaves history-control scores unchanged while hurting A, which implicates identity dependence. Fragmentation is the most damaging isolated operation for the history control. This separates sensitivity to unfamiliar identity from sensitivity to description-based grouping; it does not provide an additive causal decomposition.

The suite is a reusable stress envelope, not a calibrated simulator of the observed cohort shift. Severe produces 77.67% unseen payment descriptions, versus 1.17% in VALID and 0.99% in TEST. Those external inputs were inspected only after TRAIN scoring and did not change any parameter.

## Protocol

The prompt placeholders were resolved from the fetched frozen baseline ref, which exactly matched initial HEAD. The unrelated untracked `scripts/emergency_submission.py` was preserved.
Read V3 discovery/protocol, dataset analysis, mapping, NONE-gate, calibration, disagreement and seven-way synthesis. They show that clean OOF gains can fail under transfer; these historical reports already disclose reused VALID results. No names or thresholds from those results enter this suite.

Five outer stratified folds of unique TRAIN clients, sorted by ID, seed 42: identical to the baseline. Each model is fitted once per fold on clean histories; all views of a held client remain in that fold. V3's five inner folds still cross-fit supervised identity. Corruptions are inference stress, not data augmentation. The unchanged V3 model and 75/25 weights are reused. A, V2 and the history classifier share those fits. No baseline hyperparameter or corruption level is selected using scores.

The frozen CatBoost recipe is 300 iterations, depth 4, learning rate 0.05, balanced class weights and seed 42. A uses the existing history and supervised identity features, with 75% classifier / 25% periodicity heuristic scores. The full frozen model is fitted unchanged, even though only A, V2 and history-control components enter this scorecard.

Only descriptions in streams anchored by outgoing card payments change, including matching transfers/refunds. Every date, amount, MCC, type, currency, direction, fee, client and target stays intact. No event is dropped. IDs only key deterministic randomness/folds. The generic vocabulary is learned independently inside each outer fit, without labels.

Parameters fixed before the first model fit; seed 20260925. Family order: generic → opaque stream mask → compatible collision → temporal alias fragmentation → token abbreviation → event dropout.

| level | dropout | generic | stream_mask | fragmentation | collision | decoration | aliases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2 |
| mild | 0.02 | 0.05 | 0.05 | 0.08 | 0.04 | 0.08 | 2 |
| medium | 0.05 | 0.12 | 0.12 | 0.18 | 0.1 | 0.18 | 3 |
| severe | 0.1 | 0.25 | 0.25 | 0.35 | 0.2 | 0.3 | 4 |

Dropout uses an explicit missing token on sampled events. Generic masking uses pairs of TRAIN tokens occurring in >=3 MCCs and >=10 distinct descriptions (top 12 by breadth); a neutral sentinel is the fallback. Opaque masking replaces a whole stream by a unique neutral alias, preserving its grouping. Fragmentation adds 2/3/4 temporally contiguous suffix aliases to selected repeated streams. Collision shares a synthetic alias only within equal MCC/type/currency/direction. Decoration abbreviates tokens longer than four characters to three and adds punctuation. Random draws are SHA-256 keyed and independent across families; composition can overwrite earlier operations. Probabilities describe operation selection, not the eventual unique fraction changed.

## Pooled TRAIN OOF scorecard

| model | view | Macro-F1 | accuracy | F1 drop | relative drop | fold mean ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| A | clean | 0.459793827 | 0.4710 | +0.000000 | 0.00% | 0.457708 ± 0.022165 |
| A | mild | 0.452061182 | 0.4745 | +0.007733 | 1.68% | 0.449616 ± 0.017286 |
| A | medium | 0.432375381 | 0.4740 | +0.027418 | 5.96% | 0.430927 ± 0.009245 |
| A | severe | 0.373383822 | 0.4430 | +0.086410 | 18.79% | 0.370917 ± 0.031174 |
| V2 | clean | 0.408883545 | 0.4230 | +0.000000 | 0.00% | 0.407361 ± 0.019453 |
| V2 | mild | 0.401278861 | 0.4170 | +0.007605 | 1.86% | 0.399817 ± 0.017923 |
| V2 | medium | 0.393705321 | 0.4100 | +0.015178 | 3.71% | 0.392364 ± 0.017726 |
| V2 | severe | 0.394071435 | 0.4120 | +0.014812 | 3.62% | 0.393058 ± 0.022719 |
| history_control | clean | 0.399766788 | 0.4200 | +0.000000 | 0.00% | 0.398532 ± 0.019513 |
| history_control | mild | 0.390421578 | 0.4115 | +0.009345 | 2.34% | 0.389498 ± 0.026760 |
| history_control | medium | 0.391650878 | 0.4140 | +0.008116 | 2.03% | 0.390448 ± 0.018406 |
| history_control | severe | 0.388967735 | 0.4125 | +0.010799 | 2.70% | 0.387478 ± 0.023492 |

Drops = clean minus corrupted (negative means improvement). Headline is pooled fixed-eight-class F1, not the mean of fold F1. Fold SD is descriptive over five dependent fits.

## A: class F1, prediction counts and confusion

### clean

| class | F1 | drop vs clean | predicted | support |
| --- | --- | --- | --- | --- |
| cloud | 0.463964 | +0.000000 | 254 | 190 |
| gym | 0.469526 | +0.000000 | 253 | 190 |
| insurance | 0.474438 | +0.000000 | 275 | 214 |
| mobile | 0.450704 | +0.000000 | 235 | 191 |
| music | 0.395062 | +0.000000 | 207 | 198 |
| software | 0.429224 | +0.000000 | 243 | 195 |
| streaming | 0.444934 | +0.000000 | 229 | 225 |
| none | 0.550499 | +0.000000 | 304 | 597 |

Rows true; columns predicted:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 103 | 9 | 17 | 18 | 6 | 12 | 20 | 5 |
| gym | 13 | 104 | 13 | 17 | 13 | 12 | 12 | 6 |
| insurance | 15 | 11 | 116 | 14 | 16 | 21 | 15 | 6 |
| mobile | 22 | 12 | 14 | 96 | 11 | 16 | 10 | 10 |
| music | 12 | 27 | 24 | 16 | 80 | 15 | 13 | 11 |
| software | 16 | 19 | 20 | 16 | 9 | 94 | 11 | 10 |
| streaming | 20 | 25 | 23 | 16 | 17 | 15 | 101 | 8 |
| none | 53 | 46 | 48 | 42 | 55 | 58 | 47 | 248 |

Fold F1 drops: +0.000000, +0.000000, +0.000000, +0.000000, +0.000000.

### mild

| class | F1 | drop vs clean | predicted | support |
| --- | --- | --- | --- | --- |
| cloud | 0.431373 | +0.032591 | 218 | 190 |
| gym | 0.451613 | +0.017913 | 244 | 190 |
| insurance | 0.465812 | +0.008626 | 254 | 214 |
| mobile | 0.422062 | +0.028642 | 226 | 191 |
| music | 0.389163 | +0.005899 | 208 | 198 |
| software | 0.432558 | -0.003334 | 235 | 195 |
| streaming | 0.420323 | +0.024611 | 208 | 225 |
| none | 0.603586 | -0.053086 | 407 | 597 |

Rows true; columns predicted:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 88 | 18 | 18 | 17 | 7 | 14 | 19 | 9 |
| gym | 12 | 98 | 14 | 17 | 14 | 15 | 13 | 7 |
| insurance | 12 | 14 | 109 | 12 | 20 | 19 | 14 | 14 |
| mobile | 18 | 13 | 15 | 88 | 13 | 16 | 9 | 19 |
| music | 11 | 20 | 21 | 19 | 79 | 13 | 17 | 18 |
| software | 14 | 15 | 19 | 16 | 9 | 93 | 7 | 22 |
| streaming | 17 | 24 | 23 | 21 | 20 | 14 | 91 | 15 |
| none | 46 | 42 | 35 | 36 | 46 | 51 | 38 | 303 |

Fold F1 drops: -0.005965, +0.014718, +0.010203, +0.008457, +0.013047.

### medium

| class | F1 | drop vs clean | predicted | support |
| --- | --- | --- | --- | --- |
| cloud | 0.355301 | +0.108663 | 159 | 190 |
| gym | 0.431169 | +0.038357 | 195 | 190 |
| insurance | 0.489510 | -0.015073 | 215 | 214 |
| mobile | 0.456311 | -0.005606 | 221 | 191 |
| music | 0.328042 | +0.067019 | 180 | 198 |
| software | 0.400000 | +0.029224 | 200 | 195 |
| streaming | 0.380952 | +0.063982 | 174 | 225 |
| none | 0.617717 | -0.067218 | 656 | 597 |

Rows true; columns predicted:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 62 | 17 | 15 | 20 | 8 | 19 | 12 | 37 |
| gym | 11 | 83 | 15 | 15 | 13 | 17 | 13 | 23 |
| insurance | 11 | 9 | 105 | 12 | 15 | 14 | 11 | 37 |
| mobile | 14 | 10 | 11 | 94 | 13 | 10 | 6 | 33 |
| music | 11 | 12 | 17 | 17 | 62 | 12 | 21 | 46 |
| software | 9 | 14 | 12 | 11 | 9 | 79 | 4 | 57 |
| streaming | 13 | 22 | 20 | 22 | 23 | 13 | 76 | 36 |
| none | 28 | 28 | 20 | 30 | 37 | 36 | 31 | 387 |

Fold F1 drops: -0.004280, +0.028252, +0.033609, +0.035947, +0.040378.

### severe

| class | F1 | drop vs clean | predicted | support |
| --- | --- | --- | --- | --- |
| cloud | 0.263736 | +0.200228 | 83 | 190 |
| gym | 0.360248 | +0.109278 | 132 | 190 |
| insurance | 0.436464 | +0.037974 | 148 | 214 |
| mobile | 0.482587 | -0.031883 | 211 | 191 |
| music | 0.237942 | +0.157120 | 113 | 198 |
| software | 0.357576 | +0.071648 | 135 | 195 |
| streaming | 0.282209 | +0.162725 | 101 | 225 |
| none | 0.566308 | -0.015809 | 1077 | 597 |

Rows true; columns predicted:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 36 | 12 | 8 | 24 | 8 | 14 | 11 | 77 |
| gym | 6 | 58 | 8 | 18 | 11 | 11 | 7 | 71 |
| insurance | 7 | 4 | 79 | 10 | 8 | 8 | 8 | 90 |
| mobile | 3 | 3 | 9 | 97 | 6 | 6 | 5 | 62 |
| music | 7 | 12 | 9 | 16 | 37 | 7 | 11 | 99 |
| software | 1 | 7 | 5 | 6 | 5 | 59 | 1 | 111 |
| streaming | 8 | 19 | 14 | 17 | 18 | 10 | 46 | 93 |
| none | 15 | 17 | 16 | 23 | 20 | 20 | 12 | 474 |

Fold F1 drops: +0.083917, +0.100305, +0.056878, +0.100975, +0.091879.

## Isolated families at medium severity

These controls use the same fitted models and held clients, with one family enabled. They explain sensitivity without altering the predefined main suite. Effects are not additive.

| operation | A F1 | A drop | V2 drop | history drop |
| --- | --- | --- | --- | --- |
| dropout | 0.455070 | +0.004724 | +0.006705 | +0.006890 |
| generic | 0.456938 | +0.002856 | +0.000637 | +0.002560 |
| stream_mask | 0.451832 | +0.007962 | +0.000507 | +0.000000 |
| fragmentation | 0.455427 | +0.004367 | +0.008252 | +0.014091 |
| collision | 0.454130 | +0.005664 | +0.001768 | +0.003820 |
| decoration | 0.462569 | -0.002775 | +0.005011 | +0.000000 |

## Label-free shift and score geometry

Means below are across held folds. Vocabulary reference is each clean fit partition; external diagnostics use all TRAIN as reference after the suite is frozen. Generic fraction is a lexical proxy (only TRAIN broad tokens or synthetic generic/missing/collision masks), not annotated merchant genericity. Amount CV uses repeated same-client/description/currency streams. Currency units are never mixed.

| view | changed_payment_fraction | generic_description_fraction | unseen_description_rate | vocabulary_overlap_jaccard | description_entropy_nats | streams_per_client | fragmented_original_stream_fraction | collided_result_stream_fraction | amount_cv_median_same_currency_repeated_streams |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.0000 | 0.1020 | 0.0063 | 0.4266 | 3.8653 | 19.5010 | 0.0000 | 0.0000 | 0.2279 |
| mild | 0.2694 | 0.1690 | 0.2588 | 0.2424 | 4.8958 | 20.8445 | 0.0843 | 0.0137 | 0.2306 |
| medium | 0.5409 | 0.2422 | 0.5170 | 0.1427 | 5.6988 | 23.0185 | 0.1900 | 0.0382 | 0.2496 |
| severe | 0.8029 | 0.3163 | 0.7767 | 0.0673 | 6.3966 | 26.2760 | 0.3286 | 0.0526 | 0.2728 |

MCC/type/currency distributions are identical across paired views by construction and recorded in JSON. Synthetic alias/merge rates use original event alignment; on external inputs only ordinary stream proxies are available. Model scores are mixture scores, not calibrated confidence.

| view | A mean max score | A entropy nats | A mean none score |
| --- | --- | --- | --- |
| clean | 0.4194 | 1.6476 | 0.1373 |
| mild | 0.4072 | 1.6872 | 0.1661 |
| medium | 0.3838 | 1.7445 | 0.2142 |
| severe | 0.3580 | 1.8023 | 0.2748 |

External input diagnostics were run only after all TRAIN scoring and protocol freeze; no target labels or model predictions on VALID/TEST are read. These values never select severity.

| input | generic proxy | unseen | vocab Jaccard | entropy | streams/client | amount CV |
| --- | --- | --- | --- | --- | --- | --- |
| train | 0.0925 | 0.0000 | 1.0000 | 3.8962 | 19.5010 | 0.2284 |
| valid | 0.1543 | 0.0117 | 0.6029 | 4.3570 | 26.2200 | 0.4487 |
| test | 0.1851 | 0.0099 | 0.5909 | 4.0183 | 23.6760 | 0.5442 |

## Conclusions and limitations

A's severe Macro-F1 drop is 0.086410. Most sensitive isolated operation: `only_stream_mask` (+0.007962). Largest severe class F1 losses: cloud (+0.2002), streaming (+0.1627), music (+0.1571).

A versus V2 contrasts probe additional supervised identity; V2 versus history probes its mapped heuristic. History still contains description-grouped recurrence features, so it is not text independent. The isolated stream mask preserves recurrence grouping and isolates unfamiliar identity better than event dropout; fragmentation/collisions also disturb the recurrence representation even though true timestamps stay fixed.

This suite is a paired stress test, not an estimator of the hidden test score and not proof that synthetic text changes explain all observed cohort shift. TRAIN labels remain latent truth after renaming merchants. Broad tokens can include ambiguous terms and synthetic aliases can be more unfamiliar than real aliases. Map-size transfer, semantic changes, missing events, new merchants, and behavior drift are not simulated. One corruption seed and five folds do not establish seed robustness. External genericity and amount dispersion are imperfect proxies; historic reports informed the task, so no claim of untouched hypothesis development is made.

No family was removed after inspecting scores. Probability-one/all-text erasure is excluded from the principal protocol as a destructive extreme. The explicit collision operator is constrained to equal MCC/type/currency/direction. Shared generic and missing-description tokens can still incidentally merge streams with different contexts when a downstream model groups only by client/description; the context fields themselves remain unchanged. Even severe preserves all nontextual evidence and 19.71% of unaltered payment descriptions. Report both clean and stress scores for future models; do not retune this suite to make a candidate win.

The observed amount-dispersion shift is also larger than this simulation: median within-stream amount CV is 0.2728 under severe, versus 0.4487 in VALID and 0.5442 in TEST. This suite cannot establish whether merchant identity, grouping, monetary behavior or another shift is the dominant cause of the real performance gap. Future candidates should report the same clean and four-level paired scorecards, including isolated families, before interpreting any apparent robustness gain.

An interrupted implementation pilot masked only card rows, leaving 14,989 TRAIN transfer/refund events with matching stream keys visible. It was discarded before completion. The corrected run masks complete anchored streams. Its probabilities and seeds are unchanged; pilot outputs and source remain in an ignored archive. No result in this report comes from that pilot.

## Reproduction and API

```powershell
$env:PYTHONPATH='src;.'
.venv/Scripts/python.exe scripts/experiments/v4_shift_corruption.py --external-inputs `
  --output-dir outputs/metrics/v4_shift_corruption_ginestar_repro
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
```

```python
from transaction_forecasting.ubs.corruption import CorruptionSuite
from transaction_forecasting.ubs.corruption_evaluation import (
    StressDataset,
    evaluate_under_corruption,
)

dataset = StressDataset(train_history, train_labels)
scorecard = evaluate_under_corruption(model_factory, dataset, CorruptionSuite())
```

Factories return fresh fit(history, labels) models with indexed client predictions or official-order probabilities. Multiple named probability components are supported; no baseline-specific imports exist in the reusable evaluator. Raw IDs and local probabilities stay under ignored outputs. summary.json contains every model/view's eight-class metrics, confusion, prediction counts, per-fold scores, confidence, shifts, parameters, hashes and timings.

Total evaluation runtime: 1781.3 seconds (29.7 minutes) for the corrected run, excluding setup, tests and the discarded pilot; environment and source/input hashes are in JSON.

## Verification and artifacts

- Full pytest: **102 passed**; the corruption module adds 14 tests covering seeds, cutoff rejection, clean identity, untouched labels/IDs/metadata, client-fold isolation, label-blind transformations and complete stream scope.
- `ruff check .`, `ruff format --check .` and `pre-commit run --all-files`: **PASS**.
- Existing `scripts/quality_gate.py`: **WARN**, with no failed checks. Tests, lint, formatting and Git hygiene pass. The gate also requests V1 candidate/submission evidence, a comparison test record and peak-memory evidence that this TRAIN-only stress experiment does not produce. Its generic leakage warning describes the existing V1 validation workflow; that workflow was not executed here. `--evaluate` was deliberately not used because it invokes validation evaluation and a train+valid refit.
- Gate details: `outputs/metrics/quality_gate/20260925T000256273635Z/report.json`.
- Machine-readable results: `outputs/metrics/v4_shift_corruption_ginestar/summary.json`; the adjacent `frozen_protocol.json` records parameters and hashes before fitting, and `fold_assignments.csv` records the exact client folds. Per-fold and pooled predictions remain local and ignored.
- The official evaluator and frozen baseline sources are unchanged. No datasets, client-level predictions or unrelated work are staged. Only reusable code, the runner, tests and this aggregate report are committed.

The runner regenerates all numeric tables and the base report; the findings and verification commentary above are the final reviewed handoff.
