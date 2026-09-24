# V3 V2/V3-A disagreement gate — Ginestar

## 1. Executive summary

The frozen learned gate is **small_gradient_boosting**. On TRAIN it changes V3-A Macro-F1 by
**-0.002195** under a second five-fold client-level cross-fitting layer. On
previously unopened VALID it changes Macro-F1 by **-0.006817**. It routes
315 disagreements with accuracy
**0.4000**, changing
37 V3-A predictions (12 correct,
13 incorrect, 12 between
two wrong classes). Recommendation: **REJECT**.

This is a hard per-client selector. It is not a global probability blend and it
does not interpolate V2 and V3-A scores.

## 2. Hypothesis

V2 and V3-A encode different evidence. V3-A adds cross-fitted merchant-family
identity features, while V2 retains the older history representation. If their
errors differ systematically, inference-time confidence geometry may identify
which prediction to trust for an individual client.

## 3. Anti-leakage design

- Five outer stratified TRAIN folds (seed 42) produce V2 and V3-A probabilities.
  Neither base model sees its held client's label.
- A second five-fold stratified split (seed 314159) evaluates every learned gate.
  Each routed TRAIN prediction comes from a gate fitted without that client.
- Only TRAIN OOF results select the gate family. All candidate hyperparameters and
  the 0.5 decision threshold are fixed in source.
- The final gate trains on decisive TRAIN OOF disagreements only. VALID predictions
  are written to disk before `valid_labels.csv` is read.
- Client ID is an index only. No IDs, VALID labels, TEST labels, or post-cutoff
  transactions enter a model feature.

The TRAIN figure after selecting among three cross-fitted gates remains a model
selection estimate, not a pristine external estimate. VALID is the untouched
confirmation set for this workstream.

## 4. Construction of OOF predictions

Each base fold fits the unchanged `IntegratedV2Model` and the frozen V3-A
identity-only arm. V3-A uses the same 75% CatBoost / 25% periodicity composition
as the discovery study. Its supervised family map is also internally cross-fitted
for the base-fold training matrix. OOF CSVs contain one probability vector per
TRAIN client and are ignored by Git.

## 5. Gate architecture

Agreements preserve the shared class. On a disagreement, the fixed rule selects
the higher maximum probability. Learned gates predict `choose V3-A` versus
`choose V2`. Their training target exists only when exactly one system is correct;
both-wrong cases are excluded from fitting and retained as routing failures during
evaluation. The frozen shortlist is logistic regression, a depth-3 tree, and a
small histogram gradient booster.

| Learned candidate | TRAIN cross-fitted Macro-F1 | Accuracy |
|---|---:|---:|
| logistic | 0.442745 | 0.462500 |
| shallow_tree | 0.435968 | 0.453000 |
| small_gradient_boosting (selected) | 0.457599 | 0.472500 |

## 6. Feature set

The gate uses 27 inference-time numeric features: both eight-class probability
vectors, each model's maximum probability, top-1/top-2 margin, normalized entropy,
the V3-A-minus-V2 differences for those summaries, and both `none` probabilities.
No transaction feature was added because the V3-A probability vector already
summarizes family evidence and the initial test should keep gate capacity low.

## 7. TRAIN OOF results

| Method | Macro-F1 | Accuracy |
|---|---:|---:|
| V2 | 0.408884 | 0.423000 |
| V3-A | 0.459794 | 0.471000 |
| Fixed confidence routing | 0.444519 | 0.455000 |
| Learned disagreement gate | 0.457599 | 0.472500 |

### TRAIN per-class F1

| Actual class | V2 F1 | V3-A F1 | Gate F1 | Gate - V3-A |
|---|---:|---:|---:|---:|
| cloud | 0.435185 | 0.463964 | 0.466819 | +0.002855 |
| gym | 0.454545 | 0.469526 | 0.481236 | +0.011710 |
| insurance | 0.478664 | 0.474438 | 0.491018 | +0.016580 |
| mobile | 0.426778 | 0.450704 | 0.428571 | -0.022133 |
| music | 0.238095 | 0.395062 | 0.380952 | -0.014109 |
| software | 0.412281 | 0.429224 | 0.417234 | -0.011990 |
| streaming | 0.352078 | 0.444934 | 0.430913 | -0.014021 |
| none | 0.473441 | 0.550499 | 0.564047 | +0.013548 |

### TRAIN confusion matrices

**V2**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 94 | 15 | 18 | 26 | 3 | 14 | 14 | 6 |
| gym | 12 | 110 | 12 | 18 | 5 | 20 | 8 | 5 |
| insurance | 14 | 11 | 129 | 15 | 13 | 18 | 8 | 6 |
| mobile | 16 | 12 | 17 | 102 | 11 | 17 | 4 | 12 |
| music | 14 | 31 | 29 | 24 | 40 | 16 | 30 | 14 |
| software | 19 | 18 | 22 | 15 | 6 | 94 | 9 | 12 |
| streaming | 16 | 30 | 28 | 26 | 24 | 20 | 72 | 9 |
| none | 57 | 67 | 70 | 61 | 36 | 62 | 39 | 205 |

**V3-A**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 103 | 9 | 17 | 18 | 6 | 12 | 20 | 5 |
| gym | 13 | 104 | 13 | 17 | 13 | 12 | 12 | 6 |
| insurance | 15 | 11 | 116 | 14 | 16 | 21 | 15 | 6 |
| mobile | 22 | 12 | 14 | 96 | 11 | 16 | 10 | 10 |
| music | 12 | 27 | 24 | 16 | 80 | 15 | 13 | 11 |
| software | 16 | 19 | 20 | 16 | 9 | 94 | 11 | 10 |
| streaming | 20 | 25 | 23 | 16 | 17 | 15 | 101 | 8 |
| none | 53 | 46 | 48 | 42 | 55 | 58 | 47 | 248 |

**Fixed confidence routing**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 98 | 11 | 18 | 21 | 6 | 14 | 17 | 5 |
| gym | 12 | 105 | 14 | 21 | 10 | 12 | 10 | 6 |
| insurance | 15 | 12 | 121 | 14 | 15 | 21 | 11 | 5 |
| mobile | 21 | 13 | 14 | 95 | 12 | 17 | 9 | 10 |
| music | 11 | 29 | 26 | 18 | 71 | 16 | 15 | 12 |
| software | 20 | 20 | 21 | 17 | 7 | 91 | 11 | 8 |
| streaming | 20 | 27 | 24 | 18 | 17 | 15 | 98 | 6 |
| none | 55 | 62 | 51 | 46 | 54 | 55 | 43 | 231 |

**Learned disagreement gate**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 102 | 11 | 17 | 22 | 6 | 12 | 14 | 6 |
| gym | 14 | 109 | 12 | 18 | 9 | 14 | 8 | 6 |
| insurance | 13 | 11 | 123 | 14 | 14 | 20 | 12 | 7 |
| mobile | 19 | 13 | 17 | 93 | 9 | 18 | 9 | 13 |
| music | 13 | 29 | 26 | 17 | 72 | 14 | 13 | 14 |
| software | 17 | 19 | 20 | 17 | 8 | 92 | 9 | 13 |
| streaming | 17 | 28 | 26 | 19 | 16 | 16 | 92 | 11 |
| none | 52 | 43 | 46 | 43 | 46 | 60 | 45 | 262 |

## 8. VALID results

| Method | Macro-F1 | Accuracy |
|---|---:|---:|
| V2 | 0.391549 | 0.424000 |
| V3-A | 0.424111 | 0.461000 |
| Fixed confidence routing | 0.441966 | 0.471000 |
| Learned disagreement gate | 0.417294 | 0.460000 |

### VALID per-class F1

| Actual class | V2 F1 | V3-A F1 | Gate F1 | Gate - V3-A |
|---|---:|---:|---:|---:|
| cloud | 0.450000 | 0.481928 | 0.459627 | -0.022300 |
| gym | 0.487273 | 0.442478 | 0.433628 | -0.008850 |
| insurance | 0.429150 | 0.427273 | 0.429224 | +0.001951 |
| mobile | 0.452675 | 0.468085 | 0.454148 | -0.013937 |
| music | 0.169935 | 0.323810 | 0.315271 | -0.008539 |
| software | 0.373984 | 0.334802 | 0.331839 | -0.002963 |
| streaming | 0.250000 | 0.306569 | 0.300752 | -0.005817 |
| none | 0.519380 | 0.607945 | 0.613861 | +0.005917 |

### VALID confusion matrices

**V2**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

**V3-A**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 40 | 2 | 7 | 8 | 6 | 9 | 3 | 14 |
| gym | 5 | 50 | 13 | 12 | 9 | 11 | 2 | 19 |
| insurance | 5 | 5 | 47 | 5 | 8 | 13 | 1 | 15 |
| mobile | 4 | 5 | 12 | 55 | 6 | 7 | 0 | 15 |
| music | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| software | 7 | 8 | 7 | 11 | 8 | 38 | 2 | 23 |
| streaming | 5 | 10 | 7 | 8 | 24 | 12 | 21 | 10 |
| none | 7 | 17 | 16 | 24 | 22 | 22 | 9 | 176 |

**Fixed confidence routing**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 42 | 3 | 9 | 6 | 6 | 10 | 2 | 11 |
| gym | 5 | 59 | 10 | 10 | 9 | 7 | 3 | 18 |
| insurance | 5 | 4 | 53 | 5 | 7 | 11 | 2 | 12 |
| mobile | 4 | 6 | 14 | 56 | 5 | 6 | 0 | 13 |
| music | 3 | 10 | 15 | 8 | 30 | 11 | 2 | 14 |
| software | 5 | 9 | 8 | 12 | 4 | 47 | 1 | 18 |
| streaming | 4 | 14 | 9 | 8 | 19 | 11 | 23 | 9 |
| none | 12 | 22 | 15 | 28 | 22 | 26 | 7 | 161 |

**Learned disagreement gate**

| true \ pred | cloud | gym | insurance | mobile | music | software | streaming | none |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cloud | 37 | 2 | 7 | 8 | 6 | 9 | 1 | 19 |
| gym | 5 | 49 | 12 | 12 | 9 | 12 | 2 | 20 |
| insurance | 5 | 5 | 47 | 4 | 8 | 11 | 1 | 18 |
| mobile | 3 | 5 | 12 | 52 | 7 | 7 | 0 | 18 |
| music | 4 | 8 | 12 | 10 | 32 | 10 | 2 | 15 |
| software | 7 | 8 | 7 | 11 | 6 | 37 | 2 | 26 |
| streaming | 4 | 11 | 8 | 8 | 22 | 13 | 20 | 11 |
| none | 7 | 17 | 15 | 20 | 20 | 20 | 8 | 186 |

## 9. Disagreement decomposition

### TRAIN OOF learned gate

| Actual class | Disagree | V2 only | V3-A only | Both wrong | Route acc. | To V2 | To V3-A | Changed | Correct | Incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 531 | 123 | 219 | 189 | 0.4181 | 141 | 390 | 141 | 48 | 45 |
| cloud | 40 | 8 | 17 | 15 | 0.4000 | 12 | 28 | 12 | 4 | 5 |
| gym | 34 | 13 | 7 | 14 | 0.3529 | 12 | 22 | 12 | 7 | 2 |
| insurance | 50 | 23 | 10 | 17 | 0.3400 | 15 | 35 | 15 | 9 | 2 |
| mobile | 34 | 15 | 9 | 10 | 0.1765 | 14 | 20 | 14 | 4 | 7 |
| music | 69 | 4 | 44 | 21 | 0.5217 | 15 | 54 | 15 | 0 | 8 |
| software | 49 | 13 | 13 | 23 | 0.2245 | 15 | 34 | 15 | 3 | 5 |
| streaming | 73 | 15 | 44 | 14 | 0.4795 | 22 | 51 | 22 | 4 | 13 |
| none | 182 | 32 | 75 | 75 | 0.4890 | 36 | 146 | 36 | 17 | 3 |

### VALID learned gate

| Actual class | Disagree | V2 only | V3-A only | Both wrong | Route acc. | To V2 | To V3-A | Changed | Correct | Incorrect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 315 | 90 | 127 | 98 | 0.4000 | 37 | 278 | 37 | 12 | 13 |
| cloud | 27 | 6 | 10 | 11 | 0.2593 | 5 | 22 | 5 | 0 | 3 |
| gym | 36 | 22 | 5 | 9 | 0.1111 | 2 | 34 | 2 | 0 | 1 |
| insurance | 29 | 11 | 5 | 13 | 0.1724 | 4 | 25 | 4 | 1 | 1 |
| mobile | 28 | 11 | 11 | 6 | 0.2857 | 6 | 22 | 6 | 1 | 4 |
| music | 33 | 1 | 22 | 10 | 0.6061 | 3 | 30 | 3 | 0 | 2 |
| software | 33 | 14 | 6 | 13 | 0.1515 | 3 | 30 | 3 | 0 | 1 |
| streaming | 40 | 11 | 12 | 17 | 0.2750 | 4 | 36 | 4 | 0 | 1 |
| none | 89 | 14 | 56 | 19 | 0.7416 | 10 | 79 | 10 | 10 | 0 |

## 10. Per-class changes

The per-class tables above compare F1, which includes both false positives from
other actual classes and false negatives inside each row. The routing tables give
the complementary client counts by actual class. A class should be treated as a
real gain only when its F1 improves without being driven by a tiny number of
changes.

## 11. Fold stability

| Gate fold | Clients | V2 | V3-A | Fixed confidence | Learned gate | Gate - V3-A |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 400 | 0.402351 | 0.465864 | 0.464552 | 0.461393 | -0.004470 |
| 2 | 400 | 0.387324 | 0.419816 | 0.416370 | 0.421597 | +0.001781 |
| 3 | 400 | 0.387605 | 0.453577 | 0.435545 | 0.434154 | -0.019423 |
| 4 | 400 | 0.449765 | 0.474821 | 0.454100 | 0.492578 | +0.017757 |
| 5 | 400 | 0.413508 | 0.482956 | 0.447647 | 0.475377 | -0.007579 |

The learned gate beats V3-A in **2/5** gate folds. These are the
additional gate folds, not the outer folds that generated the base probabilities.

## 12. Comparison against V2 and V3-A

The fixed-confidence router is the no-training control. The learned router is
only useful if it improves on V3-A out of sample and the gain is not isolated to
one fold or one tiny class. VALID is reported once after freezing; its result did
not alter the selected gate, features, threshold, or targeted classes.

## 13. Recommendation

**REJECT.** The decision rule was fixed in code: promote only when the
gate improves both TRAIN cross-fitted and VALID Macro-F1 and wins at least three
of five gate folds; mark promising when exactly one aggregate split improves;
otherwise reject. A promotion still requires normal PR review and must not replace
the frozen V2/V3-A artifacts silently.

## 14. Reproduction commands

```powershell
python scripts/run_v3_disagreement_gate.py --phase oof
python scripts/run_v3_disagreement_gate.py --phase valid
python -m pytest -q
python -m ruff check .
```

Generated evidence is under `outputs/metrics/v3_disagreement_gate_ginestar` and intentionally ignored. The
tracked report contains no client IDs, probabilities, dataset rows, or predictions.
