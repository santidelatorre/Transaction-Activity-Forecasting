# V3 family identity mapping handoff — Javier

## 1. Executive summary

The frozen TRAIN OOF winner adds smoothed, client-presence family probabilities to V3-A's 21 hard identity features. It improves TRAIN OOF Macro-F1 from **0.459794 to 0.480617** (+0.020823), but fails on the single subsequent VALID evaluation: **0.424111 to 0.324374** (−0.099737). VALID `none` predictions double from 286 to 576. **Do not promote this candidate. Retain current V3-A** as the identity-only research baseline and V2 as the fallback.

## 2. Hypothesis

A description may carry family information even when the current support/lift rule calls it `unknown`. Smoothed `P(family | description)` aggregated across a client's outgoing card payments might capture that graded evidence, particularly for music, streaming, cloud, and software. Exact normalized aliases and high-confidence character aliases were tested as smaller alternatives. This is client-target association, not verified stream ground truth.

## 3. Leakage safeguards

- All supervised counts use unique TRAIN-client presence, never transaction count as independent labels.
- Five stratified outer folds (seed 42) evaluate TRAIN OOF. Within each outer fit, five client folds fit every supervised map only on the other inner clients. A held-out client's label cannot affect any of its hard aliases or probability aggregates. The map also rejects transforming a fit client.
- VALID uses a map fitted on all TRAIN clients, after `probability` was frozen by TRAIN OOF. Predictions were saved before the runner read VALID labels. No TEST labels were used; no TEST phase was run.
- `test_identity_features_cannot_use_own_label` changes a client's label with fold membership fixed and asserts identical features for normalized, probability, and character-alias configurations.

## 4. Current V3-A family mapping

The unchanged V3-A control maps exact descriptions by smoothed positive-family client-presence lift, requiring at least five fit clients and lift ≥1.5. Its 21 features are count, share, and alias count for each of seven positive families. The numeric model has 167 features total: 146 V2 history features plus 21 identity features. Its 75% CatBoost / 25% V2 heuristic blend and CatBoost settings are unchanged here. Freshly reproduced V2 and V3-A scores match the discovery synthesis to the shown precision.

## 5. Proposed changes

`IdentityMap` uses the existing V2 description normalizer, then fits the same hard map plus a Dirichlet-smoothed seven-family posterior. The posterior adds 21 features: per-family mean probability over outgoing payments, maximum probability, and mean over descriptions recurring within a client. No supervised `none` posterior is added. The resulting classifier has **188 features**. The optional character fallback maps an otherwise unknown normalized name to a known eligible alias only at TF-IDF character similarity ≥0.88; it adds no columns. There is no generic clustering, temporal extension, or unlabeled pretraining.

## 6. TRAIN OOF ablations

| Arm | Description | Features | Macro-F1 | Accuracy | Δ vs V3-A |
| --- | --- | ---: | ---: | ---: | ---: |
| V2 | Unchanged control | 146 | 0.408884 | 0.423 | −0.050910 |
| V3-A | Exact hard identity | 167 | 0.459794 | 0.471 | — |
| Normalized | Normalized hard identity | 167 | 0.459794 | 0.471 | 0.000000 |
| Probability | Normalized hard identity + posterior aggregates | 188 | **0.480617** | **0.496** | **+0.020823** |
| Char + probability | Probability arm + high-similarity alias fallback | 188 | 0.478708 | 0.493 | +0.018914 |

Normalization is a no-op on this input: TRAIN has 1,475 distinct raw descriptions and 1,475 normalized descriptions. Character matching reduced aggregate OOF by 0.001909 versus probability alone. The probability arm was frozen as the maximum TRAIN OOF arm before VALID scoring. This small ablation set does not support a claim that every posterior or aggregation choice was explored.

## 7. TRAIN OOF

Five outer client folds, 2,000 clients. V2 / V3-A / frozen candidate Macro-F1: **0.408884 / 0.459794 / 0.480617**. Accuracy: **0.423 / 0.471 / 0.496**. Candidate gains against V3-A in all eight classes on OOF. OOF prediction counts by class order below are V3-A `[254, 253, 275, 235, 207, 243, 229, 304]` and candidate `[269, 224, 279, 257, 204, 227, 226, 314]`.

## 8. VALID

The frozen candidate scores **0.324374 Macro-F1, 0.387 accuracy**, versus current V3-A **0.424111, 0.461** and V2 **0.391549, 0.424**. Candidate deltas are **−0.099737 Macro-F1 and −0.074 accuracy** versus V3-A. VALID prediction counts by class order are V3-A `[77, 105, 121, 131, 117, 123, 40, 286]` and candidate `[59, 59, 72, 49, 31, 100, 54, 576]`. This is a material failure, especially music and the positive-versus-none gate.

## 9. Per-class F1

Class order: cloud, gym, insurance, mobile, music, software, streaming, none.

| Class | OOF V2 | OOF V3-A | OOF candidate | OOF Δ | VALID V2 | VALID V3-A | VALID candidate | VALID Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | .435185 | .463964 | .466231 | +.002267 | .450000 | .481928 | .405405 | −.076522 |
| gym | .454545 | .469526 | .478261 | +.008735 | .487273 | .442478 | .300000 | −.142478 |
| insurance | .478664 | .474438 | .494929 | +.020491 | .429150 | .427273 | .339181 | −.088091 |
| mobile | .426778 | .450704 | .459821 | +.009117 | .452675 | .468085 | .339869 | −.128216 |
| music | .238095 | .395062 | .437811 | +.042749 | .169935 | .323810 | .112903 | −.210906 |
| software | .412281 | .429224 | .440758 | +.011535 | .373984 | .334802 | .313725 | −.021076 |
| streaming | .352078 | .444934 | .461197 | +.016263 | .250000 | .306569 | .291391 | −.015179 |
| none | .473441 | .550499 | .605928 | +.055428 | .519380 | .607945 | .492520 | −.115425 |

### Confusion matrices

Within each cell, the eight counts are predictions in the class order above for the true class named by that table column. The machine-readable matrices are in the ignored `outputs/metrics/v3_family_mapping_javier/{oof,valid}_results.json` files.

| Split / arm | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OOF V2 | 94 15 18 26 3 14 14 6 | 12 110 12 18 5 20 8 5 | 14 11 129 15 13 18 8 6 | 16 12 17 102 11 17 4 12 | 14 31 29 24 40 16 30 14 | 19 18 22 15 6 94 9 12 | 16 30 28 26 24 20 72 9 | 57 67 70 61 36 62 39 205 |
| OOF V3-A | 103 9 17 18 6 12 20 5 | 13 104 13 17 13 12 12 6 | 15 11 116 14 16 21 15 6 | 22 12 14 96 11 16 10 10 | 12 27 24 16 80 15 13 11 | 16 19 20 16 9 94 11 10 | 20 25 23 16 17 15 101 8 | 53 46 48 42 55 58 47 248 |
| OOF candidate | 107 5 16 19 10 12 18 3 | 18 99 12 18 13 14 13 3 | 15 11 122 16 14 20 12 4 | 22 11 14 103 11 12 10 8 | 16 19 26 18 88 10 15 6 | 18 12 20 20 11 93 11 10 | 24 27 22 18 14 12 104 4 | 49 40 47 45 43 54 43 276 |
| VALID V2 | 36 5 11 6 4 10 5 12 | 7 67 9 9 5 7 5 12 | 4 6 53 4 2 7 7 16 | 0 7 15 55 3 8 0 16 | 3 14 16 12 13 12 11 12 | 5 12 10 12 2 46 4 13 | 2 17 10 10 17 13 20 8 | 14 26 24 31 14 39 11 134 |
| VALID V3-A | 40 2 7 8 6 9 3 14 | 5 50 13 12 9 11 2 19 | 5 5 47 5 8 13 1 15 | 4 5 12 55 6 7 0 15 | 4 8 12 8 34 11 2 14 | 7 8 7 11 8 38 2 23 | 5 10 7 8 24 12 21 10 | 7 17 16 24 22 22 9 176 |
| VALID candidate | 30 2 3 2 2 3 6 41 | 3 27 8 5 5 7 1 65 | 4 5 29 3 5 8 3 42 | 3 3 9 26 1 6 2 54 | 2 5 7 2 7 9 4 57 | 4 3 2 1 2 32 3 57 | 5 4 4 3 2 11 22 46 | 8 10 10 7 7 24 13 214 |

## 10. Coverage and mapping quality

Full TRAIN map: 1,475 distinct descriptions, of which 429 pass the hard-map support/lift rule: cloud 89, gym 68, insurance 60, mobile 69, music 46, software 57, streaming 40. The other 1,046 remain `unknown`. On VALID outgoing card payments, **14,599 / 46,474 rows (31.4%)** map hard to a positive family; **99.7% of clients** have at least one mapped row. Thus client coverage is broad while row coverage is narrow. Mapped-row fit-client support quantiles (min, Q1, median, Q3, max) are **5, 22, 443, 474, 1,083**. Maximum seven-family posterior on each VALID outgoing row has quantiles **0, .112, .117, .173, .383**. These probabilities describe client-target co-occurrence, not true merchant-family purity or calibrated prediction confidence.

## 11. Prediction changes versus V3-A

| Split | Changed | Corrected | Regressed | Wrong to different wrong |
| --- | ---: | ---: | ---: | ---: |
| TRAIN OOF (2,000) | 333 | 149 | 99 | 85 |
| VALID (1,000) | 427 | 89 | 163 | 175 |

On VALID, 328 previously positive V3-A predictions become `none`; 38 previously `none` predictions become positive. The largest positive-to-none changes are mobile 82, music 65, gym 54, and insurance 45. This overwhelms the modest family gains seen in OOF.

## 12. Failure analysis

The TRAIN OOF gain is not stable under the frozen VALID evaluation. Cross-fitted TRAIN probability-mean features average **0.1037** across family columns; VALID features from the full TRAIN map average **0.0951**. The median client maximum among family probability-mean columns falls from **0.1348** to **0.1118**. Candidate `none` predictions rise from 314/2,000 in OOF to 576/1,000 on VALID, although V3-A does not show a comparable shift. We infer that the added posterior aggregates interact badly with the positive-versus-none decision under this fit/VALID feature distribution. The difference between inner-fit and full-fit map sizes also changes posterior values, so this diagnostic does not isolate whether sample size, cohort shift, or both cause the failure. Do not tune a new threshold from these VALID errors.

The unchanged normalization result is expected because the challenge descriptions are already normalized to the same 1,475 keys. Character fallback gives no aggregate OOF gain over the probability arm. High support for common descriptions does not imply clean family identity: clients transact with several families.

## 13. Recommendation

**Reject the probability candidate for promotion.** Keep the current 167-feature V3-A arm for any further identity work and preserve the V2 fallback. A future experiment would need a new independent cohort or a TRAIN-only simulation of map-size/feature distribution robustness before further VALID use. The OOF score alone is insufficient given this 0.099737 VALID loss.

## 14. Reproduction commands

From this branch, with the repository's working Python environment and local ignored UBS data:

```powershell
python scripts/experiments/v3_family_mapping_javier.py --phase oof
python scripts/experiments/v3_family_mapping_javier.py --phase valid
python -m pytest -q
ruff check .
```

The runner writes fold predictions, metrics, map audit, and frozen selection under ignored `outputs/metrics/v3_family_mapping_javier/`. Run `valid` only after `oof` writes `frozen_selection.json`. The Python executable on this workstation is `.venv/runtime_v2/python.exe`; the ordinary `.venv/Scripts/python.exe` launcher points to a removed interpreter.
