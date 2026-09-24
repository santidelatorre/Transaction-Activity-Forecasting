# V3 dedicated NONE gate - Santiago

Branch: `exp/v3-none-gate-santiago`

Base: `origin/integration/v3-discovery` at `df9fe4135ff812add0e2f6e0733706050b813cc2`

## 1. Executive summary

The dedicated binary gate is **rejected**. A compact logistic gate appeared strong under
second-level TRAIN cross-validation, raising Macro-F1 from **0.459793827** for V3-A to
**0.485019029** (+0.025225202). The gain was present in all five gate folds. With the
architecture, features, regularization and threshold frozen, however, VALID Macro-F1 fell from
**0.424111098** to **0.357072946** (-0.067038152). The paired VALID bootstrap interval for the
delta was entirely negative: [-0.090212, -0.045161].

The failure is an over-aggressive transfer into NONE. V3-A already predicted 286 NONE clients
on VALID, close to the 293 true NONE clients. The gate increased this to 502. It changed 228
V3-A predictions: 57 became correct, 91 became incorrect, and 80 changed between two wrong
answers. Every class F1 declined, including NONE. No VALID-driven retry or threshold change was
made.

Recommendation: **REJECT**. Retain V3-A as the experimental identity candidate and V2 as the
reliable frozen baseline.

## 2. Exact hypothesis

V3-A's cross-fitted family identity representation is responsible for most of the V3 gain, but
its multiclass argmax may make a suboptimal decision between any positive subscription family
and NONE. A separately learned, compact binary gate might improve the final eight-class
Macro-F1 while leaving V3-A's positive-family ranking untouched.

The gate therefore performs only this operation:

1. V3-A selects the highest-probability family among the seven positive families.
2. The binary gate selects positive subscription or NONE.
3. If positive is selected, the final family is exactly V3-A's positive-family choice.

## 3. Leakage controls

- All adaptive choices used TRAIN only. VALID labels were first read after the frozen VALID
  predictions were written to disk.
- The V3-A base layer used five stratified outer folds of unique TRAIN clients. Each client's
  base probabilities and gate features came from a model and family map that excluded that
  client.
- Within every V3-A outer fit, supervised family identity features were themselves built with
  five inner client folds through `cross_fitted_family_features`.
- Every row used to fit a second-level gate contained OOF V2/V3-A predictions, not in-sample
  base predictions.
- The reported TRAIN gate estimate used a separate five-fold client cross-fit (seed 43). For
  each gate fold, the held-out clients were absent from the gate fit.
- Each gate-fold threshold was selected with four inner folds inside that gate's fit partition.
  The reported TRAIN OOF score uses those fold-specific thresholds, not the final threshold.
- After candidate selection, the final threshold 0.53 was chosen using the selected candidate's
  cross-fitted TRAIN scores. It was frozen before VALID.
- Client IDs were alignment and fold keys only. No ID is a model feature.
- No TEST labels exist or were used. No TEST data were used in this experiment.
- The official evaluator was not modified.

This is standard stacked OOF evaluation: a client's own target never enters features derived for
that same client, and the second-level estimate is evaluated on clients unseen by the gate.

## 4. Experimental design

The experiment compared:

- A. frozen V2: 75% history CatBoost plus 25% periodicity heuristic;
- B. V3-A identity-only: the exact 167-feature identity arm from V3 discovery;
- C. V3-A positive-family prediction plus a dedicated binary NONE gate.

The base OOF split was five-fold stratified client CV with seed 42. Gate candidates used another
five-fold stratified client CV with seed 43. Logistic candidates used four inner folds to choose
the gate threshold. Candidate selection maximized the fixed-eight-class TRAIN OOF Macro-F1,
not binary F1 or accuracy.

The predefined candidates were:

| candidate | model | features | C | TRAIN OOF Macro-F1 | accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| score_a | normalized V3-A positive-vs-NONE score | probability | - | 0.476210497 | 0.5155 |
| logit_prob_c01 | logistic regression | probability | 0.1 | 0.479703525 | 0.5180 |
| logit_prob_c1 | logistic regression | probability | 1.0 | 0.478940976 | 0.5175 |
| **logit_compact_c01** | **logistic regression** | **compact** | **0.1** | **0.485019029** | **0.5205** |
| logit_compact_c1 | logistic regression | compact | 1.0 | 0.479438191 | 0.5130 |
| tree_compact | depth-2 decision tree, minimum leaf 30 | compact | - | 0.473412038 | 0.5180 |

The selected fold thresholds were 0.56, 0.56, 0.52, 0.53 and 0.54. The final frozen threshold,
selected from cross-fitted TRAIN gate scores, was 0.53. The one permitted VALID evaluation then
used the selected configuration without alteration.

## 5. Features used

The selected compact gate used 27 numeric features:

- V3-A: NONE probability, maximum positive probability, their margin, overall maximum,
  top-1/top-2 margin, and normalized entropy;
- V2: NONE probability, maximum positive probability, their margin, top-1/top-2 margin, and
  normalized entropy;
- V3-A minus V2 NONE probability;
- V2/V3-A agreement and positive-family agreement;
- transaction count and outbound card-payment count;
- number of recurrent exact descriptions and recurrent-event share;
- mapped-family event share, number of mapped candidate families, maximum mapped-family share,
  and mapped share for V3-A's selected positive family;
- history span and recency;
- number of recurrent payment streams, maximum regularity, and strongest recurrence score.

All behavioral features are calculated from pre-cutoff transactions. Mapped-family evidence uses
the fold-isolated family map. The gate receives no target, raw client ID, or post-cutoff feature.

## 6. Model/gate used

The frozen gate is a scikit-learn pipeline:

1. median imputation;
2. standard scaling;
3. logistic regression with `C=0.1`, `solver="lbfgs"`, `max_iter=1000`, and seed 42;
4. positive if probability is at least 0.53, otherwise NONE.

No class weights were used. V3-A and V2 retain their frozen CatBoost and heuristic recipes. The
dedicated `IdentityV3Model` only avoids fitting unused B/AB arms; its TRAIN OOF V2 and V3-A
scores exactly reproduce the discovery results.

## 7. TRAIN OOF results

### Headline metrics

| model | Macro-F1 | accuracy | positive-client accuracy | NONE precision | NONE recall | NONE F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 0.408883545 | 0.4230 | 0.456878 | 0.762082 | 0.343384 | 0.473441 |
| V3-A | 0.459793827 | 0.4710 | 0.494654 | 0.815789 | 0.415410 | 0.550499 |
| V3-A + gate | **0.485019029** | **0.5205** | 0.476123 | 0.778706 | 0.624791 | **0.693309** |

The candidate gained +0.025225202 Macro-F1 over V3-A. Its paired client bootstrap interval was
[+0.015930, +0.034351]. Despite the overall gain, positive-client accuracy fell from 694/1403
to 668/1403; the improvement was mainly the NONE correction.

Prediction counts in label order cloud, gym, insurance, mobile, music, software, streaming,
NONE were:

- V2: 242, 294, 325, 287, 138, 261, 184, 269.
- V3-A: 254, 253, 275, 235, 207, 243, 229, 304.
- gate: 234, 233, 246, 204, 177, 217, 210, 479.
- true TRAIN counts: 190, 190, 214, 191, 198, 195, 225, 597.

### TRAIN OOF confusion matrices

Rows are true labels and columns are predictions, both in the order cloud, gym, insurance,
mobile, music, software, streaming, NONE.

V2:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 94 | 15 | 18 | 26 | 3 | 14 | 14 | 6 |
| gym | 12 | 110 | 12 | 18 | 5 | 20 | 8 | 5 |
| insurance | 14 | 11 | 129 | 15 | 13 | 18 | 8 | 6 |
| mobile | 16 | 12 | 17 | 102 | 11 | 17 | 4 | 12 |
| music | 14 | 31 | 29 | 24 | 40 | 16 | 30 | 14 |
| software | 19 | 18 | 22 | 15 | 6 | 94 | 9 | 12 |
| streaming | 16 | 30 | 28 | 26 | 24 | 20 | 72 | 9 |
| none | 57 | 67 | 70 | 61 | 36 | 62 | 39 | 205 |

V3-A:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 103 | 9 | 17 | 18 | 6 | 12 | 20 | 5 |
| gym | 13 | 104 | 13 | 17 | 13 | 12 | 12 | 6 |
| insurance | 15 | 11 | 116 | 14 | 16 | 21 | 15 | 6 |
| mobile | 22 | 12 | 14 | 96 | 11 | 16 | 10 | 10 |
| music | 12 | 27 | 24 | 16 | 80 | 15 | 13 | 11 |
| software | 16 | 19 | 20 | 16 | 9 | 94 | 11 | 10 |
| streaming | 20 | 25 | 23 | 16 | 17 | 15 | 101 | 8 |
| none | 53 | 46 | 48 | 42 | 55 | 58 | 47 | 248 |

V3-A + gate:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 97 | 9 | 17 | 16 | 6 | 12 | 19 | 14 |
| gym | 13 | 105 | 13 | 18 | 13 | 12 | 12 | 4 |
| insurance | 16 | 9 | 112 | 13 | 15 | 20 | 13 | 16 |
| mobile | 21 | 12 | 14 | 87 | 10 | 15 | 10 | 22 |
| music | 13 | 27 | 22 | 16 | 74 | 14 | 13 | 19 |
| software | 14 | 19 | 19 | 16 | 7 | 91 | 9 | 20 |
| streaming | 20 | 25 | 20 | 16 | 17 | 14 | 102 | 11 |
| none | 40 | 27 | 29 | 22 | 35 | 39 | 32 | 373 |

## 8. VALID results

### Headline metrics

| model | Macro-F1 | accuracy | positive-client accuracy | NONE precision | NONE recall | NONE F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 0.391549456 | 0.4240 | 0.410184 | 0.600897 | 0.457338 | 0.519380 |
| V3-A | **0.424111098** | **0.4610** | **0.403112** | **0.615385** | 0.600683 | **0.607945** |
| V3-A + gate | 0.357072946 | 0.4270 | 0.282885 | 0.452191 | **0.774744** | 0.571069 |

The candidate delta versus V3-A was **-0.067038152** Macro-F1 and -0.034 accuracy. Its
paired client bootstrap interval was [-0.090212, -0.045161]. Higher NONE recall did not
compensate for the severe precision and positive-family losses.

Prediction counts in label order cloud, gym, insurance, mobile, music, software, streaming,
NONE were:

- V2: 71, 154, 148, 139, 60, 142, 63, 223.
- V3-A: 77, 105, 121, 131, 117, 123, 40, 286.
- gate: 55, 76, 86, 71, 83, 99, 28, 502.
- true VALID counts: 89, 121, 99, 104, 93, 104, 97, 293.

### VALID confusion matrices

V2:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

V3-A:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 40 | 2 | 7 | 8 | 6 | 9 | 3 | 14 |
| gym | 5 | 50 | 13 | 12 | 9 | 11 | 2 | 19 |
| insurance | 5 | 5 | 47 | 5 | 8 | 13 | 1 | 15 |
| mobile | 4 | 5 | 12 | 55 | 6 | 7 | 0 | 15 |
| music | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| software | 7 | 8 | 7 | 11 | 8 | 38 | 2 | 23 |
| streaming | 5 | 10 | 7 | 8 | 24 | 12 | 21 | 10 |
| none | 7 | 17 | 16 | 24 | 22 | 22 | 9 | 176 |

V3-A + gate:

| true | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 24 | 2 | 4 | 3 | 4 | 8 | 3 | 41 |
| gym | 4 | 36 | 10 | 9 | 8 | 8 | 1 | 45 |
| insurance | 3 | 5 | 36 | 4 | 3 | 12 | 1 | 35 |
| mobile | 4 | 4 | 6 | 35 | 5 | 6 | 0 | 44 |
| music | 3 | 5 | 10 | 3 | 25 | 10 | 2 | 35 |
| software | 4 | 6 | 7 | 8 | 4 | 31 | 2 | 42 |
| streaming | 5 | 8 | 4 | 4 | 20 | 10 | 13 | 33 |
| none | 8 | 10 | 9 | 5 | 14 | 14 | 6 | 227 |

## 9. Per-class comparison

### TRAIN OOF F1

| class | V2 | V3-A | gate | gate - V3-A |
| --- | ---: | ---: | ---: | ---: |
| cloud | 0.435185 | 0.463964 | 0.457547 | -0.006417 |
| gym | 0.454545 | 0.469526 | 0.496454 | +0.026928 |
| insurance | 0.478664 | 0.474438 | 0.486957 | +0.012519 |
| mobile | 0.426778 | 0.450704 | 0.440506 | -0.010198 |
| music | 0.238095 | 0.395062 | 0.394667 | -0.000395 |
| software | 0.412281 | 0.429224 | 0.441748 | +0.012524 |
| streaming | 0.352078 | 0.444934 | 0.468966 | +0.024032 |
| none | 0.473441 | 0.550499 | 0.693309 | +0.142809 |

### VALID F1

| class | V2 | V3-A | gate | gate - V3-A |
| --- | ---: | ---: | ---: | ---: |
| cloud | 0.450000 | 0.481928 | 0.333333 | **-0.148594** |
| gym | 0.487273 | 0.442478 | 0.365482 | **-0.076996** |
| insurance | 0.429150 | 0.427273 | 0.389189 | **-0.038084** |
| mobile | 0.452675 | 0.468085 | 0.400000 | **-0.068085** |
| music | 0.169935 | 0.323810 | 0.284091 | **-0.039719** |
| software | 0.373984 | 0.334802 | 0.305419 | **-0.029383** |
| streaming | 0.250000 | 0.306569 | 0.208000 | **-0.098569** |
| none | 0.519380 | 0.607945 | 0.571069 | **-0.036876** |

The VALID regression is broad, not a rare-family trade that preserves most other classes. All
eight classes declined.

## 10. Prediction-change analysis

| measure versus V3-A | TRAIN OOF | VALID |
| --- | ---: | ---: |
| predictions changed | 207 | 228 |
| changes into NONE | 191 | 222 |
| changes out of NONE | 16 | 6 |
| correct changes | 136 | 57 |
| incorrect changes | 37 | 91 |
| wrong-to-different-wrong | 34 | 80 |
| V3-A positive clients correct | 694/1403 | 285/707 |
| gate positive clients correct | 668/1403 | 200/707 |

TRAIN OOF V3-A predicted NONE for only 304/2000 clients while 597 were truly NONE, so a learned
correction toward NONE was useful there. On VALID, V3-A already predicted NONE for 286/1000,
nearly matching the 293 true NONE clients. The same learned correction then produced 502 NONE
predictions. This base-probability/decision-rate shift is the central transfer failure. It also
explains why optimizing binary NONE behavior or accuracy in TRAIN was insufficient for the
eight-class objective.

## 11. Stability across folds

| gate fold | V2 Macro-F1 | V3-A Macro-F1 | gate Macro-F1 | gate - V3-A | selected threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.437352 | 0.470671 | 0.483701 | +0.013030 | 0.56 |
| 2 | 0.396426 | 0.447690 | 0.491928 | +0.044238 | 0.56 |
| 3 | 0.402811 | 0.468791 | 0.483970 | +0.015179 | 0.52 |
| 4 | 0.373363 | 0.441304 | 0.467980 | +0.026676 | 0.53 |
| 5 | 0.435310 | 0.468654 | 0.496805 | +0.028152 | 0.54 |

The fold delta mean was +0.025455 and sample standard deviation 0.012465; the gate beat V3-A in
5/5 folds. Thresholds were also relatively stable. This makes the negative VALID result more
informative: ordinary re-splitting of TRAIN did not reveal the distribution shift, so the OOF
gain should not be promoted despite appearing internally robust.

Verification completed with 91/91 tests passing, `ruff check .` passing, and all 67 Python files
passing `ruff format --check src scripts tests`. Repository-wide format traversal encountered
pre-existing unreadable Windows temp directories. A workspace-local pre-commit attempt could not
initialize its pinned hooks because outbound GitHub access was unavailable; the equivalent local
Ruff and format checks passed. No scorer code changed.

## 12. Recommendation

**REJECT.**

Do not promote this gate, do not replace V3-A's multiclass NONE decision with it, and do not
retune its threshold on VALID. It damages the primary metric by 0.067038 and lowers every class
F1. Preserve the reusable code as negative experimental evidence and retain:

- V2 as the reliable frozen baseline;
- V3-A identity-only as the stronger experimental model;
- the lesson that a gate trained on OOF base probabilities can fail when the full-fit base
  model's NONE decision rate shifts at inference.

Any future NONE-gate work needs a design that explicitly addresses OOF-to-full-fit score shift
using TRAIN-only simulation or a genuinely new labelled cohort. This experiment does not justify
such a follow-up on the already inspected VALID set.

## Reproduction commands

From the repository root, with project dependencies installed:

```powershell
git branch --show-current
git status
git rev-parse HEAD
git merge-base HEAD origin/integration/v3-discovery

python scripts/run_v3_none_gate.py --phase oof --output-dir outputs/metrics/ubs_v3_none_gate_santiago_repro
python scripts/run_v3_none_gate.py --phase valid --output-dir outputs/metrics/ubs_v3_none_gate_santiago_repro

python -m pytest -q --basetemp=test-temp-none-gate/full
ruff check .
ruff format --check .
```

On the experiment machine, replace `python` with `.venv/Scripts/python.exe`. The OOF run took
327.8 seconds and the frozen VALID run 131.5 seconds. Generated features, probabilities,
predictions, provenance JSON and metrics remain under ignored `outputs/`; no dataset, cache,
prediction or trained model is versioned.
