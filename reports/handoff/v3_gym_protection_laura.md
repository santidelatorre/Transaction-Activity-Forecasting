# V3 GYM protection — Laura

## 1. Executive summary

**TRAIN result: do not promote this routing gate.** The pooled OOF-selected
threshold is 0.40; its apparent Macro-F1 gain is only +0.000408496 over V3-A.
Gate-level cross-validation reverses that gain to **-0.000898517**, with two
changed clients and zero correct overrides. The requested cross-validated-value
criterion is not met. VALID is a single frozen diagnostic, not a second chance
to select a rule. Frozen VALID improves **0.424111098 -> 0.427115231** (+0.003004133),
but that is only three recovered gym clients and does not overturn the failed
TRAIN criterion. Results distinguish the fixed candidate from fold-selected
gate predictions.

Branch: `exp/v3-gym-protection-laura`. Base/execution HEAD:
`df9fe4135ff812add0e2f6e0733706050b813cc2` (also `origin/integration/v3-discovery` at preflight).
The experiment began with a clean tracked working tree. New contributions are
`src/transaction_forecasting/gym_protection.py`,
`scripts/run_v3_gym_protection.py`, `tests/test_gym_protection.py`, and this report.
V2, V3, their features and the official evaluator are unchanged.

## 2. Hypothesis

A conservative routing rule could recover true gym cases where V2 retains
stronger evidence than identity-only V3-A. It must preserve A elsewhere and
show TRAIN cross-validated value. V2/V3 probabilities are blended decision
scores, **not demonstrated calibrated confidences**; comparing their magnitudes
is itself an experimental assumption.

Only existing probabilities are used. No merchant rule, recurrence feature,
class-specific weight, new classifier, or TEST input is introduced. Preliminary
TRAIN-only threshold/dominance probes preceded the final narrow grid, so this
is not a pristine preregistered research study. No new VALID errors were read
to construct or revise this rule.

## 3. OOF disagreement analysis

The five cached client-held-out folds cover exactly 2,000 unique TRAIN clients;
190 have target gym. Each saved fold was checked against the original
`StratifiedKFold(5, shuffle=True, random_state=42)` membership. Probability
argmax matches the saved V2/A predictions. All recorded OOF source/TRAIN hashes
and generation package/Python versions match this execution.

| TRAIN group | Clients | True gym | Mean V2 gym score | Mean A gym score | V2 gym score > A gym score |
| --- | --- | --- | --- | --- | --- |
| V2_only_gym | 74 | 13 | 0.319607682 | 0.234805917 | 65 |
| A_only_gym | 33 | 7 | 0.239580573 | 0.313592179 | 2 |
| both_gym | 220 | 97 | 0.425872960 | 0.453563957 | 68 |
| true_gym_both_miss | 73 | 73 | 0.144682203 | 0.143270066 | 34 |

Only 13/74 V2-only gym predictions are true gym (17.57%). Their other true labels:
none 28, music 9, streaming 8, software 6, cloud 6, mobile 2, insurance 2.
Of 220 shared gym predictions, 68 have a higher V2 gym score; most instead
have stronger A scores. The 73 gym clients missed by both are outside this
policy's recoverable set. OOF gym F1 already favors A over V2; a VALID-context
regression does not establish a TRAIN-recoverable routing signal.

## 4. Leakage safeguards

- Selection reads TRAIN labels only; inference receives probabilities, no labels.
- Original base-model OOF uses client-isolated outer fits and cross-fitted family
  mappings inside those fits. Source and fold memberships were inspected.
- Gate-level held-fold targets are excluded from that fold's threshold selection
  (also unit tested). **This is not fully nested base-model CV**: labels of a
  gate-held fold may have entered base fits generating other folds' OOF features.
  These results are a limited cross-validation diagnostic, not an unbiased nested
  estimate. A positive promotion claim would require fully nested base refits.
- V3 public entry points and shared loader reject timestamps on/after
  2026-01-01 and missing timestamps. TRAIN/VALID client overlap is rejected.
- VALID prediction uses TRAIN-only fit of the unchanged V3 API. It computes
  other original arms internally, but this experiment uses only V2 and A.
- A freeze is written before VALID history/prediction. VALID labels are read
  semantically only after candidate predictions are saved; their file hash may
  be calculated earlier. A persistent exclusive marker prevents silent reruns.
- Historical discovery reports already exposed VALID results and motivated
  this hypothesis. VALID therefore is **not an independent untouched holdout**.
- No TEST labels, TEST histories, model-selection feedback from VALID, or final
  TRAIN+VALID refit are used here. No universal absence-of-leakage claim is made.

## 5. Gate/rule

Start with A's argmax. Override to gym only if all conditions hold:

1. V2 argmax is gym.
2. A argmax is not gym.
3. `P_V2(gym) >= 0.40`.
4. `P_V2(gym) >= max_class P_A(class)`.

No overrides away from gym. Probability columns are aligned to the official
class order before argmax (including deterministic ties); IDs must match
exactly, be unique/non-null, and probabilities finite/nonnegative and sum to one.

Search: explicit no-op plus thresholds 0.250 to 0.750 inclusive in increments
of 0.025. Dominance margin is fixed at zero. Maximize official pooled TRAIN
OOF Macro-F1; exact ties prefer fewer overrides, then no-op, then higher
threshold. These are experimental engineering choices, not official thresholds.

Freeze: `outputs/metrics/v3_gym_protection/frozen_policy.json`, created
`2026-09-24T21:50:23.483654+00:00`. SHA-256:
`4bb93a958b48a89f8a87a42bc864e036cbf00a62348ef208a46ad762dfd08f99`.
`train_cv_supported=false` is recorded before VALID. The frozen candidate is
retained for a diagnostic evaluation only and cannot be promoted by this run.

## 6. TRAIN cross-validation

| Estimator | Macro-F1 | Accuracy | Gym F1 | None F1 |
| --- | --- | --- | --- | --- |
| V2 | 0.408883545 | 0.423000000 | 0.454545455 | 0.473441109 |
| A | 0.459793827 | 0.471000000 | 0.469525959 | 0.550499445 |
| candidate | 0.460202323 | 0.471000000 | 0.475555556 | 0.547274750 |
| gate CV A | 0.459793827 | 0.471000000 | 0.469525959 | 0.550499445 |
| gate CV candidate | 0.458895310 | 0.470000000 | 0.467415730 | 0.548888889 |

The first candidate row is scored on the **same pooled OOF labels used to choose
0.40**, and is optimistically selected. Gate CV selects a threshold on four
original OOF folds and predicts the fifth, then pools those held-fold predictions.
It is a different prediction vector, not a claim that the frozen 0.40 rule has
been independently nested-validated.

| Fold | Selected threshold | V2 Macro-F1 | A Macro-F1 | Gate Macro-F1 | V2 gym F1 | A gym F1 | Gate gym F1 | Overrides/correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4 | 0.386089231 | 0.422836749 | 0.421256760 | 0.435643564 | 0.442105263 | 0.437500000 | 1/0 |
| 2 | 0.4 | 0.414130599 | 0.449722327 | 0.449722327 | 0.448979592 | 0.451612903 | 0.451612903 | 0/0 |
| 3 | no-op | 0.423242384 | 0.476689542 | 0.476689542 | 0.417582418 | 0.373333333 | 0.373333333 | 0/0 |
| 4 | no-op | 0.426194003 | 0.474124751 | 0.474124751 | 0.444444444 | 0.512195122 | 0.512195122 | 0/0 |
| 5 | 0.4 | 0.387149451 | 0.465166029 | 0.462247314 | 0.519230769 | 0.551020408 | 0.545454545 | 1/0 |

Two folds select no-op, two folds lose Macro-F1, and three remain equal. No fold
improves. The positive pooled selection score is insufficient evidence.

## 7. VALID

| Model | Macro-F1 | Accuracy | Delta vs A |
| --- | --- | --- | --- |
| V2 | 0.391549456 | 0.424000000 | -0.032561642 |
| A | 0.424111098 | 0.461000000 | +0.000000000 |
| candidate | 0.427115231 | 0.464000000 | +0.003004133 |

All 1,000 official VALID clients were scored once, after freezing. The V2 and
A controls reproduce the existing report references to their reported precision;
these are actual local results, not copied reference metrics. Model fitting and
prediction/evaluation took 136.494 seconds (not a controlled benchmark).
The official fixed-eight-class evaluator is reused unchanged.

Candidate vs A: **+0.003004133** absolute Macro-F1
(+0.7083% relative), +0.003 accuracy.
The result is positive descriptively; it does not rescue the negative TRAIN
cross-validation result or provide independent generalization evidence.
No threshold was changed after this evaluation.

| VALID input/artifact | SHA-256 |
| --- | --- |
| VALID transactions | `2c30941d49ca20f7b082526407f24529c5a704205610373eff952f4f444a3622` |
| VALID labels | `979bd9b69253084fc9e2b1ef473c07a628a8b18091169051ba304cfed0fb5642` |
| V2 VALID probabilities | `5abb9c6f67dd754407bf0bd166383944e4a498c70469f2a4a17e325917646bee` |
| A VALID probabilities | `b6844f573cf541a084c6c6191d8d8a4b1dd7c2e283cb1d17aacd06267b9e7125` |


## 8. Gym-specific metrics

| Partition/estimator | Model | Gym precision | Gym recall | Gym F1 | None F1 |
| --- | --- | --- | --- | --- | --- |
| TRAIN selected OOF | V2 | 0.374149660 | 0.578947368 | 0.454545455 | 0.473441109 |
| TRAIN selected OOF | A | 0.411067194 | 0.547368421 | 0.469525959 | 0.550499445 |
| TRAIN selected OOF | candidate | 0.411538462 | 0.563157895 | 0.475555556 | 0.547274750 |
| TRAIN gate CV | V2 | 0.374149660 | 0.578947368 | 0.454545455 | 0.473441109 |
| TRAIN gate CV | A | 0.411067194 | 0.547368421 | 0.469525959 | 0.550499445 |
| TRAIN gate CV | candidate | 0.407843137 | 0.547368421 | 0.467415730 | 0.548888889 |
| VALID frozen | V2 | 0.435064935 | 0.553719008 | 0.487272727 | 0.519379845 |
| VALID frozen | A | 0.476190476 | 0.413223140 | 0.442477876 | 0.607944732 |
| VALID frozen | candidate | 0.481818182 | 0.438016529 | 0.458874459 | 0.611111111 |

VALID gym true positives rise from 50 to 53 of 121. V2 recovers 67, so the
candidate still does not recover V2 gym F1 (0.487272727). Music gains slightly,
streaming is unchanged, and none F1 rises through fewer false none predictions.


## 9. Full per-class metrics

| Class | OOF V2 | OOF A | OOF selected gate | VALID V2 | VALID A | VALID gate |
| --- | --- | --- | --- | --- | --- | --- |
| cloud | 0.435185185 | 0.463963964 | 0.460496614 | 0.450000000 | 0.481927711 | 0.484848485 |
| gym | 0.454545455 | 0.469525959 | 0.475555556 | 0.487272727 | 0.442477876 | 0.458874459 |
| insurance | 0.478664193 | 0.474437628 | 0.474437628 | 0.429149798 | 0.427272727 | 0.427272727 |
| mobile | 0.426778243 | 0.450704225 | 0.450704225 | 0.452674897 | 0.468085106 | 0.468085106 |
| music | 0.238095238 | 0.395061728 | 0.398009950 | 0.169934641 | 0.323809524 | 0.325358852 |
| software | 0.412280702 | 0.429223744 | 0.429223744 | 0.373983740 | 0.334801762 | 0.334801762 |
| streaming | 0.352078240 | 0.444933921 | 0.445916115 | 0.250000000 | 0.306569343 | 0.306569343 |
| none | 0.473441109 | 0.550499445 | 0.547274750 | 0.519379845 | 0.607944732 | 0.611111111 |

VALID prediction distribution:

| Class | VALID V2 count | VALID A count | VALID gate count |
| --- | --- | --- | --- |
| cloud | 71 | 77 | 76 |
| gym | 154 | 105 | 110 |
| insurance | 148 | 121 | 121 |
| mobile | 139 | 131 | 131 |
| music | 60 | 117 | 116 |
| software | 142 | 123 | 123 |
| streaming | 63 | 40 | 40 |
| none | 223 | 286 | 283 |

Frozen candidate VALID confusion matrix (rows true, columns predicted):

| True / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cloud | 40 | 2 | 7 | 8 | 6 | 9 | 3 | 14 |
| gym | 5 | 53 | 13 | 12 | 8 | 11 | 2 | 17 |
| insurance | 5 | 5 | 47 | 5 | 8 | 13 | 1 | 15 |
| mobile | 4 | 5 | 12 | 55 | 6 | 7 | 0 | 15 |
| music | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| software | 7 | 9 | 7 | 11 | 8 | 38 | 2 | 22 |
| streaming | 4 | 11 | 7 | 8 | 24 | 12 | 21 | 10 |
| none | 7 | 17 | 16 | 24 | 22 | 22 | 9 | 176 |

Complete precision/recall/F1, confusion matrices and prediction distributions
for all three models are persisted in `train_results.json` / `valid_results.json`.


## 10. Override analysis

The selected TRAIN rule changes 7/2,000 clients: 3 correct overrides, 4 incorrect.
Three previously correct predictions become wrong (none 2, cloud 1), and one
none client remains incorrect but changes to gym. Net correct predictions: zero.
TRAIN gym F1 rises while cloud/none F1 fall; precision/recall tradeoffs matter.
Gate CV changes only 2 clients; both were correct and become false gym (none 1,
cloud 1). These small counts explain why a score improvement in selection is
not robust.

VALID changes **5/1,000 clients**, all towards gym: **3 correct, 2 incorrect**,
zero away from gym, zero previously correct predictions harmed. The two incorrect
overrides are true software and true streaming, both already misclassified by A
(none and cloud respectively). Of the three recovered gym clients, A predicted
none twice and music once. No class loses a true positive; no per-class F1 falls.
This is a five-client diagnostic, not robust evidence for broad routing.


## 11. Stability

The outcome depends on a handful of clients. No bootstrap confidence interval
or statistical-significance claim is made. Gate CV is not fully nested and the
threshold search adds selection optimism. The existing official VALID partition
has been reused across the wider team's discovery work. Neither a tiny positive
VALID contrast nor a negative one establishes behavior on the hidden leaderboard.

Provenance: seed 42; Python 3.11.16; packages
numpy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1, catboost 1.2.10.
Full source, OOF probability, saved prediction and TRAIN hashes are in the freeze.
Hashes detect changed bytes; they do not authenticate who generated an artifact.

| Input | SHA-256 |
| --- | --- |
| TRAIN transactions | `ba0902b33b1181921ee5b1f8f445693201c0968ca52d70a92e12cbb114e7d70a` |
| TRAIN labels | `cab0ec46064348f970ffafdd4f42a12d41a8db660e9db835e7235851b556ff9f` |
| V2 OOF probability manifest | `fcd2b465fe365007f8b6e34c63d34d6e1174b7c77055bb71628fc0ee0a2aeea5` |
| V3-A OOF probability manifest | `c395e3f2771b336ede9198fac4ac8e347f803d8c90ff0c23c83db5263168ba3d` |

Manifest hash = SHA-256 of the sorted compact JSON mapping of the five saved
probability paths to their file SHA-256 hashes; it is path-sensitive. The freeze
also records the exact script/module hashes since these were new uncommitted
files at execution time. A commit SHA alone would not capture their contents.
VALID data/probability hashes and runtime are recorded in `valid_results.json`.

Validation actually executed: `.venv/bin/python -m pytest -q` **103 passed**;
new targeted tests **16 passed**; `.venv/bin/python -m ruff check .` **PASS**;
`.venv/bin/python -m ruff format --check .` **PASS**, 85 files; Ruff 0.16.8,
pytest 8.4.2. Tests cover probability contracts, held-fold label exclusion,
no-op selection, overrides, fingerprint tampering, exclusive writes, and the
prediction-before-VALID-label sequence. Mock tests are not evidence of real
end-to-end model performance; actual local results are reported separately.

## 12. Recommendation

**Do not integrate the gym routing policy into production or change V2/V3-A.**
Keep this additive runner, tests and negative research report available for
review. The policy has not demonstrated the required TRAIN cross-validated
value. Avoid expanding the grid after seeing VALID. If this hypothesis is
revisited, predefine stronger TRAIN evidence, use fully nested base/gate folds,
and seek a genuinely new holdout; the present subset is too small to support
high-capacity routing. This study does not itself authorize promotion of V3-A.

## 13. Reproduction commands

From the repository root with the project installed in `.venv` (no new packages):

```bash
# Only if OOF files are missing; existing complete OOF was reused for this run.
.venv/bin/python scripts/run_ubs_v3.py --phase oof

# Fresh output directory required; do not overwrite the recorded experiment.
.venv/bin/python scripts/run_v3_gym_protection.py --phase train \
  --output-dir outputs/metrics/v3_gym_protection_reproduction

# Inspect TRAIN result/freeze. A negative result is not promotion-eligible.
# Optional one-time frozen diagnostic; never tune from this result.
.venv/bin/python scripts/run_v3_gym_protection.py --phase valid \
  --output-dir outputs/metrics/v3_gym_protection_reproduction

.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

For a different OOF directory pass `--oof-dir PATH`. The original OOF runner
stores literal paths in provenance; relocating a cache may require regenerating
it in a fresh directory, not editing its fingerprints. An environment mismatch
fails early. Do not weaken these checks to force a comparison.

Outputs under ignored `outputs/metrics/v3_gym_protection/`:
`threshold_search.csv`, `train_results.json`, `oof_predictions.csv`,
`frozen_policy.json`, `valid_started.json`, `valid_V2_probabilities.csv`,
`valid_A_probabilities.csv`, `valid_predictions.csv`, `valid_results.json`.
JSON metrics include confusion matrices (rows true, columns predicted) and
prediction counts in official order: cloud, gym, insurance, mobile, music,
software, streaming, none. No submission was generated or sent. Local datasets,
OOF, outputs and `.venv` remain ignored and excluded from the research commit.
