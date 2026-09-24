# V3 music/streaming specialist — Christian

## 1. Executive summary

**Recommendation: keep V3-A; do not add this specialist to the submission.** On
five client-grouped TRAIN OOF folds, the frozen text + recurrence specialist
raises V3-A Macro-F1 from **0.459794 to 0.467720** (+0.007926). On official VALID,
the same policy makes **zero overrides** and remains at **0.424111**. The
specialist's score distribution shifts substantially: the 95th percentile of
its music/streaming maximum is 0.8445 OOF versus 0.4387 VALID. The apparent OOF
gain does not transfer under the selected conservative gate.

The task was limited to music and streaming corrections. V2 and V3-A are fixed
controls. The official evaluator, V2, V3-A and TEST were unchanged.

## 2. Hypothesis

Music and streaming subscriptions may carry distinctive merchant/description
text and recurring payment patterns that the general eight-class model misses.
The specialist predicts only `music`, `streaming`, or `other`; it can abstain and
leave V3-A unchanged.

## 3. Leakage controls

- Five stratified outer folds of unique TRAIN clients, seed 42. Every V2/V3-A
  OOF probability and specialist score is produced by a model that excludes the
  scored client.
- V3-A's supervised FamilyMap training features use the repository's five
  inner client folds. Each outer held-out client is excluded from the family map
  and CatBoost fit. The specialist uses no supervised merchant map; its text
  vocabulary and coefficients are fitted on outer-fit clients only.
- Descriptions and recurrence summaries use only each client's transactions
  before the official cutoff. IDs are used for grouping and alignment, never as
  predictors.
- The entire policy grid was scored on TRAIN OOF. `frozen_policy.json` was
  written before reading VALID histories/labels. VALID scores were saved before
  reading VALID labels. No VALID threshold changes were made.
- TEST and unlabeled pretrain were not used. The VALID split has been used by
  earlier V3 discovery, so its result is a follow-up diagnostic rather than a
  fresh independent holdout.

## 4. Specialist architecture

The base reproduces the exact V2 and V3-A identity arm: the V2 heuristic remains
25% of the probability blend, with the fixed 300-iteration depth-4 CatBoost
recipe. The specialist is one three-class logistic regression (`music`,
`streaming`, `other`; `C=0.5`, balanced classes, seed 42). It acts after V3-A and
does not replace the general classifier.

The three ablations are:

1. **General identity evidence only:** unchanged V3-A.
2. **Specialist text identity:** char n-gram TF-IDF of normalized outgoing card
   payment descriptions, with repeated names weighted up to four times.
3. **Specialist text + recurrence:** the same text plus eight simple client
   summaries of repeated payment streams, cadence, amount stability and recency.

## 5. Features

Text uses the existing `normalize_description` function, character n-grams of
length 3–5, `min_df=3`, and at most 5,000 terms. Numeric recurrence features
count payment and repeated streams, repeated events, monthly/stable streams,
minimum amount/gap CV, and recency of the most recent repeated stream. They are
derived from each client's own history without family labels. V3-A's music,
streaming and none evidence enters through its frozen predicted class in the
override gate; specialist fitting does not consume an in-sample V3 prediction.

## 6. Override policy

The specialist may override only a V3-A prediction in another positive class.
It preserves `none`, `music` and `streaming`. A promotion requires the target
specialist probability to exceed a threshold and to exceed `other` by a margin.
The predeclared small TRAIN OOF grid used thresholds 0.65, 0.75, 0.85, 0.90 and
margins 0.10, 0.20. Selection maximized eight-class Macro-F1, then preferred
fewer overrides and a larger margin. It selected **text + recurrence**, minimum
probability **0.75**, margin **0.20**. The no-override identity arm was eligible
in the same selection. These scores are not calibrated probabilities.

## 7. TRAIN OOF results

| Arm | Macro-F1 | Accuracy | Music F1 | Streaming F1 | None F1 | Overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 0.408884 | 0.4230 | 0.238095 | 0.352078 | 0.473441 | — |
| V3-A / identity only | 0.459794 | 0.4710 | 0.395062 | 0.444934 | 0.550499 | 0 |
| Specialist text | 0.465842 | 0.4760 | 0.426540 | 0.466102 | 0.550499 | 35 |
| Specialist text + recurrence | **0.467720** | **0.4775** | **0.437647** | 0.462845 | 0.550499 | 37 |

The selected OOF gain over V3-A is +0.007926. These same OOF predictions were
used to choose the policy, so this is selection evidence, not an unbiased final
estimate. Recurrence gives a modest additional Macro-F1 gain over text alone
(+0.001878), while text alone has slightly higher streaming F1.

## 8. VALID results

| Arm | Macro-F1 | Accuracy | Overrides |
| --- | ---: | ---: | ---: |
| V2 | 0.391549 | 0.424 | — |
| V3-A | 0.424111 | 0.461 | 0 |
| V3-A + frozen specialist | **0.424111** | **0.461** | **0** |

The candidate delta versus V3-A is **0.000000**. It retains V3-A's +0.032562
Macro-F1 lead over V2. Only 4 of 1,000 VALID clients reached specialist
music/streaming probability 0.75; none met the complete override policy from a
different positive V3-A class. OOF had 222/2,000 clients above 0.75. The 95th
percentile of the maximum score falls from 0.8445 to 0.4387. This distribution
shift explains the zero-action result without changing the frozen threshold.

## 9. Per-class metrics

| Class | V2 VALID F1 | V3-A VALID F1 | Candidate VALID F1 |
| --- | ---: | ---: | ---: |
| cloud | 0.450000 | 0.481928 | 0.481928 |
| gym | 0.487273 | 0.442478 | 0.442478 |
| insurance | 0.429150 | 0.427273 | 0.427273 |
| mobile | 0.452675 | 0.468085 | 0.468085 |
| music | 0.169935 | 0.323810 | 0.323810 |
| software | 0.373984 | 0.334802 | 0.334802 |
| streaming | 0.250000 | 0.306569 | 0.306569 |
| none | 0.519380 | 0.607945 | 0.607945 |

On VALID, music precision/recall/F1 is 0.290598 / 0.365591 / 0.323810 for
both V3-A and candidate. Streaming is 0.525000 / 0.216495 / 0.306569.
None F1 is 0.607945. The unchanged candidate has the same complete confusion
matrix as V3-A. Rows are truth; columns are cloud, gym, insurance, mobile,
music, software, streaming, none:

| True class | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 40 | 2 | 7 | 8 | 6 | 9 | 3 | 14 |
| gym | 5 | 50 | 13 | 12 | 9 | 11 | 2 | 19 |
| insurance | 5 | 5 | 47 | 5 | 8 | 13 | 1 | 15 |
| mobile | 4 | 5 | 12 | 55 | 6 | 7 | 0 | 15 |
| music | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| software | 7 | 8 | 7 | 11 | 8 | 38 | 2 | 23 |
| streaming | 5 | 10 | 7 | 8 | 24 | 12 | 21 | 10 |
| none | 7 | 17 | 16 | 24 | 22 | 22 | 9 | 176 |

## 10. Music/streaming confusion analysis

The VALID matrix uses rows = true class and columns in the fixed order cloud,
gym, insurance, mobile, music, software, streaming, none:

| True class | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| music (93) | 4 | 8 | 12 | 8 | 34 | 11 | 2 | 14 |
| streaming (97) | 5 | 10 | 7 | 8 | 24 | 12 | 21 | 10 |

Streaming is most often confused with music (24 cases), slightly more than
correct streaming predictions (21). Music has broad confusion with insurance,
software and none. The specialist's OOF correction mainly addresses these
errors, but its high-confidence region nearly disappears on VALID.

## 11. Override analysis

| Split | Total | Into music | Into streaming | From music | From streaming | Correct | Incorrect | New correct | Lost correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN OOF, selected | 37 | 20 | 17 | 0 | 0 | 21 | 16 | 21 | 8 |
| VALID, frozen | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Of the 16 OOF incorrect overrides, eight replace a previously correct V3-A
prediction; the other eight change one wrong class to another wrong class. The
gate deliberately disallows music/streaming removals and none overrides.

## 12. Recommendation

**Do not integrate this specialist.** The OOF gain is small and selected on the
same folds. VALID shows no operational effect because specialist confidence is
much lower. Keep V3-A unchanged for this branch. A later investigation could
study why the specialist score distribution shifts, but a lower VALID-driven
threshold would violate this experiment's selection rule and is not justified
here.

## 13. Reproduction instructions

With official files in ignored `data/raw/ubs_2026`:

```bash
python scripts/experiments/v3_music_streaming_christian.py --phase oof
python scripts/experiments/v3_music_streaming_christian.py --phase valid
python -m pytest -q
ruff check .
```

The script writes ignored artifacts to
`outputs/metrics/v3_music_streaming_christian/`: fold scores, OOF scores,
`frozen_policy.json`, VALID scores and metric JSON. TRAIN input hashes are checked
when resuming. The official evaluator and test partition are untouched.
