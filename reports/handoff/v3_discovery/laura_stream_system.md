# Laura — Stream System Discovery

**Status: IMPLEMENTED in source; NOT YET EXECUTED; BLOCKED BY MISSING INPUTS.**
All experimental results below are **NOT YET VERIFIED**. No UBS data or V2
validation predictions were available locally at implementation time.

## 1. Scope

This is an independent architecture baseline on the official UBS validation
split. It asks whether exact-description recurring streams can compete with V2.
It does not alter V1, V2, the official evaluator, or submission schema.

## 2. Architecture

The script loads the official partitions, fits an exact-description family map
on TRAIN, summarizes VALID streams from pre-cutoff history, selects a candidate
per client, and scores A–D through the existing official evaluator. Paired V2
analysis is optional when an aligned V2 validation prediction file exists.

## 3. Stream definition

One stream is `(client_id, exact description)` after the repository loader's
existing lowercase/strip normalization. There is no clustering, fuzzy match,
or identity feature learned from client IDs. IDs only group and align records.
Repeated transactions at the same timestamp count as one event.

## 4. Recurrence logic

The complete system requires at least three distinct event times (two gaps).
Its fixed none gate rejects streams with median absolute gap deviation divided
by the median gap above 0.35. A/B deliberately admit two-event streams so the
effect of this gate is visible. These thresholds are frozen before VALID is run.

## 5. Next-date estimation

The interval is the median of positive consecutive gaps in days. The initial
next date is last event plus that interval. If overdue, whole median intervals
advance it to the first date at or after the 2026-01-01 UTC cutoff. C/D reject
a stream when its initial estimate was overdue by more than one median interval.
D additionally keeps only estimates in `[cutoff, cutoff + 90 days)`. The
cutoff-day lower bound is inclusive; the upper bound is exclusive. No calendar
month model or learned time-series model is used.

## 6. Family mapping

Each TRAIN client contributes once per exact description. The map counts the
client-level next-family label for each description, chooses the most frequent
non-`none` family, and divides its count by *all* client counts, including
`none`. Official label order resolves ties. At least three TRAIN clients must
have the description. C/D also require a winning-family share of at least 0.5.
Unseen descriptions have no map and cannot become candidates. A description
seen only among `none` clients also supplies no positive family evidence.

This association is a weak proxy: a client label does not certify that every
description in that client's history belongs to that family. No TRAIN behavior
is scored. If TRAIN behavior is evaluated later, this map must be fitted
out-of-fold for those predictions; fitting it on all TRAIN and predicting the
same clients would leak their labels.

## 7. None decision

A/B predict `none` only when no mapped repeated stream exists. C/D additionally
require three events, family share ≥0.5, relative gap MAD ≤0.35, and no more
than one missed median interval. D requires an in-horizon next date. If no
candidate survives, predict `none`. No threshold is chosen from VALID.

## 8. Leakage audit

**IMPLEMENTED:** the repository loader rejects transactions at/after cutoff;
the stream code checks again. The family map is fit only from TRAIN histories
and labels. VALID labels enter only the official scoring and paired-error
functions. TEST is loaded by the shared contract checker but is never used to
fit, choose, or score this experiment. No label-derived preprocessing is fit
on VALID. Client IDs group streams and align predictions, not predict identity.

**NOT YET VERIFIED:** actual source-file contents and hashes, behavior on the
official files, and V2 artifact provenance. Comparing predefined ensembles on
VALID still creates selection bias. Probability comparability and calibration
cannot be established from filenames alone.

## 9. Ablations

All variants use the same VALID clients and official fixed-eight-class scorer.

| Variant | Fixed definition | Macro-F1 / accuracy |
| --- | --- | --- |
| A | Repeated mapped stream ranked by family share, then mapping support and event count | NOT YET VERIFIED |
| B | A's candidates ranked by earliest estimated next date | NOT YET VERIFIED |
| C | B plus the fixed recurrence, family, and freshness none gate | NOT YET VERIFIED |
| D | C plus the strict 90-day horizon filter; complete system | NOT YET VERIFIED |

## 10. Stream-system metrics

Macro-F1, accuracy, prediction distribution, predicted `none` share, and
clients without an eligible candidate: **NOT YET VERIFIED**. The script saves
the official metric report for every variant when run.

## 11. Per-class results

Precision, recall, F1, support, and prediction count for all eight classes:
**NOT YET VERIFIED**.

## 12. Confusion matrix

The official true-by-predicted matrix is **NOT YET VERIFIED**.

## 13. Comparison with V2

**SUPPLIED V2 REFERENCE:** Macro-F1 0.391549456. **REPRODUCIBLY VERIFIED V2
RESULT:** NOT YET VERIFIED. The local V2 hard-prediction artifact is missing.
When supplied, the script checks exact VALID client coverage, scores it with
the official evaluator, and flags whether it matches the supplied score within
1e-9. A matching scalar alone does not prove artifact provenance.

## 14. Error complementarity

**NOT YET VERIFIED.** With aligned V2 hard predictions, the script counts
V2-only correct, stream-only correct, both correct, and neither correct,
overall and within each true class. It records both counts and percentages.
These paired results, rather than aggregate score alone, determine whether
stream errors are useful to V3.

## 15. Small ensemble experiment

**NOT YET VERIFIED.** Predefined options are V2 alone, streams alone, one
high-confidence override, and optional 75/25 and 50/50 weighted blends. The
override replaces V2 only for a selected D stream with ≥4 events, family
share ≥0.8, relative gap MAD ≤0.15, and no overdue initial estimate. Weighted
blends run only when both aligned, genuine eight-class probability tables are
provided; hard labels are never converted into invented probabilities. The
stream baseline does not itself produce calibrated probabilities, so weighted
blends are expected to remain unavailable unless a defensible external
probability source is added. Both probability files must agree with their
respective hard predictions. Comparing even these fixed options on VALID has
selection bias; no weight or threshold search is implemented.

## 16. Representative errors

**NOT YET VERIFIED.** When V2 predictions exist, the script saves up to 40
VALID rows where either system errs, in stable official client order. These
are for inspection, not fitting or threshold tuning.

## 17. Limitations

Exact descriptions fragment when text varies. Client-level family labels make
description association noisy. Median gaps cannot model variable calendars;
two-gap regularity can be fragile. Advancing overdue streams assumes a stable
cycle. The hand-set none rule may under- or over-predict `none`. No score or
generalization claim is possible until official data and V2 predictions exist.

## 18. Recommended V3 architecture

**NOT YET VERIFIED.** Keep this stream system as an isolated candidate until
its official VALID Macro-F1, per-class behavior, paired errors, and small
ensemble outcomes are measured. No replace/ensemble decision is justified now.

## 19. Files generated

**IMPLEMENTED:** `scripts/experiments/laura_stream_system.py` and this report.
**NOT YET EXECUTED:** on a successful run, ignored files under
`outputs/metrics/v3_discovery/laura_stream_system/` include
`train_family_mapping.csv`, `ablation_predictions.csv`,
`validation_predictions.csv`, `chosen_candidates.csv`, `ablations.json`, and
`summary.json`. If V2 predictions are supplied, the script also writes
`complementarity.json`, `ensembles.json`, and `representative_errors.csv`.
No generated output has been created or committed.

## 20. Executive summary for V3 integration

The baseline and analysis hooks are implemented; all results are **NOT YET
VERIFIED**. The six official files expected under `data/raw/ubs_2026/` are
absent: `train_transactions.jsonl`, `train_labels.csv`,
`valid_transactions.jsonl`, `valid_labels.csv`, `test_transactions.jsonl`, and
`sample_submission.csv`. V2 paired analysis additionally needs its validation
hard predictions, normally `outputs/metrics/ubs_v2/validation_predictions.csv`.
The documented `outputs/metrics/v2_integration/results.json` is also absent.

Once the official files are restored, run:

```bash
.venv/bin/python scripts/experiments/laura_stream_system.py \
  --v2-predictions outputs/metrics/ubs_v2/validation_predictions.csv
```

The script checks required files before any training or output write. Omit the
V2 argument to evaluate only the standalone baseline if its predictions are
still unavailable. Inspect `summary.json`, `ablations.json`, and the paired
reports before making the V3 architecture decision.
