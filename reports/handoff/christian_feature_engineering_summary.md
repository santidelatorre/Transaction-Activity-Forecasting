# Feature Engineering — Meeting Summary

## Baseline

Macro-F1: 0.2710243 (team-provided; not verified in this workspace)
Accuracy: 0.2660 (team-provided; not verified in this workspace)

## Best result

Macro-F1: NOT AVAILABLE — no experiments could run
Delta vs V1: NOT AVAILABLE

## Top 5 useful features

No feature has measured usefulness yet. These are unranked experiment candidates:

1. Description top-1/top-3 share
2. Description HHI / normalized entropy
3. MCC top-1/top-3 share
4. MCC HHI / normalized entropy
5. Active-day density and transactions per active day

## Top 3 feature groups

1. Concentration (first candidate to test)
2. Recent versus prior activity
3. Active-day density

These are experiment priorities, not proven winners.

## 5 key findings

1. V1 already uses transaction counts, in/out counts, type/MCC/currency shares, simple unique counts, amount moments, recent-window counts, monthly variability, gaps, and recurrence streams.
2. The checked-in profile reports about 74 transactions/client and roughly 405 history days; these figures were not recalculated because raw files are absent.
3. The profile reports repeated descriptions for all train clients (median repeat appearances 3); this motivates testing concentration but does not prove class separation.
4. About 99.5% of clients reportedly transact within 30 days of cutoff; an activity contrast may add more than absolute recency, but it remains untested.
5. Four currencies are present; new amount statistics should be calculated per currency, not pooled.

## What did NOT work

1. No new feature experiment ran; there is no evidence to label a candidate worse or neutral.
2. Baseline reproduction was blocked by missing raw data.
3. Automated tests were blocked because the configured Python 3.11 interpreter is missing.

## Main risk

Current V1 code has target-derived description lift applied to its own training rows and tunes the recurrence heuristic/ensemble using validation labels. Resolve or explicitly account for these risks before asserting leakage-safe comparative scores.

## Recommendation for V2

Do not integrate new features based on this blocked run. Restore train/validation data and Python, verify V1, then run FE-001 concentration with per-class distribution analysis and individual-component ablations. Coordinate recent/recurrence features with the temporal and text workstreams.
