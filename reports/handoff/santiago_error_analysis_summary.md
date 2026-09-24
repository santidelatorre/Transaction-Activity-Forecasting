# Error Analysis — Meeting Summary

## Baseline
Macro-F1: **0.2710243**
Accuracy: **0.2660**

## 3 weakest classes

1. `none` — F1 0.094, recall 0.058.
2. `music` — F1 0.239, recall 0.204.
3. `streaming` — F1 0.247, recall 0.237.

## 3 biggest confusions

1. `none→cloud` — 89 errors.
2. `none→mobile` — 43 errors.
3. `none→insurance` — 33 errors.

## 5 key findings

1. `none` is the main failure: only 17/293 correct; V1 suppresses abstention.
2. The five largest bidirectional pairs are a positive family versus `none`.
3. Selected V1 uses 32/218 features and ignores strong family MCC signals.
4. Family text is useful but also appears in `none`; text must be tied to active streams.
5. Selection/calibration on one validation split creates material overfitting risk.

## Who should investigate what

| team | task | priority |
| --- | --- | --- |
| MODELS / TUNING | EA-001: OOF candidate-validity gate for `none` | CRITICAL |
| TEMPORAL / RECURRENCE | EA-002: stream lifecycle, phase and missed cycles | CRITICAL |
| FEATURE ENGINEERING | EA-003: candidate MCC/recency/amount table | HIGH |
| TEXT / MERCHANTS | EA-004: candidate-conditioned normalized text | HIGH |
| EXPERIMENT TRACKING | EA-005: fixed client-level OOF protocol | HIGH |
| INTEGRATION / VALIDATION | EA-006: per-class/pair regression gates | HIGH |

## Top 3 recommended experiments

1. **EA-001:** candidate gate for `none`, using score strength and top-1/top-2 margin.
2. **EA-002:** lifecycle/phase features to distinguish active family streams from ended history.
3. **EA-003:** candidate-level MCC ablation with fold-stability and shortcut checks.

## Main risk

Overfitting thresholds, MCC or vocabulary to the single official validation or synthetic generator.

## Recommendation for V2

Fix candidate validity/`none` first, then lifecycle and candidate evidence. Require OOF gains and
per-class regression gates before integrating; keep V1 unchanged as fallback.
