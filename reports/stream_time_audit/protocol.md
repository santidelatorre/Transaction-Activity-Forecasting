# Stream identity and relative-time investigation

Source: **only** `research/import-v2-stream-identity`, pinned commit
`e4aa4c58175242e198cefd32d6ac4558145523af`. Research branch:
`research/ginestar-stream-time-audit`. No other branch's code, models, caches,
metrics or conclusions are inputs to this study.

## Questions fixed before new model evaluation

1. Which relative-time and family-linked measurements actually enter the final predictor?
2. Does the compact component depend on temporal columns, refund evidence and
   correct association of temporal evidence with family candidates?
3. Does a small explicit cycle-state representation improve its paired TRAIN
   cross-validation results under the source branch's unchanged corruption views?
4. What can be said about individual predictions and the jury story with measured evidence?

## Design

- Raw files must match the source branch's committed SHA-256 manifest.
- Only TRAIN targets are loaded for this investigation. The disjoint unlabeled
  histories supply the source implementation's price profiles. VALID and TEST
  labels/predictions do not select parameters or variants. Historical committed
  validation results are reviewed as existing evidence, not new measurements.
- Five stratified client folds, shuffle seed 42. All eight candidates and all
  augmented views of a client remain in that client's fold.
- Features are rebuilt from raw transactions with the source `build_features`,
  `min_count=3`; original, valid_like and test_like corruptions, corruption seed
  2026. The names describe synthetic TRAIN stress views, not official datasets.
- Each experimental fit is the source compact family ranker + separate none
  classifier, seed 42, unchanged 500-tree/15-leaf/.035 settings. This isolates
  the representation used by the dominant ensemble component. It is **not**
  the full three-seed/nine-estimator production ensemble and must not be
  compared as if its score reproduced the historical 0.619493 official result.
- Four variants: `control`, `no_temporal`, `no_refund`, `cycle_state`. These
  differ only in supplied columns. `no_temporal` removes downstream temporal
  columns, including derived comparisons; it does NOT remove time-dependent
  upstream stream selection or grouping.
- `cycle_state` adds explicit observed-cycle availability, cycle fit, elapsed
  cycles, delay, overdue/missed cycles and active-within-horizon evidence.
  Weekly cycles are represented; stale candidates are not automatically moved
  into the future by modulo arithmetic. Existing overdue_ratio already
  contains part of this information; added value is an empirical question.
- Pooled fixed-eight-class Macro-F1 is primary. Report all variants and views,
  per-class F1/confusion, fold scores, prediction frequencies, and paired
  client-bootstrap delta intervals. No new official holdout access or automatic
  baseline promotion is part of this study.

## Attribution and counterfactual diagnostics

- On control held-out clients, permute complete candidate blocks of each feature
  group across clients, retaining within-client family ordering; three fixed
  repeats. Record Macro-F1 loss and decision changes. Dependence between groups
  means these are sensitivity diagnostics, not additive causal effects.
- Rotate the seven positive families' temporal blocks within each client while
  retaining family identity and the none row. This preserves the client's
  temporal multiset but breaks its family association. The none detector's
  symmetric aggregates should stay unchanged within floating-point tolerance.
  This is a deliberately incoherent stress intervention, not a plausible future.
- Record native LightGBM contribution values for a fixed, target-independent
  sample of held-out clients, for both ranker and none detector. These explain
  component raw scores, not causal behavior or the whole production ensemble.
- Inspect real TRAIN incidence of missing/sentinel states, supported cycles,
  and ambiguous nearest-timestamp currency association; separate observed
  incidence from synthetic counterexamples.

## Interpretation limits

Feature extraction includes fixed vocabulary/semantic rules and amount grouping
already researched in the source branch. Reusing TRAIN for further experiments
does not create a pristine holdout. Price priors inherit cross-currency pooling
and weak lexical supervision from the source. Corruption views are synthetic.
Bootstrap intervals condition on predictions and do not correct for researcher
selection. Changing units alone adds no information. These experiments can
identify useful or fragile representations; they cannot establish a reachable
0.80 leaderboard score, another team's method, or real banking impact.

## Reproduction and scope

The research runner writes detailed models/predictions only below ignored
`outputs/stream_time_audit/`. Small aggregate reports, protocol, source hashes,
test coverage and executable research code are committed. Original production
source, frozen submissions and historical ledgers stay unchanged.
