# V3 protocol frozen before new VALID predictions

Baseline source: `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`.
The seven discovery reports, including their VALID results, were read first.
This protocol is therefore informed by previously reused VALID, not independent.

## Operational hypothesis

V2 deliberately removes supervised family features to avoid own-label leakage.
Its numeric classifier consequently loses semantic family identity, especially
music versus streaming sharing MCC 5812. Exact-description recurrence also mixes
refunds with payments and separates semantically different aliases. A family-aware
representation may preserve useful signal without requiring an unreliable
earliest-stream hard decision. This is a hypothesis, not a claimed discovery.

## Fixed factorial experiment

- Control: unchanged V2 (75% history CatBoost, 25% periodicity heuristic).
- A: add family identity (counts/shares/alias counts of outgoing card payments).
- B: add outgoing-payment recurrence, pooled across exact-description/currency streams.
- A+B: both blocks and recurrence resolved by family, including family/currency
  timelines and strongest exact-description stream evidence. This interaction
  explicitly tests combining aliases; it does not assert they are the same merchant.
- Full V3: identical to A+B, with no extra tuning or hidden component.
- Ensemble: one fixed 50/50 probability mixture of full V3 and V2.
- Javier focused correction: reproduce the fixed char model and threshold 0.85,
  preserving V2 none, correcting only music/streaming; assess matched TRAIN OOF.

The CatBoost recipe remains 300 iterations, depth 4, learning rate .05,
balanced classes, seed 42. Each factorial arm retains the same V2 heuristic
and 75/25 weights. No class-bias tuning, threshold search, or VALID early stopping.
Family maps use presence by TRAIN client, smoothed positive-family lift,
minimum five clients and lift >=1.5. No supervised none mapping enters ML features.

## Isolation and selection

Five stratified folds of unique TRAIN clients, seed 42. For each outer fit,
five inner client folds cross-fit all supervised family features. A held-out
client's label never participates in its map or the fitted model. Final TRAIN
features are also cross-fitted; inference uses a full-TRAIN map. All history
must be strictly before the cutoff; IDs are alignment/grouping keys only.

Evaluate all predefined arms once on VALID after TRAIN OOF is written. Save
per-class F1, accuracy, fixed-eight-class Macro-F1, confusion matrices and
prediction counts. Report the OOF-selected candidate separately from the
highest VALID score. Bootstrap paired client predictions descriptively;
it does not remove model-selection bias.

No learned temporal ranker, extra normalization, or unlabeled priors are added
to V3 unless their isolated reproduction changes the negative evidence.
Pseudo-cutoff metrics remain proxy diagnostics: history lacks official
recurring-family ground truth at those dates.
