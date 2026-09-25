# Research decisions and remaining hypotheses

All numbers below refer to executed records in `experiment_results.jsonl`.
The frozen final validation result and reproduction evidence are generated in
`final_results.md`; no score here should be substituted for that result.

## Kept

- **Shift-aware training.** Feature-only train-versus-validation/test AUC was
  0.960/0.976. The ordinary-CV winner scored 0.6851 OOF but only 0.1772 on the
  first official holdout check. Fixed client-preserving corruption views
  materially improved transfer. Text masking, category confusion and noisy
  merchant names are separate corruption mechanisms.
- **Latent stream and symmetric family representations.** Pooled recurrence
  ranking improved ordinary OOF from about 0.54 for direct tabular/template
  models to 0.63, and joint family/none ranking to about 0.68. All eight
  candidate rows exist for every client; `family_index` is the hypothesized
  family, never the client's true label.
- **Unlabeled price profiles.** Semantic anchors in the independent pretraining
  histories supply broad family price ranges. Soft candidate assignment gives
  complementary errors when MCC or descriptions are corrupted.
- **Refund evidence.** Under test-like corruption, refund-only contextual
  features reached 0.6073 versus roughly 0.568 for fee-only or time-only
  additions. The full payment-context version reached 0.6107 for seed 42.
  This is an empirical predictive association, not proof that a refund causes
  cancellation. Amount/currency matching may itself be imperfect.
- **Dedicated none detector and seed averaging.** The final compact system
  compares family rankers with a separate none classifier. All three tested
  seeds are retained: harsh-stress F1 0.6107, 0.6102, 0.5963.
- **Small complementary blend.** A fixed 25% contribution from the earlier
  diverse robustness rankers improved mean stress F1 by 0.0090. Its retention
  threshold was fixed at 0.005 before the final holdout check.

## Rejected or not retained

| Direction | Executed evidence / reason |
|---|---|
| Majority/prior | 0.05747 macro-F1 despite 29.85% accuracy. |
| Word/character/enriched text | Ordinary OOF 0.4885 / 0.4823 / 0.5054; highly vulnerable to masking. |
| Generic direct tabular models | Logistic 0.5134; CatBoost 0.5407 ordinary OOF. |
| Maximum template diversity | 0.4571; correlated with targets but insufficient. |
| Simple stream rules | About 0.377–0.499; real candidate competition and none require more than one rule. |
| Direct wide stream classifier | 0.5972 ordinary OOF versus 0.6303 for pooled family learning. |
| Extra circular-clock ranking features | 0.6766 ordinary OOF versus 0.6851 without the extra block; recurrence features remain, but this added block was not retained. |
| Removing all text statistics | More stable under masking but lower performance; semantic evidence still helps. |
| Auxiliary historical labels mixed into training | No improvement on the masking stress conditions; did not scale to all 10,000 unlabeled clients. |
| Auxiliary historical task | 0.7698 CV on its explicitly reconstructed target, **not challenge performance**. Direct transfer was about 0.47–0.49; adding its scores did not improve the final compact model. |
| Two-observation amount candidates | Test-like OOF 0.6081 versus 0.6107 for the selected three-observation definition. |
| Class-specific OOF decision biases | Improved decision-fit diagnostics but not the first frozen holdout check (0.5866 versus 0.5873); final decision remains argmax. |
| Large sequence/embedding models | Not trained: simpler text models and explicit recurrence features localized the signal, the labeled sample is only 2,000 clients, and auxiliary transfer did not justify a larger training effort. This is a research prioritization decision, not an experimental result. |

## What remains uncertain

The exact label generator, future cancellations/new subscriptions, and its tie
handling are not disclosed. Apparently regular visible streams do not always
match the target. We did not prove a Bayes-optimal ceiling or reconstruct the
generator exactly. Therefore the evidence supports “0.80 was not achieved,”
not “0.80 is mathematically impossible.”

The official holdout was accessed in two frozen selection batches, followed
by reproduction. It is no longer a pristine never-seen test set; all accesses
are logged. The hidden challenge score is unknown. Feature-only test audit
informed robustness assumptions, but no test labels were available or used.

## Remaining hypotheses, ranked by expected value

1. **Better weakly supervised stream identity.** Learn joint amount/text/MCC
   denoising from the clean pretraining histories, with held-out corruption
   seeds and train-side nested evaluation. Could improve sparse/masked family
   assignment; likely incremental, and hard negatives matter.
2. **Explicit continuation/survival model.** Learn stream termination and
   continuation from multiple historical cutoffs, including refund trajectories.
   The failed auxiliary target transfer shows the risk of mismatch with the
   undisclosed official target definition.
3. **More representative labeled data or generator clarification.** The
   strongest path to a major improvement would be authoritative clarification
   of label construction or additional properly separated shifted training
   labels. This cannot be simulated into a claimed hidden-test result.
4. **Small permutation-equivariant sequence/set model.** Only after a cheap
   prototype beats the compact model under fresh corruption seeds. A larger
   neural model without that evidence has low expected value.

No client, class, difficult cohort, or losing seed was removed to improve a
reported score. No classifier uses client IDs, file ordering or validation
labels as features. A clean raw-data rebuild is the final acceptance gate.
