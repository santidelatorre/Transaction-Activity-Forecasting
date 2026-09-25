# Stream identity and relative-time representation: source audit

Audit date: 2026-09-25. Scope: only the worktree created from
`research/import-v2-stream-identity`, source commit
`e4aa4c58175242e198cefd32d6ac4558145523af`. No other branch, checkout, or history
was inspected. This review does not modify the prediction package, fit models,
or read new official validation labels. Existing committed results are treated
as historical evidence, not as newly reproduced measurements.

## Main conclusion

This implementation already represents time relative to the prediction cutoff
and attaches substantial temporal evidence to individual family candidates.
The expert's advice should therefore become a test of **representation fidelity
and actual model reliance**, rather than a claim that time or family linkage
is missing.

The highest-value uncertainties are whether approximate amount groups really
preserve one payment stream, whether their continuation is distinguished from
their semantic identity, and whether missing candidates are represented without
distorting temporal competition. Changing days to seconds adds no information;
improving the meaning of a candidate's age, cadence, refund match, and due date
could. No improvement in challenge Macro-F1 follows automatically from any
source-level issue identified here.

## 1. What reaches the final predictor

The production path is `build_features()` in
[`model.py:16`](../../src/ubs_recurrence/model.py#L16):

1. Build transaction-description template features.
2. Extract outgoing card-payment groups, separated by client and currency,
   using amount neighborhoods and broader semantic family groups.
3. Attach one or more groups to each hypothesized recurring family, using
   semantic evidence, MCC and independently learned price support.
4. Build eight rows per client, including a `none` candidate.
5. Select compact per-stream statistics and add client-level context.
6. Add matching-refund, fee, and time-of-day evidence.

The final predictor is not a single model with a single feature list.

| Component | Final weight | Time and identity information actually supplied |
|---|---:|---|
| Three compact rankers plus separate none classifiers | 75% total | Per-family amount0, amount1 and broad0 timing; recency, intervals, predicted next offsets, overdue ratio, compact activity state, family confidence, refund context; client-level aggregates |
| Hard-family LightGBM ranker | One third of remaining 25% | Broader ranking table with hard group-family assignment, including raw circular phase statistics and template competition |
| Soft-family LightGBM and XGBoost rankers | Two thirds of remaining 25% | Broader ranking table with price-supported candidate assignment, raw circular phase statistics, and template competition |

Implementation: [`model.py:51`](../../src/ubs_recurrence/model.py#L51) and
[`model.py:87`](../../src/ubs_recurrence/model.py#L87).

**Important distinction:** `clocks=False` does not remove all phase or temporal
features. It disables the additional fitted-clock selection block in
[`ranking.py:30`](../../src/ubs_recurrence/ranking.py#L30). The original numeric
stream columns already include `phase_strength_*`, `phase_next_*`, and
`period_error_*`; those survive in the legacy ranking tables. The compact
selection drops these particular raw columns but retains its own activity and
next-offset transforms. An ablation of only `clocks` cannot answer whether the
ensemble uses relative time.

The binary none classifier aggregates candidate features by minimum, maximum
and mean after removing `family_index` and `is_none`
([`model.py:33`](../../src/ubs_recurrence/model.py#L33)). Its score replaces the
compact ranker's none score; the other seven scores are renormalized to the
remaining mass. Consequently, explaining only a family ranker's raw score
would not explain the final recurring-versus-none decision.

## 2. Relative-time features already present

`extract_streams()` converts timestamps to signed days relative to the fixed
cutoff: `t = (timestamp - cutoff) / 86400 seconds`
([`streams.py:67`](../../src/ubs_recurrence/streams.py#L67)). `fast_stats()` then
computes:

| Feature | Meaning |
|---|---|
| `first_age`, `last_age` | Days from first/last observed group payment to cutoff |
| `gap_median`, `gap_mean`, `gap_recent`, `gap_last` | Historical cadence estimates in days |
| `gap_std`, `gap_mad`, `gap_cv`, `linear_residual` | Cadence uncertainty or irregularity proxies |
| `next_median` | Median interval minus age of last payment; a signed expected offset from cutoff |
| `next_recent`, `linear_next` | Alternative signed next-offset estimates |
| `overdue_ratio` | Age of last payment divided by median interval, with denominator floor one day |
| `count_30/60/90/180` | Recent evidence density |
| `phase_strength_*` | Circular concentration at several candidate periods |
| `phase_next_*` | Positive circular phase relative to the cutoff |

Source: [`streams.py:21`](../../src/ubs_recurrence/streams.py#L21).

These variables already distinguish a monthly payment last seen 28 days ago
from a monthly payment last seen four days ago. They are observations and
extrapolations, not disclosed future dates. A negative `next_median` means the
simple clock is already overdue; it does not prove cancellation. A positive
phase estimate does not prove that the stream will continue.

All clients share the same cutoff. Subtracting a common date from a numeric
timestamp, or converting days to seconds, does not create new information.
The useful semantic change is the relationship between age, period,
uncertainty, and continuity. The official label is a family, not an exact
payment date; improving a historical date proxy is not automatically improving
that family target. The target-construction details remain undisclosed in the
branch's [`requirements.md`](../requirements.md).

## 3. Concrete representation risks

### R1. Reconstructed group identity can combine unrelated transactions

Amount candidates join neighboring log amounts within a radius of 0.035 and
retain groups of at least three payments. They are separated by currency, but
not by merchant identity, description, or coherent calendar phase. Broad
family groups can also combine several subscriptions in the same family.
These are candidate processes, not ground-truth stream identities.

Source: [`streams.py:13`](../../src/ubs_recurrence/streams.py#L13),
[`streams.py:72`](../../src/ubs_recurrence/streams.py#L72).

For example, two independent monthly payments of similar amounts can produce
one apparent fortnightly process. Their median gap is then a property of the
merged sequence, not of either subscription. Adding more sophisticated clock
features to that merged group may confidently describe the wrong process.
This is a plausible mechanism; its dataset prevalence was not measured in
this read-only audit.

### R2. Family linkage exists, but candidate retention loses some alternatives

`family_features()` orders eligible groups by
`count * exp(-last_age / 60)`, keeping the first two groups for amount and
broad-family representations
([`streams.py:108`](../../src/ubs_recurrence/streams.py#L108)). This ranking is
based on evidence volume and recency, not due-time imminence or recurrence
quality. The compact model then retains amount0, amount1, and broad0, dropping
broad1 ([`compact.py:20`](../../src/ubs_recurrence/compact.py#L20)).

Therefore, "there is a temporal feature for each family" is true, but it does
not guarantee that the relevant stream is among the retained summaries. The
soft family assignment also lets the same group appear under several families;
that is useful uncertainty representation, but it makes family competition
partly a contest over shared evidence.

The hard expert resolves a complete zero-evidence family tie using `argmax`,
which chooses the first family, cloud
([`streams.py:81`](../../src/ubs_recurrence/streams.py#L81)). The soft experts
offer alternative hypotheses; the hard fallback should not be explained as
actual observed cloud identity.

### R3. Compact period rounding can misrepresent unsupported cadence classes

The compact activity transform accepts median intervals from 5 to 120 days,
then rounds to one of `[14, 28, 30, 60, 90]`. It marks activity when
`last_age < 1.35 * rounded_period` and maps the signed next offset modulo that
rounded period ([`compact.py:25`](../../src/ubs_recurrence/compact.py#L25)).

A tiny synthetic feature probe, executed without model fitting, used five
weekly payments, a median interval of seven days, and a last payment eight
days before cutoff. The existing code returned:

```text
period_rounded = 14
active = 1
next_active = 13
```

The weekly modulo residual would instead be six days. Raw gap and age features
remain available, so this does not establish that the model must make an
incorrect prediction. Moreover, the committed cadence audit found zero
weekly-like candidates among 2,398 regular groups with at least five events
([`cadence_audit.json:3`](../cadence_audit.json#L3)). Treat this primarily as a
domain-boundary and interpretation issue unless broader train-side incidence
justifies a predictive experiment. Annual cadence is also excluded from the
compact activity transform, but the history is too short to validate annual
recurrence with the audit's five-event requirement.

### R4. A missing/inactive next payment is encoded as a numerical 999

`next_active` becomes 999 for both inactive and absent candidates. Its global
min/max/mean and ranking are calculated while masking only -999
([`compact.py:31`](../../src/ubs_recurrence/compact.py#L31),
[`compact.py:38`](../../src/ubs_recurrence/compact.py#L38)). The none classifier
also masks only -999. The artificial none row contributes another inactive
999 value.

The synthetic example above had one next offset of 13 and seven unavailable
offsets of 999. The reported `amount0_next_active_global_mean` was 875.75 days.
That is not a meaningful average expected payment date. It combines timing
and missing-candidate count into one numeric quantity.

This may nevertheless be a predictive missingness proxy that boosted trees
learn to use. Simply removing it could worsen performance. The discriminating
comparison is explicit candidate-availability and active-count features plus
masked time summaries, retaining the same other inputs. Current `stream_count`
and raw count columns already provide some availability information; the
representation is not wholly unaware of missingness.

### R5. Refund matching reconstructs stream currency from a timestamp proxy

The numeric family table does not preserve selected-stream transaction
membership or currency. `payment_context()` reconstructs currency using the
card payment closest in age to the group's last observation, then retrieves
all payments and refunds of similar amount in that currency
([`payment_context.py:13`](../../src/ubs_recurrence/payment_context.py#L13)).

A synthetic probe used two card payments at the same last timestamp:
USD 500 and CHF 10, followed by a CHF 10 refund. For a represented CHF 10
stream, swapping the tied card rows changed extracted `refund_count` from zero
to one. Both executions used the same transaction set. This demonstrates a
tie-order dependence in this function, not its prevalence in the actual UBS
data. The main loader sorts by client/time but uses stable sorting, so it does
not provide an independent tie-breaker.

Even without timestamp ties, refunds are matched to an amount neighborhood,
not uniquely to the payment that generated them. Same-price subscriptions
can share refund evidence. The ten-day `refund_after_last` flag describes an
association, not proof of cancellation. Retaining currency and transaction
membership as audit metadata would make this evidence traceable without
using identifiers as predictive inputs.

### R6. Some "per event" features divide a proportion twice

`own_semantic` and `own_mcc` already contain fractions of events in a stream
([`streams.py:33`](../../src/ubs_recurrence/streams.py#L33),
[`streams.py:116`](../../src/ubs_recurrence/streams.py#L116)). The compact block
divides those fractions by count again, alongside true counts such as
`own_template_count`
([`compact.py:33`](../../src/ubs_recurrence/compact.py#L33)).

In the synthetic five-payment example with perfectly consistent semantics,
`own_semantic_per_event` was 0.2, while `own_template_count_per_event` was 1.0.
The first variable is an inverse-length interaction, not semantic purity.
The original fractions are also retained. This is a naming/interpretation
problem and a candidate ablation, not proof that the learned interaction is
harmful. Explanations must not describe this field as the proportion of
matching transactions.

### R7. Calendar mechanics and continuity are approximated, not identified

The linear fit indexes observations with consecutive integers. Missed payments
therefore distort the fitted period. Circular phase recognizes some missed
cycles, but compact features omit those raw phase columns; the legacy experts
retain them. A 30-day approximation also differs from same-day-next-month.

The committed historical audit reports mean absolute date errors of 2.799 days
for median-interval extrapolation, 2.664 for same-day-next-month, and 2.848 for
next-business-day adjustment
([`cadence_audit.json:15`](../cadence_audit.json#L15)). The improvement of roughly
0.135 day for same-day-next-month is real in that diagnostic but small, and it
is not a challenge Macro-F1 difference. None of these date rules reveals
future cancellations or the official target tie-breaking rule.

### R8. Nominal amount context is not currency-normalized throughout

Stream extraction is correctly currency-separated. However, family price
profiles pool nominal amounts across currencies
([`price_prior.py:8`](../../src/ubs_recurrence/price_prior.py#L8)), and global
amount means/stds also pool all transactions
([`compact.py:7`](../../src/ubs_recurrence/compact.py#L7)). This may be useful
in this synthetic dataset, but it is not an economic spending amount in a
common currency. Currency-specific profile support is a possible future
identity experiment; real-time exchange-rate conversion is neither necessary
nor justified merely to reinterpret this dataset.

## 4. Existing experimental evidence constrains the next investigation

The following are committed historical results, not reruns by this audit.
Exact records are in [`experiment_log.md`](../experiment_log.md).

| Comparison | Reported Macro-F1 | Interpretation |
|---|---:|---|
| Broad family ranking, extra clocks enabled | 0.676613 original train OOF | Added fitted-clock block did not improve this protocol |
| Same ranking, extra clocks disabled | 0.685062 original train OOF | Still includes raw stream timing and circular features |
| Compact hierarchical baseline, no payment context | 0.574253 test-like train OOF | Comparator for context studies |
| Add time-of-day/weekend context only | 0.567674 test-like train OOF | Calendar-hour context did not rescue performance |
| Add fee context only | 0.568655 test-like train OOF | Fee-only addition did not help this comparison |
| Add refund context only | 0.607280 test-like train OOF | Strongest specific new context in this ablation |
| Add all payment context | 0.610699 test-like train OOF | Selected seed-42 compact component |
| Add two-observation amount candidates | 0.608056 test-like train OOF | More candidate coverage alone did not improve score |
| Final frozen ensemble | 0.619493 official VALID | External check reported by branch; not hidden-test score |

This argues against claiming that "the missing feature is time of day" or
rerunning the previously rejected additional-clock experiment unchanged. It
supports asking why refund/continuity evidence matters and whether it is
matched to the correct stream.

The frozen error report records 353 official-validation errors: 197 wrong
recurring family, 95 true recurring predicted none, and 61 true none predicted
a family. A target-family cohort without a three-event amount candidate has
140 clients and 123 errors
([`final_error_analysis.md:5`](../final_error_analysis.md#L5)). These target-aware
cohorts localize difficulty after the fact; they must not become features,
client exclusions, or an unreported tuning set. The broader semantic group
can still exist when an amount candidate is absent.

The branch already discloses repeated frozen access to official VALID. Fresh
train-side outer folds and new corruption seeds are useful robustness evidence,
but do not make past development data untouched. Fixed feature-only corruption
scenarios preserve timestamps and amounts while altering text/MCC
([`augmentation.py:21`](../../src/ubs_recurrence/augmentation.py#L21)); they do
not validate robustness to cancellations, timing drift, or missing events.

## 5. Five discriminating experiments

All experiments should preserve exact client partitions, all eight classes,
and every augmented copy within its training client's fold. Freeze candidate
definitions and the feature-group inventory before interpreting ablations.
Prefer paired client-level OOF comparisons with per-class metrics and paired
bootstrap differences. A hidden-test gain or 0.80 score cannot be promised.

### E1. Measure whether time changes decisions after stream identity is fixed

Hold extracted stream assignment constant. Compare the frozen compact model
against refits removing: (a) all recurrence-time fields, (b) only phase/due
offsets and cycle ratios, (c) only identity/semantic fields, and (d) refund
context. Include raw and derived copies, competition summaries, and their
none-classifier aggregates; otherwise information survives the supposed
ablation. Test the final blend separately because the legacy experts can
restore removed information.

Keeping stream extraction fixed measures downstream use of timing. A separate
candidate-selection ablation is needed to measure the time dependence of
`count * exp(-last_age/60)`. Do not conflate those effects.

Success criterion: stable paired evidence that a specific block supplies
incremental predictive information across original and held-out corruption
views, with explicit none-versus-family and family-versus-family changes.
This is an investigation result even if the frozen baseline wins.

### E2. Separate missingness, continuity, and due-time uncertainty

Create a compact variant with explicit candidate_present and active flags,
counts of observed/active candidates, masked next-date summaries, a signed
late/early offset, and continuous normalized lateness. Change this block only;
keep identity and refund inputs fixed. Add a separate coverage audit for
cadences outside the present rounded-period grid before expanding that grid.

Compare against the original 999-based representation. Use small coherent
history perturbations at inference to check decision sensitivity, but do not
score perturbed histories against unchanged labels when the intervention could
change their real future. Feature-level synthetic correctness and measured
challenge improvement are separate outcomes.

### E3. Preserve stream provenance and test identity collisions

Retain source-row membership, currency and a deterministic stream key as
nonpredictive audit metadata. First make refund association refer to the
selected group instead of the nearest timestamp proxy. Separately compare a
restricted split of same-price groups when descriptions or temporal phases
support two distinct recurring processes. Use multiple narrow variants; do
not change grouping, refund matching, and ranking heuristics simultaneously.

Report counts of changed candidates, changed refund matches, and score changes.
Evaluate whether any gain is concentrated in amount-collision clients or
survives beyond them. Synthetic invariance to tied-row order is a correctness
gate even if that edge case does not occur in this archive.

### E4. Learn continuation from pre-cutoff histories, without redefining success

On independent unlabeled histories, construct multiple earlier cutoffs and
predict whether an already-observed candidate actually reappears. All stream
discovery and identity decisions for each example must use its prefix; suffixes
may supply only historical outcomes. Keep every cutoff of the same client in
one outer split. Preserve censoring/insufficient-observation indicators.

Use the resulting continuation score as an optional feature in a small
challenge-model experiment. The branch already rejected a different historical
auxiliary target and expert. A high score on this surrogate is not success:
it must add value on challenge-label OOF under the same budget. This direction
is lower priority until a cheap prototype beats the refund baseline.

### E5. Test whether the model's claimed explanation matches its behavior

For a preregistered set of held-out TRAIN clients covering correct predictions,
wrong-family predictions, false-none and false-family outcomes, publish:
original source rows, selected group membership, raw/derived temporal evidence,
each expert's score, binary none score, final probabilities, and the top
alternative family. Add grouped contribution or permutation diagnostics for
time, identity, and refund blocks, explicitly accounting for correlated inputs.

For ranker explanations, show contribution to the chosen-family-versus-runner-up
score difference; explain the none detector separately. Nonlinear softmax,
conditional renormalization and ensemble averaging mean that one ranker's SHAP
values are not exact additive explanations of the final class probability.
Use coherent whole-history perturbations and rebuild features rather than
editing one time-derived column while leaving its copies unchanged.

The existing [`explanations.md`](../explanations.md) correctly describes
observed model inputs, not causal or exact feature-attribution explanations.
E5 would extend that evidence instead of replacing it with model-generated
stories.

## 6. Honest pitch implications

The existing demonstrated distinction is reconstruction of recurring candidate
streams plus training for observed text/MCC degradation and separate none
detection. Relative time is an existing design feature, not a newly discovered
missing ingredient. The strongest existing learning story is that an ordinary
CV winner failed under deployment shift, that refund context improved the
train-side stress protocol, and that the frozen ensemble was externally checked.

A focused demonstration can show one client's actual histories, two competing
candidate streams, their cycle positions and refund evidence, and the final
family decision. A second difficult example should show ambiguity or rejection.
Use actual inspected inputs and measured score differences. Do not promise
validated payment dates, causal cancellation detection, removal of all noise,
or a future 0.80 Macro-F1.

## 7. Work performed for this review

- Read only source and committed reports inside the allowed worktree.
- Traced feature flow through all final experts and the none-detector override.
- Executed two small synthetic, read-only code probes using the branch's
  `compact_features()` and `payment_context()`. No labels or models were used.
- Synthetic environment: bundled Python with NumPy 2.3.5 and pandas 3.0.1;
  these are diagnostic probes, not the pinned environment of the historic
  benchmark. No benchmark equivalence is claimed.
- Wrote this report only. No production implementation, model fit, submission,
  commit, push, or external message was performed by this audit subtask.
