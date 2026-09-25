# Stream identity, time representation, and an evidence-led pitch

**Same-source update after this review:** the allowed branch advanced to
`d5dddfd` while the experiments ran. Its README now explicitly marks the
historical 0.619493 result INVALID under a strict rule against VALID-informed
selection. The historical pitch below must call it a development-validation
result exposed to selection, never an untouched evaluation. See the
[completed investigation](README.md) for this qualification and the new
TRAIN-only attribution and ablation evidence.

## Scope and evidence standard

This investigation studies only the files supplied by `research/import-v2-stream-identity` at source commit `e4aa4c5`. No other branch, checkout, or historical implementation was consulted. Earlier conversational claims about other implementations are not evidence for this report.

The review is read-only except for this report. It did not train a model, open raw validation labels, generate a submission, or rerun the earlier raw-data reproductions. Historical measurements below come from this branch's committed experiment ledger and reports. Relevant code was inspected to determine what those measurements actually test.

An independent arithmetic check of all **95 unique experiment records** in [experiment_results.jsonl](../experiment_results.jsonl) recalculated Macro-F1 and accuracy from each eight-class confusion matrix. All 95 matched the recorded values within `1e-10`. This verifies metric consistency, not the provenance of the original predictions or a new reproduction of training.

The most relevant source files are [streams.py](../../src/ubs_recurrence/streams.py), [compact.py](../../src/ubs_recurrence/compact.py), [ranking.py](../../src/ubs_recurrence/ranking.py), [payment_context.py](../../src/ubs_recurrence/payment_context.py), [compact_experiments.py](../../scripts/compact_experiments.py), [ranking_experiments.py](../../scripts/ranking_experiments.py), and [final_report.py](../../scripts/final_report.py).

## Main conclusion

The expert's question is productive, but the branch does **not** simply feed absolute payment timestamps into a classifier. It already measures time relative to the cutoff, relates recency to cadence, attaches these quantities to family candidates, and lets candidates compete. The investigation should therefore ask which parts of that representation survive noisy identity assignment and which actually influence the decision.

The historical evidence supports three concrete statements:

1. **Recovering useful recurring candidates and preserving family identity matter.** Candidate ranking is substantially stronger than several text and direct tabular baselines, although these architecture comparisons change multiple ingredients.
2. **Additional time detail is not automatically beneficial.** The tested additional clock-selection block and the tested hour/weekend context did not improve the relevant comparisons.
3. **Robustness to ambiguous descriptions and stream continuation evidence are more convincing differentiators.** The branch documents a large train/deployment shift and a controlled gain from refund context under fixed corruption stress.

It is premature to say that any newly proposed relative-time feature will reach 0.8. The best frozen score documented in this branch is **0.619493 Macro-F1 on the supplied official validation set**. The hidden-test/leaderboard score is unknown.

## What the temporal representation already means

Let `c` be the cutoff, `t_last` the most recent observed event in a candidate, and `g` its median observed interval. The code already constructs equivalents of:

| Question the model needs to answer | Existing representation | Important limitation |
|---|---|---|
| How long since this candidate last paid? | `last_age = (c - t_last)` in fractional days | Candidate membership can be wrong. |
| What is its usual interval? | `gap_median`, `gap_mean`, `gap_recent`, `gap_last` | Missing observations and mixed streams distort gaps. |
| Where is it in its cycle? | `overdue_ratio = last_age / max(gap_median, 1)` | The one-day floor is an explicit numerical convention. |
| How far from a simple expected next event? | `next_median = gap_median - last_age`; recent-gap and fitted-line alternatives | A negative value is evidence of a missed expected occurrence, not an observed future event. |
| Does the stream appear active? | Count, period-range checks, and `last_age < 1.35 * rounded_period` | A fixed threshold is a heuristic; apparent activity does not prove continuation. |
| Which family may occur sooner? | Per-client family ranks, relative values and compact global comparisons | Relative ranks are only useful if candidate identity and cadence are useful. |
| How regular is it? | Gap dispersion, linear fit residual, broad/amount candidate comparison | Approximate clusters can create false regularity or suppress true regularity. |

These quantities are present per candidate family, with separate `amount0`, `amount1` and `broad0` evidence retained by the compact representation. Consequently, "we must add recency divided by period" describes an already implemented idea in this branch.

Changing units from days to seconds is a rescaling of the same information. It does not reveal additional event timing if fractional days already contain it. A raw date versus an age at a single fixed cutoff is also an affine transformation. Tree split order is theoretically invariant under strictly monotone unit conversion; practical rounding or numerical behavior can differ. The useful research question concerns representation, grouping, missingness and learning bias, rather than unit conversion alone.

`next_active` uses a modulo transformation after the activity gate. Modulo can make an overdue candidate look close to another future cycle, so signed lateness and missing-cycle evidence should remain available when interpreting that quantity. They are partly retained in this branch through `next_median`, `last_age` and `overdue_ratio`. A new study should test coherent groups of these variables, rather than interpreting `next_active` alone.

## What the historical comparisons establish

All train OOF and corruption results below are exploratory model-selection estimates. Reusing a set of client folds or corruption seeds for many decisions does not create an independent final test. No paired uncertainty interval for these deltas is recorded in the inspected committed reports.

### Representation and filtering comparisons

| Comparison | Evaluation | Baseline F1 | Alternative F1 | Difference | What it supports |
|---|---|---:|---:|---:|---|
| Wide direct stream classification → pooled family learning | Original train OOF, seed 42 | 0.597158 | 0.630291 | +0.033133 | Sharing candidate-level structure is promising; architecture and none handling change together. |
| Pooled streams → same pooled experiment with background-description filter | Original train OOF, seed 42 | 0.630291 | 0.633732 | +0.003441 | A small observed filtering benefit in this setting; no evidence that filtering alone explains the final result. |
| Ranker without → with extra clock-selection block | Original train OOF, seed 42 | 0.685062 | 0.676613 | -0.008449 | The added clock-selection representation hurt this experiment. Base recurrence features were present on both sides. |
| Three-event → two-event amount candidate definition in the compact payment model | Test-like train OOF, seed 42 | 0.610699 | 0.608056 | -0.002643 | Broadening this candidate threshold did not improve the selected stress comparison. It does not rule out all sparse-history methods. |

Relevant IDs: `streams_direct_e0.035_s42`, `streams_pooled_e0.035_s42`, `streams_pooled_e0.035_filtered_s42`, `ranking_ranker_clocksFalse_idTrue_s42`, `ranking_ranker_clocksTrue_idTrue_s42`, and `compact_l15_hierTrue_payments_sparse_test_like_s42`.

The `clocks=False` setting is especially easy to misreport. It disables an *additional* learned/selected clock block in `ranking_features`. It does not delete all gap statistics, recency, phase information already present in stream features, or candidate-relative information. Therefore these records do not answer "does time help at all?"

### Context ablation: the strongest narrow feature comparison

The compact experiments use the same seed 42, five client folds, 15-leaf hierarchical architecture, estimator parameters and three fixed training noise views. The base compact model already contains recurrence timing. The following additions therefore isolate payment-context groups more closely than the broad baseline ladder does.

| Added context | Original F1 | Valid-like F1 | Test-like F1 | Test-like change vs base |
|---|---:|---:|---:|---:|
| No payment-context block | 0.625282 | 0.600103 | 0.574253 | — |
| Hour variability and weekend fraction only | 0.619759 | 0.601794 | 0.567674 | -0.006579 |
| Fee context only | 0.628351 | 0.592512 | 0.568655 | -0.005598 |
| Refund context only | 0.643210 | 0.628304 | 0.607280 | +0.033027 |
| All payment context | 0.652254 | 0.632888 | 0.610699 | +0.036446 |

Relevant IDs follow `compact_l15_hierTrue[_payments[_time|_fee|_refund]]_{scenario}_s42`.

The label `time` in this experiment means the additional `hour` and `weekend` fields selected by `compact_experiments.py`. It does **not** mean the entire temporal representation. The refund result suggests continuation/termination-related context is useful beyond recurrence cadence under these stress conditions. It does not establish that a refund causes cancellation or that this effect alone transfers by the same amount to hidden test.

The full-context gain over refund-only is much smaller: +0.003419 on test-like OOF. Because tree interactions and stochastic column sampling are involved, the table does not decompose the final score into additive causal contributions. The narrow context comparisons are single-seed; the branch later checks the complete model at three seeds, not every ablation at three seeds.

### Robustness and final generalization

| Evidence | Measurement | Correct interpretation |
|---|---|---|
| Train versus validation/test classifier | AUC 0.960150 / 0.976353 | Histories are distinguishable by split; this is not family-prediction performance. |
| Original-CV winner | 0.685062 train OOF; 0.177187 official validation | Strong ordinary CV did not transfer for this model. Different datasets make the difference a generalization gap, not a paired feature ablation. |
| First frozen robust ensemble | 0.587318 official validation | Robust model transferred much better than the predeclared control. Multiple components differ. |
| Same first ensemble with OOF-fitted decision biases | 0.586551 official validation | The decision-fit improvement failed to confirm externally; biases were rejected. |
| Final compact three-seed model | 0.627467 valid-like / 0.605005 test-like OOF | Seed-averaged component on exploratory stress data. |
| Add fixed 25% complementary ensemble | 0.637367 / 0.613157 | Mean stress gain +0.009026 met the predeclared +0.005 retention rule. |
| Final frozen prediction package | 0.619493 official validation; 0.647 accuracy | Main reportable task result, not a leaderboard score. |

The ordinary control versus robust ensemble comparison is compelling operational evidence, but it cannot attribute the whole gain to changing relative time, filtering groceries, refunds, or any single feature.

## Noise: what can honestly be said was discarded or handled

There are several different operations; they should not be collapsed into "we cleaned the dataset".

1. **Candidate construction:** `extract_streams` initially restricts recurring candidate extraction to outgoing card payments, then groups by client and currency. This is a domain restriction on candidate discovery, not deletion of all other transactions from every model component.
2. **Background-description filtering:** the optional regex filter removes categories such as pharmacy, grocery, coffee and travel descriptions from candidate extraction. Its specific historical comparison yielded +0.003441 ordinary OOF. Final raw-data model construction calls filtered extraction, but other features still use broader histories. Avoid claiming that all such purchases are irrelevant for prediction.
3. **Robustness training:** fixed synthetic masking, MCC confusion and merchant-name corruption perturb histories during fitting while preserving client grouping. They teach resilience; they do not recover ground-truth clean text.
4. **Approximate stream matching:** nearby amounts within one currency support candidate identity. Equal prices do not guarantee equal merchants. False merges remain a documented failure mode.
5. **Refund evidence:** refunds are context, not automatically discarded noise. Their matching is approximate, and current matching recovers currency from the closest-in-time card event rather than retaining a definitive payment-stream key throughout.

The pitch can state that the system distinguishes plausible recurring evidence from surrounding activity and trains against identity corruption. It should not promise complete denoising or exact merchant reconstruction.

## What the branch can and cannot explain about a prediction

[explanations.md](../explanations.md) provides real observed model inputs: counts, median intervals, age of the last event, amounts and matching refunds. This is useful evidence presentation. It is not a measurement of the ensemble's actual local feature contributions.

`final_report.py` selects the **highest-scoring correctly predicted validation client for each class**. It then displays `amount0` if it has at least three events, otherwise `broad0`. This is a post-hoc presentation rule, not proof that the displayed stream drove the winning score. A demo should explicitly describe these as selected examples. They are not representative estimates of reliability and cannot be used to infer a typical success rate.

Historical ranking experiments write tree split-importance CSVs into ignored experiment folders. Those are neither final-ensemble local explanations nor causal feature effects. No final-ensemble SHAP decomposition, grouped counterfactual test, or feature-removal attribution is recorded in the committed evidence inspected here.

To answer the expert's "which feature do you base your prediction on?", a defensible next audit would:

- Show all competing family candidates for a fixed client, with the evidence source and extraction confidence.
- Separate existence of a plausible recurring stream, identity of its family, and its apparent continuity. Missing identity should not silently mean `none`.
- Measure contributions at the actual ensemble output, accounting for the separate none detector, rank-score normalization and blending; a single component's tree attribution is not the final decision explanation.
- Perturb raw observed history coherently, recompute every feature and compare the same frozen model's output. Changing one cached recency feature while leaving its derived cycle variables unchanged creates an inconsistent input.
- Inspect a predeclared mix of correct, incorrect, low-margin and masked clients, rather than only high-confidence correct examples.
- Use true retrain-without-feature-group OOF comparisons to measure predictive utility. Output sensitivity alone does not show that the changed behavior improves Macro-F1.

Potential coherent interventions include moving an entire candidate earlier by one cycle while keeping it before cutoff; masking only its descriptions; removing one event and recomputing cadence; or adding/removing a historically observed matching refund in a controlled synthetic scenario. These are model-behavior tests. Their labels are not known to remain valid under arbitrary event changes, so they are not automatically scored task examples or causal banking claims.

## A date-diagnostic caveat worth preserving

[cadence_audit.json](../cadence_audit.json) contains 12,400 retrospective next-observed-payment forecasts within recovered monthly-like training streams:

- Median-interval extrapolation: mean absolute error 2.7989 days.
- Same-day-next-month: 2.6643 days.
- Weekend adjustment to next business day: 2.8481 days.

The +0.1346-day improvement from calendar extrapolation is a descriptive result on a selected cohort, not a demonstrated improvement in merchant-family classification.

Moreover, [calendar_audit.py](../../scripts/calendar_audit.py) identifies amount clusters, filters regularity and decides monthly-like membership using the full observed stream **before** forecasting earlier prefixes. Each date forecast uses prefix gaps, but cohort selection has seen later portions of that historical stream. This is acceptable as retrospective pattern inspection if labeled correctly; it is not a fully prospective prefix-only forecasting evaluation. It should not support a claim that the production system reliably predicts exact payment dates within 2.7 days.

A prospective version would reconstruct and select candidates separately at each prefix cutoff, using only data available there, and then score later observations. That study still needs explicit alignment between its reconstructed target and UBS's undisclosed official family-label construction.

## Where the remaining errors point

The frozen official-validation report records 353 mistakes: 95 recurring-family clients predicted `none`, 61 true `none` clients predicted a family, and 197 wrong-family predictions.

The most informative descriptive cohort is **140 true-family clients without a three-event amount candidate for that family: 123 errors, 12.14% accuracy**. Clients with a regular matching candidate have 74.43% accuracy. These cohorts use true labels for post-freeze analysis and overlap with other cohorts; they are not valid deployable segment identifiers or proof of a causal bottleneck.

They nevertheless prioritize investigation. Making a richer time encoding of a missing or wrongly assigned stream may be less valuable than preserving alternative identities, detecting absent candidates, and retaining honest uncertainty. Simply lowering the candidate minimum to two observations has already failed to improve the recorded stress comparison, so candidate recovery needs more than that one threshold change.

No available evidence proves an information-theoretic performance ceiling, identifies a rival team's method, or establishes that 0.8 is attainable. The exact synthetic future-label generator, cancellations, new subscriptions and tie handling remain uncertain.

## Validation and provenance language for the jury

[holdout_access_log.jsonl](../holdout_access_log.jsonl) records:

| UTC time on 2026-09-25 | Use |
|---|---|
| 02:10:04 | First frozen batch: control, robust ensemble, calibrated variant. |
| 02:32:49 | Second frozen batch: selected final candidate. |
| 02:37:41 | Independent raw-data rebuild reproduction, no new model selection recorded. |

The reports state that all 1,000 validation clients were retained and their labels were not used for supervised fitting. The first batch informed later research, so the supplied validation set is **not an untouched final test after the full research sequence**. The reported client-bootstrap interval [0.5859, 0.6500] conditions on the selected model and sample; it does not include the complete selection process or hidden-test shift.

[reproduction.json](../reproduction.json) records identical class predictions, zero maximum normalized-score difference and byte-identical test submissions across two raw-data builds. Full source fingerprints differ because reporting/research files were added between those runs; the prediction package was documented as unchanged. These are prior committed reproduction records, not additional runs performed for this audit.

## One-minute pitch based on this branch alone

Suggested English script, approximately 130 words; rehearse rather than assuming exact timing:

> A bank statement mixes subscriptions, groceries and transfers. Our system predicts the next recurring payment family by reconstructing candidate streams and comparing their timing, identity and signs of continuity.
>
> Our key finding was that a model scoring 0.685 in training cross-validation fell to 0.177 on the supplied validation set. Descriptions and merchant codes were much noisier. We trained against that shift and tested each addition: extra clock detail did not help, while refund context improved our harsh-noise experiment by 0.033 Macro-F1.
>
> The final system recorded 0.619 Macro-F1 on 1,000 supplied development-validation clients. That set informed model selection, so this is not an independent test score. We can show observed payments behind predictions and where evidence is uncertain. The proposed use is payment reminders and subscription reviews; exact dates and real-bank impact still need validation.

Suggested Spanish version:

> Un extracto mezcla suscripciones, compras y transferencias. Nuestro sistema reconstruye posibles secuencias recurrentes y compara su familia, su ciclo de pago y las señales de continuidad.
>
> El hallazgo clave fue que un modelo con 0,685 en validación cruzada de entrenamiento cayó a 0,177 en la validación proporcionada por UBS. Las descripciones y los códigos comerciales eran mucho más ruidosos. Entrenamos frente a ese cambio y comprobamos cada incorporación: añadir más detalle de reloj no ayudó; el contexto de devoluciones mejoró en 0,033 nuestro experimento con ruido intenso.
>
> El sistema registró 0,619 de Macro-F1 en 1.000 clientes de validación de desarrollo. Ese conjunto influyó en la selección, así que no es un resultado independiente. Mostramos los pagos observados que aportan evidencia y sus límites. La aplicación propuesta es ayudar con recordatorios y revisión de suscripciones; las fechas exactas y el impacto real todavía requieren validación.

The experiment comparison in the middle sentence is a **train-side synthetic-stress comparison**, and the final 0.619 is **official supplied validation**, not a hidden-test score. Keep those labels visible on the slide. Do not present this historical result as an improvement newly achieved by the current audit.

## Three-minute Q&A preparation

| Likely question | Defensible answer |
|---|---|
| What is the technical difference? | We reconstruct approximate streams, retain family-specific cadence and continuity evidence, and train under observed identity corruption. Family candidates compete alongside a separate none detector. |
| What did the expert's suggestion reveal? | Relative time is already present. The remaining questions are whether candidate identity is reliable, whether cycle variables are coherent, and which groups actually change useful predictions. |
| What failed? | Additional selected-clock detail reduced ordinary OOF by 0.00845. OOF-fitted class biases also failed their first external check. Both results prevented us from equating more complexity with improvement. |
| What noise did you remove? | Candidate discovery limits payment types and optionally filters background descriptions; broader context remains. We also train with corrupted text/MCC rather than assuming merchant identity is clean. |
| What proves refunds help? | In the same compact seed-42 stress protocol, refund-only context raised test-like OOF from 0.57425 to 0.60728. This is predictive association under that protocol, not causality or a guaranteed hidden-test gain. |
| Can you explain this exact prediction? | We can show observed candidate evidence. Current reports do not yet prove final-ensemble local feature attribution; selected examples must be presented as evidence, not causal explanations. |
| Is 0.619 your leaderboard score? | No. It is the supplied validation result. The hidden-test score is unknown in this branch's evidence. |
| Did you learn from the validation labels? | They were not used for supervised fitting. Two frozen candidate batches were evaluated and informed research direction; access is logged, and we disclose that selection exposure. |
| Why not 0.8? | It has not been achieved or ruled out. Candidate identity, missing streams, none versus continuation and deployment shift remain difficult. We do not know the full future-label generator. |
| Is it an autonomous banking agent? | Inference is a prediction pipeline. The experiment workflow is automated; the system does not independently move money or cancel subscriptions. |
| Are these real clients? | The challenge histories are synthetic. Real-bank outcomes, savings and production reliability have not been validated. |

## Suggested single-slide evidence

Use one observed client example from [explanations.md](../explanations.md), with the explicit caption "selected correct example". For example, `C002783` has a displayed cloud candidate with eight payments, median interval 29.8 days and last payment 18.4 days before cutoff. Those numbers are from the committed evidence, not a newly inspected raw timeline. Do not invent intermediate transaction dates for a chart.

Show three elements: observed candidate evidence; the family output and competing/uncertain candidates if available; and **0.619 Macro-F1 — supplied UBS validation, 1,000 clients**. A small note can state "extra clock detail rejected; refund-context stress gain +0.033". Avoid making the score look like a calibrated individual probability or a hidden-test result.

The five jury criteria listed in this branch's [requirements.md](../requirements.md) can be served without overclaiming: a working reproducible predictor for technical functionality, intelligible evidence for user experience, an honest automated research workflow for agentic depth, demonstrated learning from shift and failed features for originality, and a proposed reminder/review application for impact. This audit relies on the branch's recorded criteria review and does not claim to have freshly checked event scheduling or website changes.
