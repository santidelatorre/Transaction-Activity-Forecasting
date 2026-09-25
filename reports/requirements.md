# Requirements and source review

Reviewed on 2026-09-25 from the official repository at commit
`796d5805ec5f8a228a3ec0de36a2b4e6e1b1a1df`:
root README, the complete `hackathons/2026/challenge.md`, repository file
inventory, Apache-2.0 license, and the seven-file dataset archive. There are
no additional data READMEs or supplied evaluation scripts in that revision.

Sources:
- https://github.com/UBS-AG/Swiss-AI-Weeks/blob/main/hackathons/2026/challenge.md
- https://zh.ai-weeks.ch/jury-process-and-criteria (complete page retrieved locally)

## Prediction contract

One prediction per client from transaction history preceding 2026-01-01.
Predict the next recurring merchant family during the following 90 days.
Classes: cloud, gym, insurance, mobile, music, software, streaming, none.
Optimize arithmetic mean of the eight class F1 values; also report accuracy,
class precision/recall/F1/support, confusion matrix and prediction frequencies.
The specification does not disclose the exact synthetic generator, handling
of tied next events, or hidden label construction; these must not be assumed.

Training labels: `target_next_recurring_merchant`. Submission columns, in
sample order: `client_id,predicted_next_recurring_merchant`. IDs must exactly
match the sample, once each, with no missing or illegal predictions.

## Scientific protocol

Start with train-only forensic analysis. Use fixed stratified client-level
five-fold CV and true OOF predictions. All supervised fitting, text vocabulary
fitting, scaling, calibration and learned template induction stay inside the
appropriate training folds. Tune decision rules on inner OOF predictions and
evaluate on untouched outer folds, or label exploratory OOF tuning as such.
Preserve official validation labels for a small number of frozen finalist
checks. Record every access and finalist rationale. Never use test labels,
identifiers, row order, or post-cutoff events as predictive features.
Investigate duplicate clients and feature-only distribution shift. Reproduce
final metrics and predictions from raw data, with immutable input hashes.

## Jury interpretation

Judging is qualitative and discretionary, not a weighted quantitative rubric.
The five discussion criteria are Technical Functionality & AI; User
Experience; Agentic Depth; Originality & Fun Factor; Potential & Market Impact.
Preferences are clean reproducible code, modern AI concepts/execution, and
addressing the selected challenge's objectives. Build a working predictor
first, then evidence-based explanations and a credible banking use case.
Do not invent agentic behavior or imply exact dates/amounts are validated.

The page specifies expert jury on Friday September 25, 14:00–16:00, submission
by 12:00, with 1-minute pitch plus 3-minute Q&A. Main jury is 18:00–18:45,
submission by 17:30, with 2-minute pitch plus 1-minute Q&A. Team eligibility,
registration and external submission are separate from generating a valid CSV.

## Research sequence

Source review → verified acquisition → forensic audit → validation framework →
baseline ladder → evidence-driven recurrence/family/tabular/text experiments →
error analysis, decision optimization and ensembles → frozen holdout check →
clean reproduction, tests, submission contract validation, documentation.
