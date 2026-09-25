# Jury brief

Source: [official Jury Process & Evaluation Criteria](https://zh.ai-weeks.ch/jury-process-and-criteria).
The criteria are qualitative, with no published numerical weights. The
technical result to present is the frozen score in `final_results.md`; the
0.80 research target was not achieved.

## One-minute pitch draft

“A transaction history is a mixture of processes: groceries, transfers,
subscriptions and occasional refunds. We reconstruct candidate recurring
streams, compare their timing and family evidence, and estimate the next
recurring merchant family—including none.

The major discovery was deployment shift. A model that looked good in ordinary
cross-validation collapsed on the official holdout because descriptions and
merchant codes became much noisier. Our research workflow detected the shift,
tested explicit corruption scenarios and replaced the fragile representation.
Refund context then produced a reproducible improvement.

The result is an auditable prediction file, a repeatable training pipeline and
actual payment evidence behind each prediction. The proposed banking use is
to surface possible upcoming payment families for reminders and subscription
reviews. We report the measured accuracy and its limits rather than claiming
perfect foresight.”

Measured result: **0.619493 macro-F1**, **0.647 accuracy** on 1,000 official
validation clients. Two independent raw-data training runs produced identical
probabilities. The 0.80 objective was not achieved. Full evidence is in
`final_results.md`.

## Evidence mapped to the actual criteria

| Criterion | Demonstrable evidence |
|---|---|
| Technical Functionality & AI | Raw-data train/evaluate/submit CLI; client-safe CV; drift audit; recurring-stream inference; learned family ranking and none detection; complete per-class results; independent rebuild. |
| User Experience | One prediction per client plus interpretable observed counts, intervals, recency and refunds. Examples are in `explanations.md`; no invented exact-date guarantee. |
| Agentic Depth | The work used an autonomous hypothesis → experiment → diagnostic → decision loop. The repository records executed evidence and has a deterministic final selection gate. Inference itself is a prediction pipeline; it is not presented as an autonomous banking-action agent. |
| Originality & Fun Factor | Recognizing that a superficially successful model failed under synthetic masking; combining recurring processes and refund context instead of merely increasing model size. |
| Potential & Market Impact | Candidate payment reminders, subscription insights and planning support. Real deployment requires separate real-data validation and a useful uncertainty-aware product experience. |

## Claims to avoid

Do not say the system reached 0.80, knows exact next dates or amounts, detects
fraud, transfers money, or has been validated on real UBS clients. It predicts
one of eight merchant-family labels on synthetic histories. Observed evidence
is not a complete causal explanation of a tree ensemble.
