# Frozen-model error analysis

This analysis was produced after final model selection and did not change the predictor. All cohorts retain their difficult clients. They are descriptive, overlapping subsets, not separate validation claims.

There are 353 errors: 95 true recurring families predicted none, 61 true none clients predicted a family, and 197 wrong-family predictions.

Ten-bin top-label calibration error is 0.0514; log loss is 1.0806. Normalized rank scores should not be described as guaranteed calibrated probabilities.

| Cohort | Clients | Accuracy | Errors |
|---|---:|---:|---:|
| all | 1000 | 0.6470 | 353 |
| low_history_under_40_events | 63 | 0.7302 | 17 |
| high_activity_over_100_events | 139 | 0.6043 | 55 |
| true_none | 293 | 0.7918 | 61 |
| true_recurring_family | 707 | 0.5870 | 292 |
| family_without_3_event_amount_candidate | 140 | 0.1214 | 123 |
| family_regular_candidate | 348 | 0.7443 | 89 |
| family_irregular_candidate | 219 | 0.6347 | 80 |
| family_last_candidate_event_within_45_days | 542 | 0.7251 | 149 |
| family_last_candidate_event_over_90_days | 13 | 0.1538 | 11 |

The none boundary is a major source of error. Suppressing none would trade missed recurring families against false alerts; the earlier OOF bias experiment failed to improve external validation. The frozen model therefore retains its unadjusted decision rule. Wrong-family errors remain spread across several pairs, rather than one vocabulary confusion explaining the entire gap.

Amount groups are approximate candidates: background payments with similar amounts can distort cadence, while masking/MCC corruption can obscure family identity. These are plausible failure mechanisms supported by the feature-only shift audit and train-side ablations, not proven causal explanations for every wrong client.

The most frequent confusion pairs are in `final_confusions.csv`. The next research priorities and rejected alternatives are in `research_decisions.md`. No post-hoc fixes were selected from these official-validation errors.
