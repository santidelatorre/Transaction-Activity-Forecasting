# Final results

**Official validation macro-F1: 0.619493; accuracy: 0.647000.** The 0.80 objective was not reached.

Client-bootstrap 95% interval for macro-F1: [0.5859, 0.6500]. This interval does not account for all model-selection uncertainty or hidden-test shift.

Two independent raw-data training runs produced identical class predictions. Maximum probability difference: 0. All 1,000 official validation clients were retained; none was used for supervised fitting.

| Class | Precision | Recall | F1 | True n | Predicted n |
|---|---:|---:|---:|---:|---:|
| cloud | 0.6354 | 0.6854 | 0.6595 | 89 | 96 |
| gym | 0.6134 | 0.6033 | 0.6083 | 121 | 119 |
| insurance | 0.6512 | 0.5657 | 0.6054 | 99 | 86 |
| mobile | 0.6701 | 0.6250 | 0.6468 | 104 | 97 |
| music | 0.5930 | 0.5484 | 0.5698 | 93 | 86 |
| software | 0.5631 | 0.5577 | 0.5604 | 104 | 103 |
| streaming | 0.5930 | 0.5258 | 0.5574 | 97 | 86 |
| none | 0.7095 | 0.7918 | 0.7484 | 293 | 327 |

95 executed evaluation records are preserved, including clearly marked auxiliary and decision-fit diagnostics. No auxiliary-task result counts toward the challenge target. See [experiment log](experiment_log.md), [reproduction evidence](reproduction.json), [error cohorts](final_error_cohorts.csv), [confusion counts](final_confusions.csv), and [observed explanations](explanations.md).

The committed submission contains exactly the required test IDs and legal labels; its contract was checked before and after writing the CSV. No hidden-test score is available.
