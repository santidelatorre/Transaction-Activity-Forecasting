# V2 Integration Results

Official eight-class validation; V1 reference 0.2710242658492452. Rejected trials are never enabled by default. delta_previous compares with the named stable/control score passed for that trial.

| step | source_branch | source_commit | change | macro_f1 | delta_previous | delta_v1 | accuracy | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | baseline | 0199a8b | baseline | 0.271024266 | 0.000000000 | 0.000000000 | 0.266000000 | BASELINE | Frozen V1 none_bias=-1; temperature=1. |
| 1 | origin/features/jaime-v2-experiment-tracking | fe85154+cd19c44 | tracking | 0.271024266 | 0.000000000 | 0.000000000 | 0.266000000 | KEEP_TOOLING | Neutral tooling; 39 tests passed; fresh frozen V1 scoring. |
| 2 | origin/features/laura-v2-integracion | 614f769 | quality_gate | 0.271024266 | 0.000000000 | 0.000000000 | 0.266000000 | KEEP_TOOLING | Neutral quality checks; 58 tests passed; fresh frozen V1 scoring. |
| 3 | origin/features/ginestar-v2-temporal | 6ba1311 | temporal_intervals | 0.267817007 | -0.003207259 | -0.003207259 | 0.262000000 | REJECT | Only intervals; frozen V1 bias; disabled by default pending stability. |
| 4 | origin/features/ginestar-v2-temporal | 6ba1311 | temporal_periodicity | 0.272636283 | 0.001612017 | 0.001612017 | 0.265000000 | KEEP_PROVISIONAL | Only periodicity; same V1 bias and mapping; independent of intervals trial. |
| 5 | origin/features/ginestar-v2-temporal | 6ba1311 | temporal_activity | 0.264064548 | -0.008571735 | -0.006959718 | 0.261000000 | REJECT | Isolated activity vs V1; compared to provisional periodicity; not stacked. |
| 6 | origin/features/ginestar-v2-temporal | 6ba1311 | temporal_horizon | 0.261599760 | -0.011036523 | -0.009424506 | 0.258000000 | REJECT | Isolated horizon vs V1; compared to provisional periodicity; not stacked. |

Full per-class F1, confusion matrices, predictions and provenance: `outputs/metrics/v2_integration/results.json` (local, ignored).
