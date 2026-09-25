# UBS recurring merchant research

Fresh implementation of the [official 2026 Transaction Activity Forecasting
challenge](https://github.com/UBS-AG/Swiss-AI-Weeks/blob/main/hackathons/2026/challenge.md).
Predict cloud, gym, insurance, mobile, music, software, streaming or none from
history before 2026-01-01, for the next 90 days.

**Research is in progress. The 0.80 macro-F1 objective has not been achieved.**
Ordinary train-side OOF reached 0.6851, but a feature-only audit found severe
description/MCC shift in official validation and test. That number must not be
treated as a reliable hidden-test estimate. Masking augmentation improves
50%-masking stress OOF from 0.1283 to 0.6111. Official validation labels have
not yet been used. See the executed [experiment log](reports/experiment_log.md),
[full metrics](reports/experiment_results.jsonl), and
[forensics report](reports/data_forensics.md).

## Reproduction during research

Python >=3.11. Install with `python -m pip install -e ".[dev]"`.
Input archive and file hashes are pinned in `reports/data_manifest.json`.
The preparation command downloads exactly the reviewed official revision.

```powershell
python scripts/prepare_data.py
python scripts/audit_data.py
python scripts/baseline_ladder.py
python scripts/template_experiments.py --subset counts
python scripts/stream_experiments.py --model pooled --filter-background
python scripts/ranking_experiments.py --model ranker --no-clocks
python scripts/audit_shift.py
python scripts/build_masked_features.py --rate .3
python scripts/build_masked_features.py --rate .5
python scripts/robust_experiments.py --augment
python -m pytest -q
```

Experiment IDs are immutable: rerunning an already recorded ID currently
fails rather than overwriting its evidence. A final reproduction/submission
CLI will be added after model selection and a frozen holdout evaluation.

All train CV splits are stratified by client. Transactions and augmented
copies of a client never cross folds. IDs, file order and labels are excluded
from features. Every completed experiment writes class precision/recall/F1,
confusion matrix, true/predicted frequencies, calibration, seed and timing.
Raw data, feature caches, model artifacts and predictions are ignored by Git.

The approach combines interpretable family evidence, amount-based recurring
stream discovery and learned ranking of eight family candidates. Product
explanations and the jury narrative will be based on actual evidence from the
frozen model, after predictive and reproducibility checks.
