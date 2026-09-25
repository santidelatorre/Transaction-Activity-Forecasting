# Final leakage and completion audit

This audit covers the frozen prediction package at commit `ffc656e` and the
two completed raw-data runs. The score is **0.619493 macro-F1**; the 0.80 target
is unmet. No claim of an information-theoretic ceiling is made.

| Requirement | Inspected evidence and outcome |
|---|---|
| Official specification and jury | Complete official files and jury page reviewed; source revision and contract in `requirements.md`. |
| Pinned data | Seven raw file hashes in `data_manifest.json`; archive SHA-256 verified by `prepare_data.py`. The audit command refuses altered inputs. |
| Pre-cutoff history | All four transaction splits end before 2026-01-01; loader rejects later events; boundary test covers the cutoff. |
| Identity and duplicates | Audit found no cross-split client IDs or identical histories, no duplicate transaction rows; label joins are by ID with exact-set checks. |
| Target isolation | Final `cli.train` reads only train labels. Features accept histories, fixed templates and unlabeled price profiles. Official evaluation writes predictions before explicitly opening labels. |
| Fold isolation | Stratified client CV; all three original/corrupted views of training clients are grouped in training. Validation clients never enter fold fitting. |
| Auxiliary isolation | Historical weak targets derive only from the disjoint pretraining histories and a separate historical cutoff. Their scores are explicitly auxiliary and the expert was rejected. |
| All classes/clients | Fixed eight-label evaluator; all candidate rows per client; 2,000 train / 1,000 official validation clients; no dropped difficult classes or clients. |
| Suspicious signals | IDs/ordering excluded, ID-quartile relationship audited. Identifier renaming and input shuffling are tested for core deterministic transforms. |
| Model selection | Exploratory OOF/stress estimates labeled as such. Class biases fitted on OOF were rejected after the first holdout batch. Final blend and all seeds frozen before the second batch. |
| Holdout exposure | Two selection batches plus one reproduction access, all logged. The official set is an external check, not an untouched final test after this research. |
| Test information | Test histories used only for feature-only drift analysis and final prediction; no test labels available. Corruption assumptions were informed by feature shift. |
| Reproduction | Both runs recomputed raw JSONL features, relearned unlabeled profiles and refit all estimators. All 1,000 validation predictions and probabilities match exactly; `reproduction.json` records this. |
| Source provenance | Prediction package unchanged from `ffc656e`. Full fingerprints differ because reporting/research scripts were added between runs; this is disclosed instead of claiming identical whole-repository source. |
| Submission | Exact 1,000 sample IDs; correct ordered columns; zero duplicates/missing values; legal labels; checked again after serialization and committed-copy creation. SHA-256 in `submissions/validation.json`. |
| Metrics | Every completed experiment has all-class precision/recall/F1, confusion matrix, true/predicted frequencies and timing. Final report is generated from actual predictions. |
| Tests | 16 passing checks cover contracts, metrics, label/alignment, cutoff, normalization, amount clustering, augmentation and resumed/interrupted stage orchestration. |
| Product claims | Evidence examples use observed features. No claim of validated date/amount forecasting, real-bank deployment, fraud detection or autonomous money movement. |

The scientific limitation is generalization under stronger hidden-test shift,
not an unreported change in evaluation. Bootstrap uncertainty conditions on
the chosen model and validation sample; it does not remove selection bias.

The completed search covers text, tabular models, explicit recurrence rules,
pooled family learning, ranking, hierarchical none detection, corruption
training, unlabeled price profiles, historical auxiliary supervision, feature
ablations, three seeds, decision optimization and diverse ensembles. Expensive
sequence/embedding models were not trained; their lower priority and remaining
hypotheses are documented in `research_decisions.md`. The honest fallback is
the strongest reproduced solution and its full evidence, not a fabricated
0.80 result.
