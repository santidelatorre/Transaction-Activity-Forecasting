# Architecture

Official-task update: see [UBS contract](OFFICIAL_CHALLENGE.md). The repository
includes a legacy mock smoke path, a generic configurable pipeline, and the
official UBS V1 adapter/features/models and training runner. UBS supervised
training uses the supplied client-level train/valid splits. Generic temporal
helpers are not a substitute for that contract.

The pipeline is intentionally modular and replaceable:

```text
source data
    -> data contract / preprocessing
    -> entity resolution
    -> recurrence signals
    -> candidates + features
    -> baseline / hybrid / ML models
    -> temporal evaluation + model selection
    -> business prediction output
    -> dashboard / demo
```

Current repository status:

- `data.contracts`: canonical validation and deterministic mock data.
- `evaluation.temporal`: two-way split with strict timestamp separation, plus
  the generic three-way row split and its optional row embargoes. The latter
  preserves its existing row-count semantics; it does not group timestamp ties.
- `models.baseline`: first runnable recurrence-rule predictor.
- `pipeline.run_smoke_pipeline`: small legacy development check.
- `pipeline.run_frame`: configurable generic regression workflow.
- `ubs.data/features/models`: official loading, client features and classifiers.
- `scripts/run_ubs_baseline.py`: UBS training, model comparison and submission.
- `evaluation.official`: shared eight-class metric/CSV-validation core and CLI.
- `ubs.evaluation`: compatibility wrapper for the runner's existing metric keys.
- `ubs.data.validate_submission`: shared checks plus strict sample order and
  test-client coverage; the None return contract is preserved.

The source dataset adapter is the only component that should know raw column
names. The dashboard only sees the prediction output contract. Model selection
is internal: compare candidates using the official metric and
return the best model plus secondary metrics.
