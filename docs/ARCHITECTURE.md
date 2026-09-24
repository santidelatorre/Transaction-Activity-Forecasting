# Architecture

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
- `evaluation.temporal`: chronological split that keeps test rows after train.
- `models.baseline`: first runnable recurrence-rule predictor.
- `pipeline.run_smoke_pipeline`: end-to-end path for development before real data.
- Empty package boundaries already exist for preprocessing, recurrence, features,
  visualization, and utilities. Team members should add implementation behind
  these boundaries rather than moving unrelated code across packages.

The source dataset adapter is the only component that should know raw column
names. The dashboard only sees the prediction output contract. Model selection
is internal: compare candidates using the official metric when available and
return the best model plus secondary metrics.
