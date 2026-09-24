# Zurich AI Weeks hackathon plan

## Current decisions — 24 September 2026

The [official contract](OFFICIAL_CHALLENGE.md) takes precedence over the
provisional plan below: eight-class client prediction, macro-F1 and a two-column
submission. Date/amount/confidence are optional demo outputs, not requirements
of the scored CSV. Keep the supplied train/valid/test split.

Carles reports the team agreed one branch per person (`dev/<name>`) with PRs
to main. Workstream branch names below are historical suggestions. Assign people
after a short team discussion; no automatic assignment is implied by this table.
Prioritize a valid official submission before expanding the dashboard.

## Goal and priority

Deliver a working path early:

```text
canonical data -> recurrence signal -> next-activity prediction
-> temporal evaluation -> business-facing output/demo
```

The model choice is an internal system decision. Executives should see the
next likely activity, confidence, expected date, detected recurrence, and a
plain-language explanation. Technical model names and metrics are secondary.

## Seven parallel workstreams

| Person | Scope | Branch | First usable output |
| --- | --- | --- | --- |
| 1 | Data / schema / EDA | `feat/data` | Raw-to-canonical loader and data-quality notes |
| 2 | Entity resolution | `feat/entity-resolution` | Stable `merchant_normalized` mapping |
| 3 | Recurrence detection | `feat/recurrence-detection` | Interval, stability, recurrence score, expected date |
| 4 | Candidate generation + features | `feat/candidate-features` | Candidate table and feature matrix |
| 5 | Statistical baseline | `feat/statistical-baseline` | First end-to-end prediction and baseline score |
| 6 | ML / model comparison | `feat/ml-ranker` | Leakage-safe tabular model compared with baseline |
| 7 | Evaluation / integration / dashboard | `feat/evaluation-pipeline` | Temporal evaluation, selection, reproducible demo |

If dashboard work becomes substantial, use `feat/dashboard` in addition to
the integration branch. Everyone can start against `mock_transactions()` and
the contracts in `docs/DATA_CONTRACT.md`; no workstream waits for the final
dataset adapter.

### Detailed responsibilities

1. **Data / schema / EDA**: inspect columns, types, dates, missingness,
   duplicates, customer and transaction distributions, and publish the loader.
2. **Entity resolution**: normalize merchant/counterparty strings, assess IDs,
   and document matching decisions for recurrence and features.
3. **Recurrence**: start with transparent rules, then add tolerances for
   weekly/monthly patterns, interval stability, frequency, and recurrence score.
4. **Candidates + features**: build recency, frequency, interval mean/std,
   amount statistics, entity, periodicity, and temporal features.
5. **Baseline**: keep the simple rules executable and integrated even if later
   models are better; this is the demo fallback.
6. **ML**: train a tabular classifier/ranker only after the baseline works;
   check leakage and probability quality before considering complexity.
7. **Evaluation / integration / dashboard**: own temporal validation, official
   metric wiring, model comparison, output compatibility, and the demo path.

## Milestones and checkpoints

### Phase 0 - start

Clone/pull, create the environment, run checks, inspect the challenge data,
confirm the official metric and submission format. Result: everyone can run the
repository.

### Phase 1 - contracts

Confirm the source schema, target, temporal validation design, and mock
interfaces. Result: parallel work can begin independently.

### Phase 2 - first end-to-end

Run canonical data (mock if necessary) through recurrence, simple prediction,
evaluation, and a demo output. This is the minimum deliverable.

### Phase 3 - parallel build

P1 improves data, P2 entities, P3 recurrence, P4 features, P5 baseline, P6 ML,
and P7 evaluation/dashboard/integration. Merge small PRs continuously.

### Checkpoint 1 - mid-morning

Ask: does the pipeline run, is the metric visible, is there leakage, which mocks
were replaced, and is anyone blocked? Reassign blocked work immediately.

### Phase 4 - first real integration

Replace mocks progressively with measured real components. Keep the previous
working path available and reject sophistication that does not improve results
or the demo.

### Checkpoint 2 - halfway

Freeze a stable version and record baseline score, best score, selected model,
contributing components, broken pieces, and worthwhile next experiments.

### Phase 5 - optimization; Phase 6 - demo

Only after stability: tune features/models, calibrate, improve explanations,
and consider hybrid/ensemble approaches. Connect real outputs to a simple demo
that tells the client story: history -> recurring streams -> next activity ->
date/confidence -> explanation -> optional quality metrics.

### Checkpoint 3 - before close

Stop risky experiments. Fix the final model, pipeline, dashboard, metrics,
README, demo, and submission artifacts.

## Git and integration rules

No functional work directly on `main`:

```bash
git checkout main
git pull origin main
git checkout -b feat/<workstream>
```

Use small commits and pull requests. Before a PR, run:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pre_commit run --all-files
```

Every integrated module needs tests, a documented interface, a minimal usage
example, compatible output, and no regression of the smoke pipeline. Do not
commit datasets, secrets, `.env`, large outputs, or heavyweight model files.

## Definition of done

- Canonical schema and source assumptions are documented.
- Baseline and at least one temporal evaluation path run reproducibly.
- Model comparison selects internally; the dashboard does not ask the user to
  choose CatBoost, regression, or Transformer.
- Prediction output includes date, confidence, entity, explanation, and model
  metadata.
- Checks pass, README is current, and the demo can run without notebooks.
