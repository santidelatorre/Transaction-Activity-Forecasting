# Data contract

## Official contract now available

The [UBS 2026 contract](OFFICIAL_CHALLENGE.md) supersedes the unresolved task
assumptions below. The table below remains the **legacy mock interface**, not
the official raw schema. Official files use `client_id`, `description`, `mcc`,
`direction`, etc.; they do not document `transaction_id` or `merchant`.
Do not treat a raw description as a known merchant family.

Official prediction output is exactly `client_id,predicted_next_recurring_merchant`.
Date, amount and explanations are optional demo metadata, kept outside that CSV.
The existing `confidence` field is heuristic and is not a calibrated probability.
The official adapter is `ubs.data`, with features/models under `ubs/` and the
training entrypoint `scripts/run_ubs_baseline.py`. `evaluation.official` provides
the shared metric and submission-check core and a CSV CLI. `ubs.evaluation`
preserves the runner's result keys (`f1-score`, matrix as a list), while
`ubs.data.validate_submission` additionally requires exact sample row order
and checks the test client set. It returns None; the CLI helper returns an
aligned copy. Neither silently drops clients.

## Legacy smoke interface

This is the boundary between source-specific data work and the rest of the
pipeline. A loader or adapter must return a pandas `DataFrame` with the
canonical columns below before calling recurrence, features, models, or
evaluation.

## Canonical table

| Column | Status in this repository | Meaning |
| --- | --- | --- |
| `customer_id` | MOCK/ASSUMED | Stable customer identifier. |
| `transaction_id` | MOCK/ASSUMED | Stable transaction identifier. |
| `timestamp` | MOCK/ASSUMED | UTC-aware transaction timestamp. |
| `amount` | MOCK/ASSUMED | Numeric transaction amount; sign convention remains to be confirmed. |
| `currency` | MOCK/ASSUMED | Currency code as supplied by the source. |
| `merchant` | MOCK/ASSUMED | Raw merchant/counterparty label. |
| `merchant_normalized` | OPTIONAL | Entity-resolution output. |
| `category` | OPTIONAL | Source category, if available. |
| `account_id` | OPTIONAL | Account identifier, if allowed and useful. |
| `debit_credit` | OPTIONAL | Direction, if available. |

These mock fields are not the official data dictionary. Use the official
contract linked above and the UBS adapter for challenge files. In particular,
do not require the mock `transaction_id` or `merchant` in official JSONL files.

`validate_transactions()` is the shared adapter boundary. It checks required
columns, parses timestamps as UTC, coerces amounts to numeric, rejects nulls in
required fields, and returns stable chronological ordering. Extra columns are
preserved.

## Output contract for prediction consumers

Model implementations should expose a table or records containing:

`customer_id`, `entity`, `expected_date`, `expected_amount`, `confidence`,
`explanation`, `model_name`, and `model_version`.

The dashboard consumes this business output and must not depend on model internals.
