# Data contract

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

The source dataset, official target, permitted identifiers, amount semantics,
timezone, and official metric are **not confirmed yet**. Do not treat the mock
fields as facts about the challenge data. Update this document when the data
dictionary arrives.

`validate_transactions()` is the shared adapter boundary. It checks required
columns, parses timestamps as UTC, coerces amounts to numeric, rejects nulls in
required fields, and returns stable chronological ordering. Extra columns are
preserved.

## Output contract for prediction consumers

Model implementations should expose a table or records containing:

`customer_id`, `entity`, `expected_date`, `expected_amount`, `confidence`,
`explanation`, `model_name`, and `model_version`.

The dashboard consumes this business output and must not depend on model internals.
