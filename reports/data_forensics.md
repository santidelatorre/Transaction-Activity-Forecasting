# Initial forensic findings

Generated input evidence is in `reports/data_manifest.json` and
`outputs/audit/`; reproduce with `python scripts/audit_data.py`.
This report uses training labels only. Official validation labels are sealed.

## Verified structure

| Split | Clients | Transactions | Mean transactions/client |
|---|---:|---:|---:|
| Train | 2,000 | 147,459 | 73.73 |
| Official validation | 1,000 | 73,898 | 73.90 |
| Test | 1,000 | 75,761 | 75.76 |
| Unlabeled pretraining | 10,000 | 749,935 | 74.99 |

No missing transaction fields, exact duplicate transactions, identical full
client histories, overlapping client IDs, or identical histories across splits
were found. Every event precedes the 2026-01-01 cutoff. Observations start on
2024-11-07; typical observed history spans about 405 days. Currencies are CHF,
EUR, USD and GBP. Identifiers are excluded from modeling.

Training targets: none 597; streaming 225; insurance 214; music 198; software
195; mobile 191; cloud 190; gym 190. Accuracy is misleading here: the
majority-only baseline is 29.85% accurate but has macro-F1 0.05747.

## Text and MCC structure

The 1,475 training descriptions are mutations of a small, interpretable
vocabulary. Common changes include pay/billing/member prefixes, online/core/
digital/service suffixes, abbreviation and truncation. Digit removal alone is
not the key normalization challenge. Only about 0.83% of validation and 0.68%
of test transaction descriptions are absent from training.

MCC generally carries family semantics (7997 gym, 6300 insurance, 4814 mobile,
5734 software), but is sometimes corrupted. Cloud shares 5732 with electronics;
music and streaming share 5812 with restaurants. Text is also intentionally
ambiguous: premium plan and digital plus appear in several families. Some
background transactions carry misleading subscription descriptions and vice
versa. Joint evidence is necessary.

Family-specific template diversity correlates with the target: the target's
mean number of distinct known templates is near 3, while unrelated families
usually average below 1. But a fixed maximum-diversity rule scores only about
0.457 macro-F1 and exact-template count trees about 0.537. This correlation is
useful evidence, not a reconstructed generator or sufficient classifier.

## Recurrence hypotheses

Amount-neighborhood clustering (relative log-amount radius 0.035, minimum three
events) finds 6,590 candidate clusters in 1,940 training clients. These are
not verified ground-truth streams. Rounded median intervals concentrate around
28–31 days; a smaller peak occurs near 14 days. Candidate family inference
recovers a cluster matching the target for 1,166 of 1,403 non-none clients;
this is a diagnostic candidate-coverage count, **not classification accuracy**.

Matching-family clusters have longer spans and more observations on average,
but extensive overlap remains. Amount-only clustering can capture unrelated
background transactions with similar prices, corrupting cadence. The next
experiments therefore compare filtered and unfiltered streams, broad family
grouping, calendar phase features, and learned family ranking. We do not yet
claim an exact reconstruction of the label-generating process.

Some none clients still have apparently regular family payments near cutoff.
Therefore an active-looking stream alone cannot safely rule out none. A
dedicated none detector is being compared with direct classification.
