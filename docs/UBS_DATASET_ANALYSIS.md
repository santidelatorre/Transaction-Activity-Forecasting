# UBS Swiss AI Weeks 2026 dataset analysis

Generated with `python scripts/analyze_ubs_dataset.py --data-dir data/raw/ubs_2026`.
All source data remains local and ignored by Git. The challenge is a client-level multiclass prediction: given each client's history before `2026-01-01`, predict the merchant family expected to recur during the following 90 days.

## Dataset size and cutoff checks

| dataset | transactions | clients | transactions/client mean | transactions/client median | transactions/client p95 | file MB | memory MB | first timestamp | last timestamp | after cutoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 147459 | 2000 | 73.7295 | 72.0 | 116.04999999999995 | 28.0847 | 43.9156 | 2024-11-07T00:07:06+00:00 | 2025-12-31T23:57:50+00:00 | 0 |
| validation | 73898 | 1000 | 73.8980 | 73.0 | 114.0 | 14.1031 | 22.0362 | 2024-11-07T00:09:28+00:00 | 2025-12-31T23:55:23+00:00 | 0 |
| test | 75761 | 1000 | 75.7610 | 74.0 | 118.0 | 14.4662 | 22.5996 | 2024-11-07T00:33:43+00:00 | 2025-12-31T23:58:30+00:00 | 0 |
| unlabeled pretrain | 749935 | 10000 | 74.9935 | not calculated (streamed) | not calculated (streamed) | 142.7669 | 34.0650 | 2024-11-07T00:06:22Z | 2025-12-31T23:57:56Z | 0 |

### Client-set overlap

| sets | shared clients |
| --- | --- |
| train / validation | 0 |
| train / test | 0 |
| validation / test | 0 |

All `after cutoff` counts must be zero. Any nonzero value invalidates a leakage-safe evaluation.

## Labels and imbalance

| label | train count | train % | validation count | validation % | absolute difference pp |
| --- | --- | --- | --- | --- | --- |
| cloud | 190 | 9.5000 | 89 | 8.9000 | 0.6000 |
| gym | 190 | 9.5000 | 121 | 12.1000 | 2.6000 |
| insurance | 214 | 10.7000 | 99 | 9.9000 | 0.8000 |
| mobile | 191 | 9.5500 | 104 | 10.4000 | 0.8500 |
| music | 198 | 9.9000 | 93 | 9.3000 | 0.6000 |
| software | 195 | 9.7500 | 104 | 10.4000 | 0.6500 |
| streaming | 225 | 11.2500 | 97 | 9.7000 | 1.5500 |
| none | 597 | 29.8500 | 293 | 29.3000 | 0.5500 |

### Label and submission contract checks

| set | label rows | unique clients | duplicate label client IDs | missing target | unknown targets | cutoff values | labels absent from transactions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2000 | 2000 | 0 | 0 | 0 | 2026-01-01 | 0 |
| validation | 1000 | 1000 | 0 | 0 | 0 | 2026-01-01 | 0 |

| sample submission rows | unique submission clients | test clients | submission/test ID symmetric difference |
| --- | --- | --- | --- |
| 1000 | 1000 | 1000 | 0 |

Macro-F1 weights every class equally. Therefore the smallest classes need explicit validation and prediction coverage; an accuracy-oriented majority-class strategy is unsuitable.

## Transaction schema (train)

| column | dtype | missing | missing % | unique | top values | min | p01 | median | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amount | float64 | 0 | 0.0000 | 49768 | 11.22 (37); 9.86 (36); 12.76 (35) | 1.0000 | 3.4100 | 73.0800 | 7983.1118 | 8999.9800 |
| client_id | object | 0 | 0.0000 | 2000 | C002168 (162); C001577 (158); C001787 (156) | nan | nan | nan | nan | nan |
| currency | object | 0 | 0.0000 | 4 | chf (73086); eur (38146); usd (21957) | nan | nan | nan | nan | nan |
| description | object | 0 | 0.0000 | 1475 | salary (12031); atm withdrawal (8083); fresh foods (6649) | nan | nan | nan | nan | nan |
| direction | object | 0 | 0.0000 | 2 | out (121598); in (25861) | nan | nan | nan | nan | nan |
| fee | float64 | 0 | 0.0000 | 342 | 0.0 (130262); 0.32 (83); 0.79 (74) | 0.0000 | 0.0000 | 0.0000 | 3.1400 | 3.5000 |
| mcc | int64 | 0 | 0.0000 | 12 | 6012 (28067); 5812 (25723); 5411 (21040) | 4111.0000 | 4111.0000 | 5812.0000 | 7997.0000 | 7997.0000 |
| timestamp | datetime64[ns, UTC] | 0 | 0.0000 | 147048 | 2025-04-28 20:59:56+00:00 (3); 2025-07-08 09:55:28+00:00 (3); 2025-07-11 20:42:44+00:00 (2) | nan | nan | nan | nan | nan |
| type | object | 0 | 0.0000 | 7 | card_payment (93464); topup (12031); p2p_transfer (12013) | nan | nan | nan | nan | nan |

## Requested transaction variables

| column | dtype | missing | missing % | unique | top values | min | p01 | median | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| amount | float64 | 0 | 0.0000 | 49768 | 11.22 (37); 9.86 (36); 12.76 (35) | 1.0000 | 3.4100 | 73.0800 | 7983.1118 | 8999.9800 |
| currency | object | 0 | 0.0000 | 4 | chf (73086); eur (38146); usd (21957) | nan | nan | nan | nan | nan |
| description | object | 0 | 0.0000 | 1475 | salary (12031); atm withdrawal (8083); fresh foods (6649) | nan | nan | nan | nan | nan |
| direction | object | 0 | 0.0000 | 2 | out (121598); in (25861) | nan | nan | nan | nan | nan |
| mcc | int64 | 0 | 0.0000 | 12 | 6012 (28067); 5812 (25723); 5411 (21040) | 4111.0000 | 4111.0000 | 5812.0000 | 7997.0000 | 7997.0000 |
| timestamp | datetime64[ns, UTC] | 0 | 0.0000 | 147048 | 2025-04-28 20:59:56+00:00 (3); 2025-07-08 09:55:28+00:00 (3); 2025-07-11 20:42:44+00:00 (2) | nan | nan | nan | nan | nan |
| type | object | 0 | 0.0000 | 7 | card_payment (93464); topup (12031); p2p_transfer (12013) | nan | nan | nan | nan | nan |

### Train anomalies

| check | count |
| --- | --- |
| exact duplicate transaction rows | 0 |
| duplicate client/timestamp pairs | 0 |
| amount <= 0 | 0 |
| fee < 0 | 0 |
| fee > 0 | 17197 |
| timestamps at/after cutoff | 0 |

`amount` includes its original currency unit. Do not compare or aggregate monetary magnitude across currencies without an explicit conversion decision.

## Time coverage and client history

| metric | value |
| --- | --- |
| history days mean | 404.5912 |
| history days median | 408.5084 |
| history days p95 | 417.5916 |
| transactions / 30d mean | 5.4444 |
| last event within 30d of cutoff % | 99.5000 |

The dataset summary above also reports first/last transaction timestamps per partition and the pre-cutoff check.

## Recurrence in histories

### Train

| repeated description groups | clients with repeat descriptions | share of clients with repeats % | median repeat appearances | regular groups (std <= 3d) % | same amount groups % | amount CV <= 5% groups % |
| --- | --- | --- | --- | --- | --- | --- |
| 33051 | 2000 | 100.0000 | 3.0000 | 30.6859 | 0.1694 | 25.2186 |

| periodicity | groups |
| --- | --- |
| quarterly-ish | 13819 |
| monthly-ish | 9338 |
| annual-ish | 7802 |
| biweekly-ish | 1286 |
| weekly-ish | 796 |
| other | 10 |

### Validation

| repeated description groups | clients with repeat descriptions | share of clients with repeats % | median repeat appearances | regular groups (std <= 3d) % | same amount groups % | amount CV <= 5% groups % |
| --- | --- | --- | --- | --- | --- | --- |
| 16489 | 1000 | 100.0000 | 3.0000 | 39.5840 | 0.1334 | 15.6347 |

| periodicity | groups |
| --- | --- |
| quarterly-ish | 6769 |
| annual-ish | 5034 |
| monthly-ish | 3714 |
| biweekly-ish | 554 |
| weekly-ish | 414 |
| other | 4 |

Repeated `(client_id, description)` groups provide the most direct candidate streams. Useful per-stream measurements are appearance count, days since last event, median interval, interval standard deviation, amount stability, direction, MCC and type. The report deliberately treats the interval bins as descriptive approximations rather than hard rules.

## Text and possible direct signals

### Most frequent train description terms

| term | count |
| --- | --- |
| shop | 13187 |
| salary | 12031 |
| plan | 10922 |
| service | 8880 |
| atm | 8083 |
| withdrawal | 8083 |
| online | 7787 |
| foods | 6751 |
| fresh | 6746 |
| booking | 6657 |
| hotel | 6653 |
| pharmacy | 6647 |
| electronics | 6624 |
| ride | 6583 |
| share | 6583 |
| coffee | 6574 |
| grocery | 6564 |
| store | 6563 |
| neighborhood | 6556 |
| market | 6549 |
| marketplace | 6529 |
| casual | 6496 |
| dining | 6495 |
| digital | 6094 |
| send | 6017 |
| receive | 5996 |
| plus | 5391 |
| premium | 5073 |
| monthly | 4539 |
| access | 4368 |

### Literal target-family words observed in train descriptions

| label term in description | transactions |
| --- | --- |
| streaming | 3119 |
| cloud | 2986 |
| gym | 2905 |
| software | 1351 |
| insurance | 1332 |

### Per-label historical associations in train

| label | clients | clients with literal label in description % | top MCC | top descriptions |
| --- | --- | --- | --- | --- |
| cloud | 190 | 91.0526 | 5732, 6012, 5812 | salary; atm withdrawal; fresh foods |
| gym | 190 | 84.2105 | 6012, 5812, 5411 | salary; atm withdrawal; fresh foods |
| insurance | 214 | 79.4393 | 6012, 5812, 5411 | salary; atm withdrawal; ride share |
| mobile | 191 | 0.0000 | 6012, 5812, 5411 | salary; atm withdrawal; hotel booking |
| music | 198 | 0.0000 | 5812, 6012, 5411 | salary; atm withdrawal; pharmacy |
| software | 195 | 74.8718 | 6012, 5812, 5411 | salary; atm withdrawal; electronics shop |
| streaming | 225 | 83.1111 | 5812, 6012, 5411 | salary; atm withdrawal; coffee shop |
| none | 597 | 0.0000 | 6012, 5812, 5411 | salary; atm withdrawal; online marketplace |

### Descriptions with high target purity in train (at least 10 clients)

| description | clients | dominant label | dominant label clients | purity % |
| --- | --- | --- | --- | --- |
| member plan | 387 | none | 282 | 72.8682 |
| digital service | 419 | none | 297 | 70.8831 |
| subscription charge | 380 | none | 266 | 70.0000 |
| video | 20 | none | 13 | 65.0000 |
| prem plan online | 11 | none | 7 | 63.6364 |
| premium online | 11 | none | 7 | 63.6364 |
| prem plan core | 12 | none | 7 | 58.3333 |
| gym membership online | 21 | none | 12 | 57.1429 |
| coffee shop online | 16 | none | 9 | 56.2500 |
| cover plan plus | 20 | none | 11 | 55.0000 |
| fit | 22 | none | 12 | 54.5455 |
| electronics shop online | 13 | none | 7 | 53.8462 |
| membership | 15 | none | 8 | 53.3333 |
| ride share core | 17 | none | 9 | 52.9412 |
| safe cover digital | 22 | none | 11 | 50.0000 |
| online marketplace core | 16 | none | 8 | 50.0000 |
| fitness monthly digital | 12 | none | 6 | 50.0000 |
| grocery store digital | 12 | none | 6 | 50.0000 |
| grocery store online | 12 | none | 6 | 50.0000 |
| member media streaming | 12 | none | 6 | 50.0000 |
| member safe cover | 12 | none | 6 | 50.0000 |
| billing gym membership | 10 | none | 5 | 50.0000 |
| member service plan | 10 | none | 5 | 50.0000 |
| gym membership core | 19 | gym | 9 | 47.3684 |
| ride share online | 15 | none | 7 | 46.6667 |

### MCC target purity in train (at least 30 clients)

| mcc | clients | dominant label | dominant label clients | purity % |
| --- | --- | --- | --- | --- |
| 5734 | 1127 | none | 411 | 36.4685 |
| 4814 | 882 | none | 291 | 32.9932 |
| 6012 | 2000 | none | 597 | 29.8500 |
| 5411 | 1994 | none | 595 | 29.8395 |
| 5732 | 1984 | none | 591 | 29.7883 |
| 5812 | 1990 | none | 590 | 29.6482 |
| 4111 | 1866 | none | 551 | 29.5284 |
| 6011 | 1910 | none | 562 | 29.4241 |
| 5912 | 1886 | none | 549 | 29.1092 |
| 7011 | 1870 | none | 536 | 28.6631 |
| 6300 | 705 | none | 202 | 28.6525 |
| 7997 | 700 | none | 181 | 25.8571 |

### Descriptions enriched by target in train

| target | description | target client coverage % | other-client coverage % | lift |
| --- | --- | --- | --- | --- |
| cloud | cloud access | 80.5263 | 15.9116 | 5.0355 |
| cloud | storage plan | 74.7368 | 16.7403 | 4.4439 |
| cloud | cloud backup | 75.2632 | 17.1823 | 4.3607 |
| cloud | service plan | 72.6316 | 17.5691 | 4.1163 |
| cloud | merchant charge | 15.7895 | 13.9779 | 1.1287 |
| gym | urban gym | 75.7895 | 15.9669 | 4.7234 |
| gym | fitness monthly | 75.2632 | 16.5746 | 4.5196 |
| gym | fit club | 75.7895 | 17.5691 | 4.2950 |
| gym | gym membership | 76.3158 | 18.6740 | 4.0703 |
| gym | service fee | 88.4211 | 82.7624 | 1.0683 |
| insurance | safe cover | 78.5047 | 15.7335 | 4.9645 |
| insurance | insurance monthly | 78.0374 | 16.1254 | 4.8157 |
| insurance | policy premium | 68.6916 | 15.1736 | 4.5040 |
| insurance | cover plan | 74.2991 | 16.5733 | 4.4622 |
| insurance | software access | 25.2336 | 22.3404 | 1.1289 |
| mobile | phone contract | 82.7225 | 17.4682 | 4.7143 |
| mobile | service bill | 78.0105 | 17.4129 | 4.4602 |
| mobile | monthly plan | 81.6754 | 33.4992 | 2.4339 |
| mobile | digital plus | 85.8639 | 49.6960 | 1.7263 |
| mobile | service payment | 16.2304 | 14.2067 | 1.1414 |
| music | member pass | 76.2626 | 16.5372 | 4.5899 |
| music | audio streaming | 74.7475 | 17.0921 | 4.3536 |
| music | premium plan | 81.8182 | 51.1099 | 1.5997 |
| music | digital plus | 76.7677 | 50.5549 | 1.5175 |
| music | casual dining | 97.4747 | 92.7858 | 1.0505 |
| software | saas billing | 75.3846 | 16.8975 | 4.4409 |
| software | productivity suite | 75.3846 | 17.5623 | 4.2738 |
| software | software access | 72.3077 | 17.2853 | 4.1649 |
| software | premium plan | 82.0513 | 51.1357 | 1.6034 |
| software | card purchase | 16.9231 | 13.2410 | 1.2760 |
| streaming | video access | 76.8889 | 17.4085 | 4.3972 |
| streaming | media streaming | 76.4444 | 18.1408 | 4.1963 |
| streaming | digital plus | 81.7778 | 49.5211 | 1.6501 |
| streaming | premium plan | 83.1111 | 50.4789 | 1.6452 |
| streaming | digital order | 20.0000 | 15.6056 | 1.2798 |
| none | member plan | 47.2362 | 7.4840 | 6.2416 |
| none | digital service | 49.7487 | 8.6957 | 5.6674 |
| none | subscription charge | 44.5561 | 8.1254 | 5.4290 |
| none | monthly plan | 51.9263 | 32.2167 | 1.6099 |
| none | service payment | 17.0854 | 13.2573 | 1.2866 |

Literal label names, merchant-family-like descriptions, target-pure MCCs, or synthetic identifier sequences can produce shortcut signals. The high-purity table makes this concrete: a description can act as a near-direct family indicator. It is valid historical evidence if it remains predictive on the disjoint validation clients, but should be treated as a generator artifact risk rather than assumed to generalize.

## Distribution comparison including unlabeled pretrain

| dataset | column | top value | top share % | unique |
| --- | --- | --- | --- | --- |
| train | currency | chf | 49.5636 | 4 |
| train | direction | out | 82.4622 | 2 |
| train | type | card_payment | 63.3830 | 7 |
| train | mcc | 6012 | 19.0338 | 12 |
| validation | currency | chf | 49.1975 | 4 |
| validation | direction | out | 82.3960 | 2 |
| validation | type | card_payment | 62.8894 | 7 |
| validation | mcc | 6012 | 19.3686 | 12 |
| test | currency | chf | 49.5849 | 4 |
| test | direction | out | 82.5979 | 2 |
| test | type | card_payment | 63.4047 | 7 |
| test | mcc | 6012 | 19.1484 | 12 |
| unlabeled pretrain (100k sample) | currency | chf | 50.7090 | 4 |
| unlabeled pretrain (100k sample) | direction | out | 82.4470 | 2 |
| unlabeled pretrain (100k sample) | type | card_payment | 63.1420 | 7 |
| unlabeled pretrain (100k sample) | mcc | 6012 | 19.3500 | 12 |

The unlabeled file is profiled exactly for rows, clients and cutoff dates. Its categorical comparison uses a deterministic 100,000-row uniform-by-file sample to keep this report fast and reproducible. It is appropriate for unsupervised vocabulary, merchant-normalization, or recurrence-statistics learning only if its type, currency, MCC and temporal distributions resemble labelled partitions. It cannot supply target labels and must never be mixed into supervised validation scoring.

## Anomaly and leakage checks to carry into modelling

1. Reject any transaction at or after the cutoff and fit every learned transformation on train clients only.
2. Preserve client disjointness across train, validation and test. The overlap table is the first guardrail.
3. Audit literal target words and highly target-pure descriptions/MCCs. A synthetic generator can make these unusually easy but they may still be legitimate historical evidence; validation determines whether they generalize.
4. Treat `client_id` as an identifier only. Never use its raw or numeric suffix as a model feature without a validation-supported reason.
5. Keep currency with amount, inspect negative/zero/extreme amounts and fees, and avoid global scaling fitted on validation/test.
6. Build features from history only: repeated descriptions/MCCs, amount stability, cadence, recency and direction/type proportions.

## First baseline recommendation

Start with a deterministic client-level recurrence scorer that identifies repeated outgoing description/MCC streams, ranks the family-like candidate by recency, monthly/weekly cadence, count and amount stability, then emits `none` when no stream is convincing. Tune its thresholds exclusively on validation macro-F1. A next step is a regularized multiclass model on the same client-level aggregates plus sparse description TF-IDF features; retain the deterministic candidate features for interpretability.
