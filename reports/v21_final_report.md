# V2.1 investigation report

Date: 2026-09-24. Branch: `v2.1-integration`. No commit or push was made.

## Baseline and protocol

The canonical baseline is frozen V2 commit
`96409b7940a991fbda5235b85ffeb40652a4087b`. Historical V1 is
`0199a8b7c8b2c790a9d3447b156f2088708c2864`; the reproduced comparison and
paired artifacts came from benchmark integration commit `172356a` (original
benchmark commit `f4b06e14a3ed6e0c5a7929a1132c0d536e3b25cd`). All comparisons use the
official 2,000/1,000 client train/validation split, cutoff 2026-01-01, eight
fixed labels and Macro-F1 as the primary metric.

| Version | Macro-F1 | Accuracy | Delta Macro-F1 vs V2 |
| --- | ---: | ---: | ---: |
| V1 | 0.271024266 | 0.266000 | -0.120525190 |
| V2 | **0.391549456** | **0.424000** | — |
| V2.1 | not promoted | not promoted | — |

No tested candidate beat V2 on official validation. Therefore this work does
not invent a V2.1 score, configuration or submission.

## Functional V1 versus V2 comparison

V1 and V2 reuse the same underlying client aggregation definitions. The key
difference is how supervised description evidence enters prediction.

- V1 builds 218 numeric columns: volume, amounts, fees, currencies, MCC/type/
  direction counts and shares, activity windows, calendar/gap aggregates,
  unsupervised recurrence, plus 72 `family_*` columns derived from exact
  description-to-label lift. It also computes word/bigram TF-IDF for logistic
  candidates. The selected V1 predictor is not the TF-IDF or CatBoost model: it
  is a 32-input recurrence heuristic using four `family_*` terms per label.
- V1's winner was calibrated on official validation to `none_bias=-1` and
  `temperature=1`. The negative bias suppresses the most frequent class,
  explaining much of its very low `none` recall (17/293).
- V2 deliberately removes all 72 supervised family columns from CatBoost.
  Its 146 features are label-free history aggregates; CatBoost is fixed at 300
  iterations, depth 4, learning rate 0.05, balanced class weights and seed 42.
- V2 fits the old supervised description map separately and uses it only for
  clients excluded from mapping fit. Carles' periodicity factor reweights its
  seven positive-family recurrence scores. Final probabilities are 75% history
  CatBoost plus 25% periodicity heuristic.
- Raw IDs never enter either design matrix. V2 additionally rejects prediction
  on fit clients. V1's ML candidates used in-sample `family_*` values influenced
  by their own labels; V2 removed that unsafe route.
- V1's exact-description TF-IDF, broader CatBoost, model grids, validation-tuned
  heuristic and validation-selected ensemble are historical candidates, not
  components merged into V2.

Thus V2 is an evolution of the common aggregation code, not a V1/V2 merge. Its
large gain mainly comes from label-free CatBoost and correct handling of
`none`; the remaining useful V1 signal is the held-out recurrence heuristic.

## Paired error analysis

The reproduced predictions contain 178 clients correct in both versions, 88
correct only in V1, 246 correct only in V2 and 488 wrong in both.

| Actual class | Both correct | V1 only | V2 only | Both wrong |
| --- | ---: | ---: | ---: | ---: |
| cloud | 25 | 21 | 11 | 32 |
| gym | 39 | 4 | 28 | 50 |
| insurance | 29 | 5 | 24 | 41 |
| mobile | 44 | 13 | 11 | 36 |
| music | 4 | 15 | 9 | 65 |
| software | 22 | 5 | 24 | 53 |
| streaming | 8 | 15 | 12 | 62 |
| none | 7 | 10 | 127 | 149 |

The decisive result is that **88/88 V1-only cases are correct under the raw V1
heuristic**. The periodicity-adjusted heuristic retains 81/88. Conversely, the
history CatBoost is correct for 242/246 V2-only cases. The lost signal is not a
missing generic amount or volume feature: it is almost exactly disagreement
between supervised recurrence evidence and the stronger history model.

V1-only rows have lower CatBoost confidence than V2-only rows (mean top-two
margin 0.1397 versus 0.2048), but they are not cleanly separable from both-wrong
rows (margin 0.1543). Their transaction counts, unique descriptions, history
length, recurrence score and monthly-stream count are also very similar. This
is why a confidence rule recovers minority classes while damaging `none`.

## Music analysis

Among 93 true `music` clients, V1 gets 19 correct, raw history CatBoost 11,
periodicity heuristic 18 and V2 blend 13. There are 15 V1-only `music` wins;
V2 sends seven to `streaming`, two to `none`, two to `gym`, two to `software`,
one to `mobile` and one to `insurance`.

The raw V1 heuristic gets all 15 of those cases right, so exact description lift
plus recurrence contains useful `music` evidence. The 25% V2 heuristic already
partly recovers it (music F1 rises from history-only 0.142857 to 0.169935), but
giving it more influence is not globally safe: the confidence blend raises
internal OOF `music` F1 from 0.238095 to 0.256881 and `streaming` from 0.352078
to 0.400943, while reducing `none` from 0.473441 to 0.410891 and lowering total
Macro-F1.

No evidence supports a music-specific threshold. The error is a general model
arbitration/description-shift problem, and no class rule was introduced.

## Hypotheses and controlled ablations

Only three hypothesis families were investigated.

1. **H1 — safer arbitration of the existing V1 recurrence signal.** Replace
   periodicity with raw V1 evidence at the same weight, or increase its weight
   continuously only when CatBoost margin is small.
2. **H2 — leakage-safe exact family features.** Cross-fit the 72 V1 `family_*`
   columns so every model-training row is transformed by a mapping that excluded
   that client, then join them to V2 history features.
3. **H3 — normalized description evidence.** Mask dates, identifiers and numbers,
   compute leave-one-client-out merchant/class features, and join those 27
   columns to V2 history features.

No client-ID rules, class thresholds, broad search or validation brute force
were used. H1 was stopped after internal OOF. H2 and H3 were taken once to the
official validation because they represented materially different feature
families.

### Five-fold internal train OOF

| Candidate | Macro-F1 | Accuracy | Delta vs V2 OOF | Decision |
| --- | ---: | ---: | ---: | --- |
| V2 frozen | 0.408883545 | 0.423000 | — | control |
| raw heuristic, fixed 25% | 0.408689921 | 0.423000 | -0.000193625 | reject |
| confidence-dependent raw heuristic | 0.407408354 | 0.415000 | -0.001475192 | reject |
| exact cross-fitted family CatBoost | 0.489784808 | 0.522000 | +0.080901263 | advance once |
| exact cross-fitted family + 25% heuristic | 0.489817907 | 0.520000 | +0.080934361 | advance once |
| normalized merchant-history CatBoost | 0.406681467 | 0.469000 | -0.002202078 | reject OOF; diagnostic official run |

The exact-family gain appeared in all five folds, but those folds are sampled
inside the same training generator/distribution. Official validation exposed
that this estimate did not transfer.

### Official validation

| Candidate | Macro-F1 | Accuracy | Delta vs V2 | Decision |
| --- | ---: | ---: | ---: | --- |
| V2 frozen | **0.391549456** | **0.424000** | — | retain |
| history CatBoost only (existing ablation) | 0.383950488 | 0.424000 | -0.007598968 | existing control |
| exact cross-fitted family CatBoost | 0.197868238 | 0.294000 | -0.193681218 | reject |
| exact cross-fitted family + 25% heuristic | 0.194961939 | 0.286000 | -0.196587517 | reject |
| normalized merchant-history CatBoost | 0.116346172 | 0.291000 | -0.275203284 | reject |

The exact-family model predicted `none` for 795/1,000 clients; the normalized
variant predicted it for 886/1,000. This is not a small statistical ambiguity.
Merchant features show large train-to-valid shifts: normalized train-frequency,
repeated share, unique count/share and entropy move by roughly 1.16–1.66 train
standard deviations; `merchant_class_none_max` moves by 1.13. Exact family
features also change their conditional relationship to the target even though
their overall non-zero coverage remains superficially similar (94.7% OOF-train
rows versus 89.7% validation rows). Internal client folds therefore do not
simulate the official split's description distribution.

### Per-class official comparison with the least-bad rejected candidate

| Class | V2 F1 | Exact-family F1 | Delta |
| --- | ---: | ---: | ---: |
| cloud | 0.450000 | 0.259259 | -0.190741 |
| gym | 0.487273 | 0.131579 | -0.355694 |
| insurance | 0.429150 | 0.184615 | -0.244534 |
| mobile | 0.452675 | 0.206897 | -0.245778 |
| music | 0.169935 | 0.085470 | -0.084465 |
| software | 0.373984 | 0.198582 | -0.175402 |
| streaming | 0.250000 | 0.117647 | -0.132353 |
| none | 0.519380 | 0.398897 | -0.120483 |

Every class regresses. There is no accepted V2.1 per-class table because there
is no accepted V2.1 model.

## Leakage, selection and reproducibility

- H2 cross-fitting excludes the held client's transactions and label from its
  mapping fit. H3 subtracts the client's contribution from normalized merchant,
  global and class counts on training rows. Tests audit these invariants.
- V2 history features remain label-free; IDs and post-cutoff events are absent.
- The experiments do not use validation labels to construct features or fit
  models. Validation labels are used only for the reported score.
- Nevertheless, official validation has already been reused across V1, V2 and
  this investigation. All official results remain selection-biased diagnostics.
- The large OOF/official reversal is direct evidence that client-stratified OOF
  alone is insufficient for description features. A future attempt should use
  group/time/domain folds that reproduce description shift, or avoid supervised
  description features entirely.

## Reproduction

From the repository root:

```powershell
# Frozen V2 baseline and its original CSV path
.\.venv\Scripts\python.exe scripts/run_ubs_v2.py

# Fixed V2.1 investigation ablations (ignored JSON outputs)
.\.venv\Scripts\python.exe scripts/evaluate_v21_candidate.py --candidate arbitration
.\.venv\Scripts\python.exe scripts/evaluate_v21_candidate.py --candidate crossfit_family
.\.venv\Scripts\python.exe scripts/evaluate_v21_candidate.py --candidate merchant_history

# Tests and quality checks
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pre_commit run --all-files
```

There is no V2.1 CSV generation command because no candidate passed acceptance.
The existing V2 command remains the correct command to generate
`outputs/submission_v2.csv`; it was not renamed or overwritten by a V2.1 file.

## Verification performed

- Fresh frozen V2 fit/refit: Macro-F1 0.3915494559105542, accuracy 0.424,
  validation SHA-256 `57ca783d...cb9313a`, submission SHA-256
  `da1314cf...95f59adc`; both hashes match the frozen evidence.
- Independent official scorer: 1,000 validation clients and identical metrics.
- Independent submission validator: PASS, 1,000 rows/IDs, exact schema, order,
  vocabulary and no null predictions.
- `pytest -q --basetemp outputs/pytest_v21`: **83 passed**; the temporary test
  directory was removed afterwards. A first run without workspace-local
  `--basetemp` had six setup errors from denied `%TEMP%` access and no test
  assertion failures.
- Ruff check: PASS. Ruff format check: PASS, 81 Python files. The benchmark
  analysis script required six mechanical formatter-only line changes.
- Pre-commit: PASS for all tracked files and again for all new files.
- Quality gate on the normal branch: targeted tests PASS, full tests PASS,
  Ruff/format PASS, Git hygiene PASS, overall WARN only for its documented
  manual/baseline/evaluation evidence checks. Report:
  `outputs/metrics/quality_gate/20260924T165143866799Z/report.json`.

## Recommendation

Keep V2 as the submission baseline. The V1-only analysis found real recurrence
signal, especially for `music` and `streaming`, but neither confidence arbitration
nor leakage-safe supervised description features improve the official objective.
Do not promote or submit either experimental model. The next defensible line is
split-aware/domain-robust text representation evaluated under a protocol that
simulates train-to-valid description shift, ideally with a new untouched holdout.
