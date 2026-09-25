# V4 merchant intelligence — Christian

Branch: `exp/v4-merchant-intelligence-christian`.

**Recommendation: retain the existing V3-A baseline; do not promote B or C.**
The OOF-selected B improves TRAIN OOF Macro-F1 from 0.459794 to 0.482502 but
regresses on the single frozen VALID evaluation from **0.424111 to 0.337778**.
The reusable retrieval representation recovers many masked/generic streams and
is complementary in TRAIN OOF, but alternating-alias recovery fails and the
downstream gain does not transfer. No model or threshold was changed after VALID.

## Base and evaluation contract

The request contained unresolved `BASE_*` placeholders. The repository's
`origin/baseline/v4-frozen`, `origin/main`, and initial task branch all pointed to
`051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`. This experiment uses that frozen V3-A
baseline: reported TRAIN OOF Macro-F1 **0.459793826869**, reported VALID
Macro-F1 **0.424111097737**.

The experiment reads TRAIN and all `unlabeled_pretrain` histories locally. No
transactions or descriptions are sent to an API. TEST is never opened. VALID
histories and labels are opened only in the separate evaluation phase, after
OOF selection is frozen. Historical baseline VALID results were already public
in the repository; this is not a previously untouched benchmark.

Five outer stratified client folds reproduce the baseline split (seed 42).
Inside each training partition, the existing baseline uses its original five
inner folds. New supervised merchant summaries use five **label-independent**
client folds: changing a client's label cannot change its own inner fold or
its own family representation. Outer held-out clients contribute no labels to
either mapping.

The unsupervised index uses all TRAIN histories, including outer held-out
histories, plus unlabeled pretraining histories. This is explicitly a
**transductive TRAIN OOF experiment** permitted by the task. It is not an
inductive unseen-client representation benchmark. VALID clients are checked
for disjointness from the index.

## Implementation

`src/transaction_forecasting/ubs/merchant_intelligence.py` contains the reusable
implementation; the runner is
`scripts/experiments/v4_merchant_intelligence_christian.py`.

Initial stream keys are client, raw description, currency, direction, type, and
a factor-two absolute-amount band. Description is a starting observation, not
an identity. Different amount bands prevent obvious collisions before matching;
they may also create extra fragments that retrieval can reconnect.

Fingerprints contain character TF-IDF (2–5 grams), token TF-IDF, MCC
distribution, median absolute amount, relative amount IQR, median gap, relative
gap dispersion, recurrence count, frequency, recency, and weekly/monthly/annual
closeness. Duplicate timestamps do not increase recurrence. No SVD or remote
embedding service is required. Currency, direction/type and dominant MCC are
hard retrieval blocks; they are separate metadata, not numeric embedding axes.

Reference archetypes group description, compatibility block, coarse amount and
gap bands. Numeric features use medians; MCC distributions average stream
distributions. Client-presence membership is retained separately from labels.

**M1** retrieves the union of 32 behavior and 32 text candidates per block and
returns up to eight compatible neighbors. The fixed score weights are 25% text
and 75% behavior; missing/OOV text uses behavior alone. Text itself is 80%
characters and 20% tokens. Behavior uses exponential mean absolute distance in
log-transformed, unsupervised-scaled coordinates. The similarity floor is 0.55.
Compatibility requires amount ratio at most 2, known-gap ratio at most 1.8,
and MCC distribution overlap at least 0.5. All choices were declared before
OOF; no threshold search or supervised weight optimization was performed.

**M2** forms a graph from compatible retrieval edges scoring at least 0.80.
Every proposed union checks compatibility across **all** pairs between the
two clusters, and clusters are capped at 64 nodes. This prevents incompatible
endpoints from merging through an intermediate node. Graph clusters are
optional evidence; downstream predictions use M1 summaries. Incompatible-pair
count zero is a construction invariant, not proof of true merchant purity.

`FamilyEvidence` adds a soft eight-class posterior. Each labeled client votes
at most once per query, using its maximum neighbor weight (similarity cubed).
Repeated events, aliases, and multiple retrieved nodes never multiply that
client's label. Three prior pseudo-observations smooth the distribution.
An inference call involving a client present in the supervised fit raises an
error. Client future targets are weak stream evidence, not stream annotations.

The new downstream block has 19 features: mean/max evidence for eight families,
mean identity confidence, unknown share, and mean posterior entropy. It uses
outgoing card streams, including singletons, and retains clients without cards
as unknown. The index itself can resolve all transaction types.

## Controlled comparison

| Arm | Features | Classifier and heuristic |
| --- | --- | --- |
| A | Unchanged V3-A history + exact family identity | Frozen CatBoost; 75% numeric / 25% existing periodicity heuristic |
| B | A + 19 merchant summary features | Identical |
| C | History + 19 merchant summaries replacing exact identity | Identical |

Selection is maximum TRAIN OOF Macro-F1, with ties preferring A. There are no
additional ensembles, VALID corrections or merchant-specific exceptions.

| Arm | TRAIN OOF Macro-F1 | Accuracy | Delta Macro-F1 vs A |
| --- | ---: | ---: | ---: |
| A — unchanged baseline | 0.459793826869 | 0.4710 | — |
| **B — add summary, selected** | **0.482502094758** | **0.5085** | **+0.022708267889** |
| C — replace identity | 0.480769621682 | 0.5065 | +0.020975794813 |

A reproduces the published baseline exactly. Fold Macro-F1 values:

| Fold | A | B | C |
| --- | ---: | ---: | ---: |
| 1 | 0.422837 | 0.469847 | 0.466183 |
| 2 | 0.449722 | 0.518279 | 0.488704 |
| 3 | 0.476690 | 0.464929 | 0.488043 |
| 4 | 0.474125 | 0.493583 | 0.493233 |
| 5 | 0.465166 | 0.459112 | 0.460167 |

Paired client bootstrap (2,000 resamples, seed 42) gives B−A a descriptive
95% interval **[+0.008367, +0.037174]** and C−A **[+0.002884, +0.038839]**.
These intervals are not corrected for OOF model selection or transductive fit.
The complete OOF run, including index construction, took **877.9 seconds**.

The gain is concentrated in `none`: A correctly predicts 248 none clients,
B 326 and C 319. Positive-family correct predictions are **694 → 691** for B
and remain 694 for C. B's `none` F1 rises 0.550499 → 0.672165; `software`
rises 0.429224 → 0.483092, while `cloud` and `mobile` F1 decline. This does
**not** establish a broad increase in positive next-family recognition.

B disagrees with A on 18.1% of clients, recovers 175 baseline errors and loses
100 previously correct predictions. The A/B diagnostic oracle reaches
**0.5585 accuracy / 0.532371 Macro-F1**, versus A's 0.4710 / 0.459794.
The A/C oracle reaches **0.5790 / 0.554856**, recovering 216 errors and losing
145. These use OOF targets to measure complementarity and are not deployable
selectors. The selection remains B even though C has a higher oracle ceiling.

### Single frozen VALID evaluation

| Arm | VALID Macro-F1 | Accuracy | Delta Macro-F1 vs A |
| --- | ---: | ---: | ---: |
| **A — existing baseline retained** | **0.424111097737** | **0.4610** | — |
| B — frozen OOF selection, rejected for promotion | 0.337777576990 | 0.4110 | −0.086333520747 |
| C — replacement diagnostic, rejected for promotion | 0.329611702849 | 0.4030 | −0.094499394887 |

| VALID class F1 | A | B | C |
| --- | ---: | ---: | ---: |
| cloud | 0.481928 | 0.373134 | 0.359375 |
| gym | 0.442478 | 0.317460 | 0.367925 |
| insurance | 0.427273 | 0.368098 | 0.335329 |
| mobile | 0.468085 | 0.384615 | 0.337079 |
| music | 0.323810 | 0.329114 | 0.287770 |
| software | 0.334802 | 0.223684 | 0.240000 |
| streaming | 0.306569 | 0.178862 | 0.185714 |
| none | 0.607945 | 0.527253 | 0.523702 |

B predicts `none` for **606/1,000** clients; C 593 and A 286, against 293 true
none clients. B gains 61 none true positives but loses 111 positive-family true
positives: positive-family correct predictions fall **285 → 174**. None
precision drops **0.6154 → 0.3911**. This is a strong transfer failure, not a
close promotion decision. A's VALID score also reproduces the published base
exactly, so the regression is not an altered baseline control.

The frozen selection file still records B. Retaining the existing baseline is
a promotion rejection, not a retroactive change to the OOF selection. There
was one VALID label read, after every arm's predictions were saved; TEST was
never read. No merchant-specific error inspection or threshold adjustment was
performed after VALID.

Possible causes include transductive TRAIN reference membership versus unseen
VALID clients, different retrieval support/confidence, and dependence on weak
client-target posteriors. These are **hypotheses**, not demonstrated causal
findings. A future experiment should use an outer-client-excluded unsupervised
reference with mappings derived from training-client retrieval, and assess
support/uncertainty shift on TRAIN-only held-out clients before any new external
evaluation. It must not retune against these VALID results.

The useful deliverable is therefore the local fingerprint/KNN resolver,
uncertainty-aware demo interface, reproducible diagnostics and a documented
negative downstream result. The graph remains optional. No clustering threshold
search was performed; the <1% coverage abandonment condition was not triggered.

The unsupervised reference contains **12,000 clients**, **2,673 descriptions**
and **27,859 archetypes**. The embedding matrix is 27,859 × 1,292 and its
compressed sparse artifact occupies 2,012,857 bytes. The resumable index,
memberships and cached TRAIN neighbors occupy 22,647,262 bytes locally.

M2 produces **7,837 clusters**: 3,817 singletons; median size 2, p90 6, p99 45,
maximum 64. Multi-alias clusters constitute **27.64%** of clusters and contain
**70.95%** of nodes. This is sufficient coverage to avoid the <1% abandonment
condition, but it is not evidence that all aliases denote the same merchant.
M1 remains the downstream representation; there is no graph-based label pooling.

Across 59,799 TRAIN payment query streams, retrieval abstains on **0.161%**;
the optional hard cluster is withheld on **38.58%** because identity confidence
is below 0.5. TRAIN recognition coverage benefits from the transductive index.

## Diagnostics and their limits

The local corruption suite samples 500 recurring TRAIN streams and changes
their underlying transactions before reaggregation: empty description,
alternating opaque aliases, deletion/decorations, and generic `payment` text.
Each original stream becomes an isolated probe so changes to recurrence count,
gap and recency are real. These tests measure recognition of **known** streams;
the reference contains the original histories. They do not establish unseen
merchant generalization or performance after collapsing a whole multi-merchant
history into one generic description.

Recovery means the original clean top neighbor reappears in the corrupted top
eight. Cluster recovery accepts any node in that neighbor's constrained cluster.
Nearest-neighbor Jaccard/top-1 stability, unknown fraction, graph size
distribution, and cross-alias node reduction are reported separately.
Exact-description recovery is equality of the original and corrupted string.

| Transaction corruption | Neighbor Jaccard | Top-1 stable | Original in top-8 | Original cluster in top-8 | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Description masking | 0.3003 | 37.0% | **79.4%** | 83.4% | 1.8% |
| Alternating alias fragmentation | 0.0459 | 1.3% | **6.0%** | 15.5% | 1.2% |
| Typo/decorations | 0.6608 | 88.4% | **97.2%** | 97.4% | 0.4% |
| Generic replacement | 0.2092 | 18.2% | **64.4%** | 68.0% | 11.0% |

Exact-description recovery is 0% for every corruption. There are 500 probes,
and fragmentation yields 1,000 actual query fragments. The fragmentation result
is a clear failure: low unknown rate does not imply correct identity recovery.
No compatibility thresholds were refined after seeing it. Graph recovery is
slightly more forgiving but does not solve that failure.

The graph's cross-alias node-reduction proxy is **50.34%**. Of reference
descriptions, **63.45%** span incompatible currency/direction/type/MCC blocks.
This confirms that string equality cannot safely be an identity rule, but does
not mean 63.45% of descriptions are proven real-world merchant collisions.

There are no ground-truth merchant IDs. Collision diagnostics therefore count
descriptions spanning incompatible behavior blocks. Family purity is a
client-balanced TRAIN-OOF agreement proxy against client future targets; it must
not be interpreted as transaction or merchant family accuracy.

The client-balanced TRAIN-OOF weak family-agreement proxy is **30.84%**.

| Positive-client candidate recall | Exact identity | Merchant evidence |
| --- | ---: | ---: |
| @1 | 48.82% | 44.69% |
| @3 | 86.24% | 86.24% |
| @5 | 94.23% | 94.94% |

There is **no material candidate-recall gain**. Exact identity supplies an
average of 2.7465 candidates at budget three, while soft evidence supplies
three. Filling exact-identity vacancies with the outer-TRAIN class prior yields
86.39% recall@3, slightly above merchant evidence. Client-level unknown share
is 0.8% for exact identity and 0% for soft evidence, partly because of smoothing.

Candidate recall is reported for positive target clients at budgets 1, 3 and 5.
Exact identity only supplies positive-count families; soft evidence can rank
every family because of smoothing. Candidate counts and this coverage
difference matter when interpreting recall. A soft posterior with no labeled
neighbors falls back to the TRAIN prior, which is not merchant-specific evidence.

The amount bands and dominant-MCC blocking can still fragment a merchant.
The known-gap compatibility constraint can reject alternating aliases whose
observed period doubles. Identity confidence is an uncalibrated similarity and
margin score, not a probability. Clustering cannot prove identity when generic
merchants have indistinguishable fingerprints.

## Reproduction and local artifacts

```powershell
.\.venv\Scripts\python.exe scripts/experiments/v4_merchant_intelligence_christian.py --phase oof --output-dir outputs/metrics/v4_merchant_intelligence_christian_final
.\.venv\Scripts\python.exe scripts/experiments/v4_merchant_intelligence_christian.py --phase valid --output-dir outputs/metrics/v4_merchant_intelligence_christian_final
```

Use a fresh directory for changed source/data. Completed folds can be resumed
with identical inputs. The runner hashes only TRAIN, unlabeled inputs and
relevant source before OOF; it does not inspect VALID labels to fingerprint
them. The VALID guard is written after predictions are persisted and before
reading labels, and refuses a second evaluation.

Ignored local outputs include per-arm OOF/VALID probabilities, the frozen
selection, source/data fingerprints, summary JSON, diagnostics JSON, graph
diagnostics, OOF identity/merchant feature tables, sparse archetype embeddings,
the index, and demo resolver/family artifacts. No datasets, model binaries or
transaction-level output are committed.

The adjacent `v4_merchant_intelligence_christian_summary.json` versions only
small aggregate metrics and the recommendation. Full local results live in
`summary.json`, `valid_results.json`, `diagnostics.json`,
`additional_representation_diagnostics.json` and `additional_oof_analysis.json`.
The two `additional_*` files are descriptive calculations from the saved index
and OOF predictions, not new model fits or selection inputs. Demo resolver and
family artifacts occupy about 8.7 MB and 9.0 MB respectively.

Demo usage after the final fit:

```python
import joblib
from transaction_forecasting.ubs.merchant_intelligence import (
    MerchantIntelligence, resolve_merchant,
)

root = "outputs/metrics/v4_merchant_intelligence_christian_final"
resolver = MerchantIntelligence.load(f"{root}/demo_resolver.joblib")
family = joblib.load(f"{root}/demo_family_evidence.joblib")
evidence = resolve_merchant(unseen_client_history, resolver=resolver, family_evidence=family)
```

Each result contains raw description, sparse merchant embedding, nearest aliases
with separate text/behavior scores, optional canonical cluster, soft family
evidence, confidence and unknown flag. For TRAIN client demos, omit
`family_evidence` or use a mapper fitted without that client. Only load trusted
local joblib files.

## Verification

Tests cover deterministic shuffled-fit embeddings, serialization, fit/transform
parity, aliases/masking, unseen-currency abstention, incompatible behavior and
graph chaining, class ordering, duplicate-presence invariance, ignored label
columns, supervised self-client rejection, own-label perturbation under
cross-fitting, future-history rejection, exact A feature parity, real alias
reaggregation, OOF label-read isolation and one-time frozen VALID evaluation.

Verification: **97 tests passed**, including nine merchant-specific tests.
`ruff check .` passed. Format checking passed for all 67 versioned/task Python
files, and all five `pre-commit run --all-files` hooks passed. The broad
`ruff format --check .` command crashed inside Ruff while scanning the working
directory; the explicit versioned/task file check passed. An unrelated,
pre-existing untracked `test_openai_api.py` was left untouched.
