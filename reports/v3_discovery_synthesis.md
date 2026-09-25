# V3 discovery synthesis

Integration branch: `integration/v3-discovery`. Baseline: main commit
`5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`. No discovery branch was merged.
Official files match the V2 fingerprints. Outputs, predictions, copied source
snapshots and datasets stay in ignored directories.

## 1. Executive summary

**The main missing signal is family identity in the numeric classifier, encoded
without the training client's own label.** V2's 146-feature CatBoost deliberately
removed supervised family columns to fix leakage, leaving music/streaming mostly
to a 25% heuristic despite their shared MCC. Restoring just 21 cross-fitted family
count/share/alias features raises TRAIN OOF Macro-F1 **.408884 ? .459794** and
VALID **.391549 ? .424111**. This is the clearest operational lever found.

The temporal extension does not transfer reliably: full V3 wins OOF at **.463395**
but reaches only **.398494** on VALID. The fixed 50/50 full-V3/V2 ensemble gives
the highest measured VALID score, **.429901** (+**.038352**, accuracy **.470**).
It improves F1 in seven classes, but loses gym and gains accuracy through none;
positive-client correct predictions fall 290?283. This is not a demonstrated
solution to the approximately .59 leaderboard gap.

**Recommendation: V3 IS PROMISING BUT NEEDS ONE MORE TARGETED ITERATION.** Keep V2
as the frozen fallback. Prefer the simpler identity-only arm as the basis of that
iteration; the ensemble's extra .005790 over it has a paired 95% interval spanning
zero. Do not promote the OOF-selected full model or retrospectively call the
VALID-best ensemble an independently selected winner.

## 2. Results of the seven experiments

Numbers below distinguish the official eight-class client target from conditional
family metrics, historical next-transaction proxies, and omniscient oracles.
The approximate leaderboard 0.59 is supplied context, not a result verified here.

| Experiment | Hypothesis / method | Primary result | Comparable delta vs V2 | VALID adjustment / cost | Author recommendation | Critical classification |
| --- | --- | --- | --- | --- | --- | --- |
| Santiago | Exact-description recurrent candidates + TRAIN presence-lift map; omniscient family selection and none | Recall 0.649222; oracle F1 0.769375; OOF recall 0.792587 | Not a deployable delta | No detector tuning on VALID; labels deliberately used by oracle; CPU stream summaries + MILP | MIXED; retain V2, improve mapping/ranking/none | **STRONG EVIDENCE** for diagnostic ambiguity; **INCONCLUSIVE** for a replacement architecture |
| Ginestar | Twelve date rules; April/July select automatic, October reserved | October top-1 0.1632, top-2 0.3034, MAE 29.669d; fixed V2 change 0.395253 | +0.003704 for the V2 diagnostic only | TRAIN temporal selection; official VALID reused for reporting; CPU, no learned ranker | Temporal ranking helps but insufficient | **PROMISING** as diagnostic; no evidence of a large gain |
| Javier | Leave-one-client-out weak candidate mining; char TF-IDF + logistic family classifier | Seven-positive-class OOF 0.448772 / VALID 0.359701; focused V2 correction reported 0.412915 | +0.021365 only for the full eight-class focused correction | Model chosen on TRAIN OOF; two correction policies compared on reused VALID; CPU sparse models | Moderate family value; reject none gate | **PROMISING**; conditional family scores cannot be compared directly to V2 |
| Christian | Conservative lexical/behavioral clustering on TRAIN + unlabeled | Report .385680 / reproduced .388552; 302 / 303 fewer TRAIN streams | Report -.005870 / reproduced -.002998 | No VALID labels in mapping; fixed comparison; expensive lexical-neighbor/profile work | Exact descriptions sufficient for tested method | **NEGATIVE**; exact result not reproduced, semantic grouping not ruled out |
| Jaime | Causal historical pseudo-cutoffs + pointwise ranker, client-group OOF | Top-1 0.1494 → 0.2142; official F1 0.1603 → 0.1055 | Official learned system -0.2860 vs V2; -0.0548 vs its own temporal rule | Ranker selected on TRAIN OOF; no official tuning; author 319.3s | Pseudo-cutoffs help moderately | **NEGATIVE** for official integration; useful proxy evaluation machinery |
| Esteban | Identical V2 components; adaptive blend with TRAIN priors versus TRAIN+10k priors | B 0.397818455 → C 0.397834319 | **+0.000015864** isolated pretrain; A→B +0.006269 is not pretrain | Gamma chosen on OOF, six VALID weights also reported; earlier failed trials inspected VALID; CPU 12k-client priors | Pretrain does not improve current system | **NEGATIVE** for a material pretrain effect; OOF priors are transductive |
| Laura | Mapped streams → next date → recurrence/family/none gates → horizon | Newly executed: A 0.283051, B 0.127304, C=D 0.056651; all-none D | D -0.334898; override +0.000000 | No measured result in handoff; frozen rules; CPU light | Do not promote before running | **NEGATIVE**, now measured; error complementarity is an all-none artifact |

### F1 per class and comparability

Order throughout this report: cloud, gym, insurance, mobile, music, software,
streaming, none. A dash means the experiment does not define that metric, not zero.

| Experiment / metric | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2, official F1 | .450000 | .487273 | .429150 | .452675 | .169935 | .373984 | .250000 | .519380 |
| Santiago, oracle F1 | .780822 | .593023 | .835294 | .837989 | .601504 | .893617 | .910112 | .702638 |
| Santiago, candidate recall | .640449 | .421488 | .717172 | .721154 | .430108 | .807692 | .835052 | — |
| Javier, seven-positive F1 | .461538 | .433628 | .305263 | .387435 | .368159 | .289157 | .272727 | — |
| Christian, reported normalized F1 | .45783 | .48375 | .42353 | .44444 | .16149 | .38710 | .21935 | .50794 |
| Christian, reproduced normalized F1 | .459627 | .477941 | .434783 | .446281 | .175000 | .392000 | .213836 | .508946 |
| Esteban B, official F1 | .459627 | .487273 | .440000 | .450820 | .172185 | .372470 | .280488 | .519685 |
| Esteban C, official F1 | .459627 | .487273 | .440000 | .450820 | .173333 | .372470 | .280488 | .518664 |
| Laura D, official F1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | .453210 |

Javier's positive-only music F1 removes false positives originating from all 293
true-none clients. It is **not** a demonstrated +.198224 official music gain.
The comparable focused correction reports music .333333 and streaming .319588,
with none unchanged. Exact reproduced Ginestar/Jaime class metrics appear in the
reproduction appendix; neither proxy defines official per-family F1 on its own.

### Javier: reproduced positive-only confusion matrix

Rows true, columns predicted; this excludes true-none clients.

| true / predicted | cloud | gym | insurance | mobile | music | software | streaming |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 66 | 4 | 4 | 4 | 6 | 2 | 3 |
| gym | 27 | 49 | 10 | 12 | 14 | 6 | 3 |
| insurance | 22 | 10 | 29 | 6 | 13 | 11 | 8 |
| mobile | 21 | 4 | 16 | 37 | 14 | 4 | 8 |
| music | 14 | 11 | 13 | 5 | 37 | 7 | 6 |
| software | 23 | 14 | 12 | 10 | 13 | 24 | 8 |
| streaming | 24 | 13 | 7 | 13 | 11 | 8 | 21 |

## 3. Leakage and reproducibility audit

| Check | Evidence / qualification |
| --- | --- |
| Features strictly before cutoff | Shared loader rejects timestamps >= 2026-01-01. Santiago/Laura repeat the check. Ginestar slices `past < T`; Jaime checks history <T and future in [T,T+90d). V3 public feature/model entry points reject future/null times. |
| No VALID labels in family maps | Santiago, Javier, Laura, Christian's supervised scorer and V3 fit mappings on TRAIN clients only. Source inspected; evaluator and prediction flow remain separate. Some source runners load VALID labels early but do not pass them to fit. |
| Own-label mapping leakage | Santiago refuses transform on map-fit clients. Javier subtracts the client's counts before candidate selection. His selection still conditions on that client's training family: legitimate weak supervision, **not verified stream ground truth**. V3 numeric family training features use inner client folds, never the own-client map. V2 ML still excludes all family columns. |
| Client grouping | Santiago/Javier/Esteban use one label row per unique client and stratified client folds. Jaime puts all cutoffs/candidates of a client in the same fold. Ginestar holds out time, **not clients**. V3 uses five outer and five inner client folds. |
| Pseudo-cutoff causality | Complete 90-day observable future before official cutoff. Features do not include future-derived winner/date columns. January labels are not recycled for historical targets. Future supervision only labels candidate reappearance, not official subscription identity. |
| Unlabeled not mislabeled | Esteban's priors accept histories, not target labels; no V2 pseudo-labels used as truth. Christian uses histories for clustering. Unlabeled prior benefit is measured B→C. |
| Unlabeled partition isolation | Direct scan confirms 10,000 clients / 749,935 transactions and zero client overlap with TRAIN, VALID or TEST. |
| Test isolation | No test-label file exists or is used. Some shared loaders validate TEST IDs/history while loading splits, without fitting or selecting on it. V3 loads TEST transactions only in submission phase; file fingerprints are nonpredictive. |
| IDs | Used for grouping, joins and fold membership. No identifier appears in model feature lists. Jaime uses description for tied-event ordering, a proxy limitation, not numeric ID prediction. |
| Normalizer fit | Christian fits on TRAIN+unlabeled, no VALID labels. Internal holdout histories participate in the unsupervised map: transductive, not a clean inductive estimate. |
| Esteban OOF qualification | `_select_gamma_oof` receives priors fitted on **all** TRAIN histories, including held-out clients. Labels remain isolated, but unsupervised preprocessing is not fold-isolated. Earlier failed approaches and forced VALID grid add selection bias. The near-zero B→C official contrast remains informative. |
| Evaluator | Git diffs confirm none of the seven branches changed `evaluation/official.py` or `ubs/evaluation.py`. V3 calls the same fixed-eight-class scorer. |
| Oracle labels | Oracle explicitly uses true labels to choose families and none, including optimizing error assignment. Valid diagnostic ceiling; **INVALID / LEAKAGE RISK if presented as predictive performance**. |

There is no identified post-cutoff or VALID-label leak in the final V3 predictor.
That does not imply independent generalization: selection bias, latent-target
noise and dataset shift remain material.

### Source commits and reproduction

`python scripts/reproduce_v3_discovery.py NAME` extracts a pinned Git archive
into `outputs/metrics/v3_reproduction/NAME`, copies local inputs, runs the
original entry point, and records command, commit, seed, time and input hashes.
No cherry-pick, merge or edits to the archived predictive code are performed.
All models use seed 42 where randomization exists.

| NAME / metric | Source commit | Expected | Obtained | Difference | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2`: Macro-F1 | `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749` | 0.391549456 | 0.391549456 | within report rounding | 555.4 |
| `santiago`: oracle Macro-F1 | `c040ec7fbf3e96e07374db95ab92a94489132dce` | 0.769374893 | 0.769374893 | within report rounding | 599.5 |
| `javier`: positive-only Macro-F1 | `c7075bbfce4b06d8a59b9978f455b2ad903cec47` | 0.359701000 | 0.359701085 | within report rounding | 609.7 |
| `ginestar`: fixed V2 date change | `027169419f510f8707d679fa1e91f1d8ef92922f` | 0.395253013 | 0.395253013 | within report rounding | 1727.8 |
| `jaime`: official ranker Macro-F1 | `101ad060b31e8f3d7deb3ae9033462d0b6f5d65d` | 0.105530715 | 0.105530715 | within report rounding | 1048.7 |
| `esteban`: isolated C-B | `25f029cd99bc277d97f3dd242089889b34ecfee5` | 0.000015864 | 0.000015864 | within report rounding | 943.6 |
| `christian`: normalized V2 Macro-F1 | `5f86fd9eb1f8277a8fcf17623f09fb88e3fb4ee1` | 0.385680000 | 0.388551863 | +0.002871863 | 715.1 |
| `laura`: full D Macro-F1 | `581f3b4e2395e4d001aa879dc596ae53ea07e8dc` | unmeasured | 0.056651199 | first execution | 28.0 |

Commands: the named wrapper above runs the original entry point with defaults;
`ginestar` adds `--v2-diagnostic`; `laura` receives the freshly reproduced V2
validation file; `v2` writes an isolated submission. Javier initially ran before
the V2 file existed, so its optional correction was measured separately by the
V3 runner, not fabricated in the original reproduction. All eight exits were 0.

V2 reproduced **146 features**, Macro-F1 **0.3915494559105542**, accuracy **0.424**,
and the exact previous validation/submission SHA-256 values:

- Validation: `57ca783d554b9aea6602f1046582dc18c93ddfe5d418ba50173ab1e36cb9313a`.
- Submission: `da1314cfdea23ffea7756ff82db4e7d4efe77c6fcaf33cb0b66a689795f59adc`.

Reproduced official per-class diagnostics (same class order as above):

| Method | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ginestar | 0.459627 | 0.489051 | 0.435484 | 0.448980 | 0.172185 | 0.377049 | 0.259259 | 0.520388 |
| Jaime learned ranker | 0.129630 | 0.046154 | 0.000000 | 0.098592 | 0.021053 | 0.066667 | 0.053097 | 0.429054 |

The local runtime is Python 3.12.10 x64, using Windows emulation on ARM. The
original ARM Python lacked dependencies; installing pandas required an absent
compiler. A local ignored x64 runtime resolved this. Core prediction versions:
NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1, CatBoost 1.2.10. Wall times include
contention from concurrent local reproductions, so they are not benchmarks.

## 4. Bottleneck decomposition

### Candidate coverage is not stream ground truth

Santiago detects 13.124 candidate streams and 3.355 candidate families per VALID
client. Of 707 positives, 459 are covered; 450/459 covered clients have multiple
candidate families. Of the other 248: one has no due stream, 67 have the mapped
family only outside the projected horizon, and 180 miss the family mapping.
This is an operational decomposition under **that detector**, not additive
Macro-F1 losses and not a decomposition of unknown true subscription events.

Music: 40 covered, 6 mapped-but-not-due, 47 mapping misses. Streaming: 81 covered,
11 mapped-but-not-due, 5 mapping misses. Thus music's deficit starts before
ranking, while streaming has substantial candidate ambiguity. Gym's recall
.421488 is another major coverage weakness. An oracle .769375 does not establish
that a real ranker can recover the gap from .391549.

### Ranking and date prediction

Ginestar's October automatic rule has top-1 16.32%, top-2 30.34%, versus random
14.08%/28.08%; median-gap top-1 is 13.56%. Its MAE is 29.669 days; only 19.18%
of observed candidate events lie within ±7 days. Event recall/precision in the
90-day window are 94.46%/57.22%. This is a weak ordering signal, not an accurate
subscription forecaster. The true next official family/date is unobserved after
January, so **official next-stream top-1 and official date MAE are unavailable**.

| Estimated cadence, October | Date MAE | Ranking top-1 / top-2 | Interpretation |
| --- | ---: | --- | --- |
| weekly | 42.278d | 100% / 100%, only 4 ranking cases | Too few cases; no robust weekly success claim |
| monthly | 21.589d | 20.83% / 44.83%, 232 cases | Most plausible useful calendar segment |
| quarterly | 32.875d | 4.59% / 12.97%, 185 cases | Worse than 14.84% random top-1; phase unreliable |
| fortnightly | 24.351d | 50.00% / 72.22%, 45 cases | Promising but small |

Jaime's client-group OOF ranker improves its own top-1 from .1494 to .2142,
top-2 .2899 to .3532, MAE 33.82d to 21.65d. Yet official Macro-F1 falls from
.1603 to .1055. The next repeated merchant among arbitrary candidates is not
the official next recurring family. Calendar and learned ranking are therefore
excluded from a hard-decision V3 chain. Pseudo-cutoffs remain useful for causal
proxy diagnostics; their improvement cannot certify official-target progress.

### Family identity

Javier's seven-positive confusion matrix has 24 streaming→cloud errors versus
11 streaming→music; music→cloud/insurance/gym are 14/13/11 versus six
music→streaming. The problem is broader than confusing those two names.
High-support aliases are semantically useful but client-label purity is only
about .30–.37 because every client can have several families. Generic
`digital plus`, `monthly plan`, `premium plan` are low-purity and cannot become
hard semantic rules. All 707 selected positive VALID winners are seen in TRAIN;
performance on genuinely unseen winning descriptions is **not measured**.

The new V3 hypothesis restores family identity to the numeric classifier using
cross-fitting, rather than treating a client's target as the label of every
historical stream. See the factorial evidence below for whether this transfers.

A post-freeze diagnostic using the new map and **all** outgoing history (including
singletons, no due filter) finds the associated target family for 659/707 positives
(**.932107**): cloud .955056, gym .966942, insurance .929293, mobile .990385,
music .860215, software .894231, streaming .917526. But 291/293 none clients
also have a mapped family. This is a broader, differently mapped candidate set,
**not a comparable improvement of Santiago's recall and not stream ground truth**.
It supports preserving family evidence before aggressive temporal pruning, while
leaving next-family selection and none unresolved. It was computed only after
predictions were frozen and was not used to change the candidate.

### None

The oracle recognizes every none by construction, although 289/293 have due mapped
candidates. Ginestar detects 0/71 temporal-none ranking cases; active veto detects
1/71 while sacrificing event recall. Jaime's learned temporal-none recall is
.0986. These proxy none definitions exclude new/unqualified streams and do not
equal official none. Javier's binary gate predicts 973/1000 none, yielding full
official F1 .074278. Laura D predicts 1000/1000 none, F1 .056651. **None is not
solved**, and unconditional high activity F1/accuracy is not evidence that it is.

Laura's paired counts are V2-only correct 290, stream-only correct 159, both
correct 134, neither 417. All 159 stream-only wins are true-none cases from the
constant-none rule. The frozen override changes zero predictions; this does not
justify probability blending, and Laura provides no calibrated probabilities.

### Normalization and pretrain

Christian reports 358 merged TRAIN description identities and 302 fewer actual
client streams. Reproduction finds 359 merges and 303 fewer streams (0.60%),
with Macro-F1 .388551863 rather than reported .385680. Relative gap MAD slightly
worsens (.45719→.45761), MCC/type stability drops .203 percentage points, and
reproduced streaming F1 declines .03616. The exact control still reproduces V2.
The additional merge is confirmed; its cause is not isolated. Nearest-neighbor
ties, floating-point thresholds and rank sorting without an explicit stable
tie key are portability risks, not a proven explanation. Lower the priority
of the exact normalization claims accordingly.
Lexical clustering does not solve a large historical fragmentation problem in
this experiment. Removing generic words merges 1,104 descriptions aggressively;
larger streams alone are not quality. Out-of-time merge stability was not
established; pooled descriptive stability is not a temporal holdout.

The unlabeled file expands description coverage from 1,475 to 2,673, covering
274/407 unseen VALID strings. B→C changes one prediction, music F1 +.001148,
none F1 approximately -.0010 and streaming unchanged. The isolated overall gain
is .000015864, not the .006285 A→C headline. This rejects a large contribution
from **the tested priors**, not every possible use of unlabeled histories.

### Synthetic regularities

TRAIN inspection finds semantic aliases sharing family-specific amount ranges;
music and streaming both predominantly use MCC 5812. Payment rows and refund
rows coexist under the same description. These are legitimate historical signals,
not ID shortcuts. V3 tests outbound-only recurrence and semantic-family grouping
without declaring family timelines to be single merchants. Different currencies
are always separate stream keys; no hidden labels or generator code are used.
There is no evidence here of a deterministic rule sufficient to explain 0.59.

## 5. Chosen V3 architecture and attribution

The reproducible **experimental full V3** was selected by the predefined maximum
TRAIN OOF rule before new official scoring:

1. Strict pre-cutoff client histories; unchanged V2 history features and heuristic.
2. TRAIN client-presence ? smoothed description/family lift; support >=5,
   winning positive-family lift >=1.5, otherwise unknown.
3. Five inner client folds build leakage-safe numeric training features.
4. A: 21 family counts, outgoing-payment shares and distinct-alias counts.
5. B: 13 pooled outbound-card recurrence summaries, keeping currencies separate.
6. AB interaction: 182 recurrence summaries resolved by seven families, using
   both exact-description streams and family/currency timelines.
7. CatBoost on **362 features** (146 history +21 identity +13 pooled +182
   interaction), same frozen 300/depth4/.05/balanced/42 settings as V2.
8. Full probabilities = **75% new CatBoost +25% unchanged V2 periodicity**.
   Argmax over all eight labels makes the none decision; no hard temporal veto.
9. The predefined ensemble averages full V3 and V2 50/50, equivalently
   **37.5% new CatBoost +37.5% V2 history CatBoost +25% V2 periodicity**.

There is no learned next-date ranker or forced earliest-stream selection.
Family timelines are features, not asserted merchant identities. A alone uses
**167 features**, B **159**. The simpler A explains most of the OOF gain and
outperforms full V3 on VALID; that is the architecture supported for the next
iteration, not a post-hoc replacement of the frozen OOF selection.

The runner retains the predefined `full` choice in `frozen_selection.json`.
Its automatically generated `submission_v3.csv` therefore belongs to **full
(.398494 VALID)**, not to the retrospectively best ensemble. All component test
probabilities are saved for review; no CSV is sent and no baseline is switched.

The fixed protocol is in `reports/v3_experiment_protocol.md`. Family presence lift
is inspired by Santiago c040ec7 and Javier c7075bb. The char-family alternative
manually extracts six relevant routines from Javier, preserves its C=2/char_wb
3–5 recipe and .85 focused correction, and adds inference client/cutoff guards.
No whole branch, new clustering, learned date ranker or pretrain prior is merged.
All V1/V2 source and entry points remain unchanged.

## 6. Ablations and overfitting control

| Arm | TRAIN OOF Macro-F1 | VALID Macro-F1 | VALID delta vs V2 | VALID accuracy | OOF folds beating V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 control | 0.408883545 | 0.391549456 | +0.000000000 | 0.4240 | 0/5 |
| V3-A identity | 0.459793827 | 0.424111098 | +0.032561642 | 0.4610 | 5/5 |
| V3-B pooled recurrence | 0.413306186 | 0.389115009 | -0.002434447 | 0.4230 | 4/5 |
| V3-A+B interaction | 0.463395166 | 0.398494213 | +0.006944758 | 0.4480 | 5/5 |
| V3 full (= AB) | 0.463395166 | 0.398494213 | +0.006944758 | 0.4480 | 5/5 |
| 50/50 full V3 + V2 | 0.451231679 | 0.429901160 | +0.038351704 | 0.4700 | 5/5 |
| Javier focused correction | 0.423908063 | 0.412915360 | +0.021365904 | 0.4310 | 5/5 |

The unblended history control is .399766788 OOF / .383950488 VALID.
OOF selects **full**. The highest VALID score is **ensemble**; these are distinct
selection statements. The 75/25 component blend and the 50/50 ensemble weights
were fixed, not optimized on VALID.

### VALID: F1 for every arm

| Arm | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 control | 0.450000 | 0.487273 | 0.429150 | 0.452675 | 0.169935 | 0.373984 | 0.250000 | 0.519380 |
| V3-A identity | 0.481928 | 0.442478 | 0.427273 | 0.468085 | 0.323810 | 0.334802 | 0.306569 | 0.607945 |
| V3-B pooled recurrence | 0.428571 | 0.480000 | 0.434426 | 0.432653 | 0.157143 | 0.382470 | 0.277108 | 0.520548 |
| V3-A+B interaction | 0.440000 | 0.351220 | 0.411765 | 0.456621 | 0.335079 | 0.361905 | 0.259542 | 0.571823 |
| V3 full (= AB) | 0.440000 | 0.351220 | 0.411765 | 0.456621 | 0.335079 | 0.361905 | 0.259542 | 0.571823 |
| 50/50 full V3 + V2 | 0.496815 | 0.435146 | 0.444444 | 0.470085 | 0.298851 | 0.403587 | 0.298507 | 0.591772 |
| Javier focused correction | 0.454545 | 0.498054 | 0.408511 | 0.415929 | 0.333333 | 0.353982 | 0.319588 | 0.519380 |

### VALID: prediction counts for every arm

| Arm | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 control | 71 | 154 | 148 | 139 | 60 | 142 | 63 | 223 |
| V3-A identity | 77 | 105 | 121 | 131 | 117 | 123 | 40 | 286 |
| V3-B pooled recurrence | 79 | 154 | 145 | 141 | 47 | 147 | 69 | 218 |
| V3-A+B interaction | 61 | 84 | 71 | 115 | 98 | 106 | 34 | 431 |
| V3 full (= AB) | 61 | 84 | 71 | 115 | 98 | 106 | 34 | 431 |
| 50/50 full V3 + V2 | 68 | 118 | 108 | 130 | 81 | 119 | 37 | 339 |
| Javier focused correction | 65 | 136 | 136 | 122 | 99 | 122 | 97 | 223 |

### TRAIN OOF: F1 for every arm

| Arm | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 control | 0.435185 | 0.454545 | 0.478664 | 0.426778 | 0.238095 | 0.412281 | 0.352078 | 0.473441 |
| V3-A identity | 0.463964 | 0.469526 | 0.474438 | 0.450704 | 0.395062 | 0.429224 | 0.444934 | 0.550499 |
| V3-B pooled recurrence | 0.438228 | 0.466667 | 0.469043 | 0.444906 | 0.236994 | 0.432203 | 0.339241 | 0.479167 |
| V3-A+B interaction | 0.457883 | 0.453704 | 0.521212 | 0.461538 | 0.375940 | 0.462617 | 0.437086 | 0.537181 |
| V3 full (= AB) | 0.457883 | 0.453704 | 0.521212 | 0.461538 | 0.375940 | 0.462617 | 0.437086 | 0.537181 |
| 50/50 full V3 + V2 | 0.431461 | 0.443966 | 0.516378 | 0.432671 | 0.373626 | 0.452915 | 0.420290 | 0.538547 |
| Javier focused correction | 0.438554 | 0.434590 | 0.455852 | 0.396135 | 0.389831 | 0.390361 | 0.412500 | 0.473441 |

### TRAIN OOF: prediction counts for every arm

| Arm | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 control | 242 | 294 | 325 | 287 | 138 | 261 | 184 | 269 |
| V3-A identity | 254 | 253 | 275 | 235 | 207 | 243 | 229 | 304 |
| V3-B pooled recurrence | 239 | 290 | 319 | 290 | 148 | 277 | 170 | 267 |
| V3-A+B interaction | 273 | 242 | 281 | 238 | 201 | 233 | 228 | 304 |
| V3 full (= AB) | 273 | 242 | 281 | 238 | 201 | 233 | 228 | 304 |
| 50/50 full V3 + V2 | 255 | 274 | 305 | 262 | 166 | 251 | 189 | 298 |
| Javier focused correction | 225 | 261 | 273 | 223 | 274 | 220 | 255 | 269 |

### Paired uncertainty and attribution

| Candidate vs V2 | OOF delta [95% interval] | VALID delta [95% interval] |
| --- | --- | --- |
| V3-A identity | +0.050910 [+0.033032, +0.070144] | +0.032562 [+0.003639, +0.061845] |
| V3-B pooled recurrence | +0.004423 [-0.009603, +0.018461] | -0.002434 [-0.022582, +0.019110] |
| V3 full (= AB) | +0.054512 [+0.033816, +0.074607] | +0.006945 [-0.026135, +0.041147] |
| 50/50 full V3 + V2 | +0.042348 [+0.027384, +0.057788] | +0.038352 [+0.011690, +0.064755] |
| Javier focused correction | +0.015025 [+0.001118, +0.029091] | +0.021366 [+0.004527, +0.038844] |

The full model beats V2 in all five OOF folds, but its VALID interval includes
a regression. Identity A contributes +.050910 OOF and +.032562 VALID;
recurrence B alone contributes +.004423 / -.002434. Adding the temporal and
family-timeline blocks to A contributes only +.003601 OOF and **-.025617 VALID**.
The descriptive factorial interaction (AB?A?B+V2) is -.000821 OOF and
**-.023182 VALID**. There is no robust evidence that those extra temporal
features are needed. These are metric contrasts, not additive causal F1 shares.

Ensemble?A on VALID is +.005790, 95% interval [-.016302,+.027676].
Full?A is -.025617, interval [-.050148,-.000146]. These extra diagnostics were
computed after freezing predictions and did not alter the saved OOF choice.

All ablations share clients, eight labels, the official scorer, CatBoost's frozen
300/depth4/.05/balanced/42 recipe, and V2's unchanged periodicity probabilities.
Full equals A+B by design: no unreported extra component. AB includes the
interaction of family identity with chronology, so its interaction gain is not
an isolated test of each individual timeline statistic.

Five outer client folds evaluate models; five inner client folds construct
supervised family features for each training partition. Final TRAIN features
are also cross-fitted; VALID inference maps are fitted on all TRAIN. The char
alternative uses client-isolated weak supervision, not numeric target encodings.

The discovery reports' VALID numbers were known when choosing this experiment.
The new five-arm comparison and thresholds were written down before producing
new VALID predictions. TRAIN OOF chooses the candidate; the same selection
OOF is not an untouched final holdout. All predefined VALID arms are reported,
including failures. Bootstrap intervals are paired resampling of clients and
are descriptive, not corrected for the seven discoveries or candidate selection.

## 7. V2 versus V3

The headline comparison uses the **best measured predefined ensemble**, not the
OOF-selected full model. It is a candidate for review, not a promoted baseline.

| Class | V2 F1 | Best measured V3 ensemble F1 | Delta |
| --- | ---: | ---: | ---: |
| cloud | 0.450000000 | 0.496815287 | +0.046815287 |
| gym | 0.487272727 | 0.435146444 | -0.052126284 |
| insurance | 0.429149798 | 0.444444444 | +0.015294647 |
| mobile | 0.452674897 | 0.470085470 | +0.017410573 |
| music | 0.169934641 | 0.298850575 | +0.128915934 |
| software | 0.373983740 | 0.403587444 | +0.029603704 |
| streaming | 0.250000000 | 0.298507463 | +0.048507463 |
| none | 0.519379845 | 0.591772152 | +0.072392307 |

**Macro-F1 .391549456 ? .429901160 (+.038351704); accuracy .424 ? .470.**
Seven of eight class F1 values increase; gym falls .052126. Music gains .128916,
streaming .048507 and none .072392. However, this is partly redistribution:
true-positive-family correct predictions **290?283**, versus none correct
**134?187**. Positive-client classification accuracy drops .410184?.400283.
Streaming true positives stay at 20; its F1 improvement comes from fewer false
positives. Music true positives rise 13?26 while gym falls 67?52. Therefore
the result is useful for the primary metric but not a broad next-event recall breakthrough.

The full model predicts none for 431/1000 VALID clients, versus 304/2000 OOF;
identity A predicts 286/1000, the ensemble 339/1000, and V2 223/1000. The true
VALID none count is 293. This transfer issue is central to the recommendation.

### V2 control: confusion matrix

Rows true, columns predicted.

| true / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 36 | 5 | 11 | 6 | 4 | 10 | 5 | 12 |
| gym | 7 | 67 | 9 | 9 | 5 | 7 | 5 | 12 |
| insurance | 4 | 6 | 53 | 4 | 2 | 7 | 7 | 16 |
| mobile | 0 | 7 | 15 | 55 | 3 | 8 | 0 | 16 |
| music | 3 | 14 | 16 | 12 | 13 | 12 | 11 | 12 |
| software | 5 | 12 | 10 | 12 | 2 | 46 | 4 | 13 |
| streaming | 2 | 17 | 10 | 10 | 17 | 13 | 20 | 8 |
| none | 14 | 26 | 24 | 31 | 14 | 39 | 11 | 134 |

### V3 full (= AB): confusion matrix

Rows true, columns predicted.

| true / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 33 | 2 | 1 | 8 | 4 | 8 | 2 | 31 |
| gym | 3 | 36 | 7 | 12 | 14 | 7 | 0 | 42 |
| insurance | 3 | 8 | 35 | 4 | 8 | 9 | 2 | 30 |
| mobile | 3 | 3 | 8 | 50 | 3 | 6 | 1 | 30 |
| music | 1 | 6 | 6 | 12 | 32 | 9 | 1 | 26 |
| software | 5 | 5 | 3 | 7 | 8 | 38 | 3 | 35 |
| streaming | 4 | 9 | 5 | 7 | 14 | 11 | 17 | 30 |
| none | 9 | 15 | 6 | 15 | 15 | 18 | 8 | 207 |

### 50/50 full V3 + V2: confusion matrix

Rows true, columns predicted.

| true / predicted | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cloud | 39 | 5 | 7 | 6 | 5 | 6 | 2 | 19 |
| gym | 3 | 52 | 8 | 12 | 8 | 7 | 4 | 27 |
| insurance | 4 | 6 | 46 | 3 | 5 | 10 | 3 | 22 |
| mobile | 2 | 7 | 9 | 55 | 3 | 7 | 0 | 21 |
| music | 3 | 10 | 13 | 12 | 26 | 10 | 0 | 19 |
| software | 5 | 9 | 7 | 9 | 4 | 45 | 1 | 24 |
| streaming | 3 | 12 | 7 | 8 | 15 | 12 | 20 | 20 |
| none | 9 | 17 | 11 | 25 | 15 | 22 | 7 | 187 |


## 8. Risks and verification

- **87 tests pass** with `python -m pytest --basetemp=.pytest_review_tmp/v3_final_verified_inputs -q`.
- `ruff check .` passes; `ruff format --check .` passes (64 files).
- `pre-commit run --all-files` passes all five hooks. Existing unrelated staged
  work was preserved and is excluded from the V3 commit.
- Original source tests: Santiago 4/4, Ginestar 5/5, Jaime 4/4. Ginestar needed
  the repository root on `sys.path` under the embedded runtime; code was not changed.
- The initial default pytest temp directory had a Windows permission error.
  Re-running with an explicit ignored repository-local temp directory passed.
- Official scorer independently confirms ensemble Macro-F1 .429901159739112,
  accuracy .470, all 1,000 VALID clients. V2 source/evaluator diffs are empty.
- Shared and official submission validators pass: exact schema, 1,000 unique
  sample IDs in sample order, eight allowed classes, no nulls.
- New critical tests cover cutoff/null rejection, own-client map isolation in
  every cross-fit, unseen descriptions, exclusion of refunds, separation of
  currencies, alias timelines, none-preserving correction, feature dimensions,
  and probability/index contracts for every factorial arm. The reproduction
  wrapper also refuses stale input copies; all 56 archived inputs were verified
  against their recorded hashes after the runs, with zero mismatches.

Local execution times: OOF 1206.5s; official VALID 164.0s; TRAIN+VALID final
refit and TEST prediction 206.5s. No model files are versioned. The x64 runtime
on this machine is `.venv/runtime_v3/python.exe`; use a supported normal Python
environment with the project dependencies on other machines. The first runs
used the `runpy` bootstrap recorded in provenance. For a fresh standard CLI
reproduction use a new `--output-dir`, since source/input path spellings are
part of the strict cache fingerprint.

### Local review artifacts (ignored, not submitted)

| File under outputs/metrics/ubs_v3 | Meaning | SHA-256 |
| --- | --- | --- |
| `submission_v3.csv` | Frozen OOF-selected full V3; VALID .398494 | `18c2fa19777d92bf968e1b0e72fe88ec8132c6b99d4ae552f0ced6c6a1ddf2f4` |
| `submission_v3_A_review.csv` | Identity-only diagnostic candidate; VALID .424111 | `6b2420f879e1ae84987e7f3e0ee8c148d0d752982d7ba9a0e66f47b7cf9f011c` |
| `submission_v3_ensemble_review.csv` | Highest measured fixed ensemble; VALID .429901 | `7e50817cb49d26272e7122f94a0e3a397ad79e8c35e55d6f72208f84860f84db` |

The runner produces the first file and all component TEST probability tables.
The two review CSVs take `idxmax(axis=1)` of the corresponding saved
`test_A_probabilities.csv` / `test_ensemble_probabilities.csv`, align to the
sample IDs and pass both validators. They do not change `frozen_selection.json`.
No challenge submission, PR merge or main update has been made.

Remaining limitations: repeated official VALID use; OOF selection optimism;
unknown hidden leaderboard distribution; noisy client→stream supervision;
OOF/full-fit vocabulary differences; unseen-description generalization;
possibly distinct merchants inside one family timeline; pseudo-cutoff target
mismatch; and conditional date errors that do not penalize all missing events.
No observed result proves the external team's method or the source of its 0.59.

## 9. Next step

**V3 IS PROMISING BUT NEEDS ONE MORE TARGETED ITERATION.**

Freeze the identity-only representation A as the experimental starting point,
remove the unsupported temporal interaction, and run one TRAIN-only nested
experiment targeting the gym/none tradeoff. Use a new labelled client/cohort
holdout for a promotion decision if available; re-splitting the already studied
TRAIN cannot manufacture an untouched holdout. Do not tune thresholds on this
VALID. The aim is to retain music/streaming identity gains without sacrificing
positive-family recall to none. Keep V2 as the production baseline until then.

The tested hypotheses of a major gain from lexical normalization, population
priors, or merely choosing the earliest repeated description are rejected.
The broader stream concept remains useful as a representation, not validated
as a deterministic reconstruction of the official future target.
