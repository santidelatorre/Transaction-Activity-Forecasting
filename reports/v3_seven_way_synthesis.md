# V3 seven-way research synthesis

Research only. Audit date: 24 September 2026 UTC / 25 September Madrid local time.
Frozen integration source: `df9fe4135ff812add0e2f6e0733706050b813cc2`.
Synthesis branch: `research/v3-seven-way-synthesis`.

## 1. Executive conclusion

**Primary strategy: B — ONE MORE TARGETED EXPERIMENT ROUND.** Keep V3-A as the
research base and V2 as the frozen fallback. None of the seven additions has
demonstrated a positive, transferable incremental gain with credible TRAIN support.
Do not build V3.1 by stacking their individually selected settings.

The important result is a repeated failure of transfer, not a close race for
first place. NONE gating, soft family posteriors and classwise correction all
look better on TRAIN OOF and worse on VALID. The music specialist loses almost
all of its high-confidence operating region. A global blend selects zero V2
weight. The gym override recovers three VALID clients but fails its gate-CV
criterion. The learned router fails both aggregate comparisons.

Three meanings of “best” must remain separate:

- **Best supported result:** V3-A, VALID **0.424111097737**, OOF
  **0.459793826869**. Jaime's TRAIN-selected alpha=1 returns exactly this model.
  No new incremental gain is established.
- **Highest frozen, TRAIN-selected new candidate:** gym protection, VALID
  **0.427115230668**, with negative gate-CV
  delta. It is a cleanly frozen evaluation, but not a promotion-qualified gain.
- **Highest predefined diagnostic control:** the fixed confidence router,
  **0.441966427217**, despite worse OOF than V3-A. Its parameters were
  not fitted on VALID. Choosing it now because it tops VALID would nevertheless
  be retrospective VALID selection. It is a useful diagnostic, not an endorsed model.

No score in this report is an independent hidden-leaderboard estimate. The
official VALID partition was already used throughout V2 and V3 discovery.
“Clean” below means no identified direct evaluation-label use in fitting or
new VALID parameter selection; it does not mean untouched data, unbiased OOF,
or a statistically established improvement.

The next highest-information test is to separate **supervised-map fit-size
effects from cohort transfer** using TRAIN-only held-client experiments. The
strongest modelling direction remains family identity with reliable transfer,
followed by official-target family ranking. Another threshold sweep cannot
answer why the current scores change meaning between OOF and full-fit inference.

## 2. Repository / provenance audit

Executed first: `git fetch --all --prune`, `git status`, current branch, full HEAD,
ten-commit log and `git branch -a`. The starting synthesis HEAD and remote
synthesis HEAD both equalled `origin/integration/v3-discovery` at `df9fe413...`.
`origin/main` was `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749`; it was not changed.
An unrelated untracked `test_openai_api.py` was present and preserved.
Fetch pruned the obsolete `origin/features/esteban-v2-models` ref.

All seven actual experiment tips descend directly from the same merge-base:
`df9fe4135ff812add0e2f6e0733706050b813cc2`. Six have one new commit; Jaime has two.
Branch names were discovered from the fetched refs, not assumed.

| Experiment | Actual remote branch | HEAD SHA | Commits after merge-base | Primary report |
| --- | --- | --- | --- | --- |
| NONE gate | `origin/exp/v3-none-gate-santiago` | `1cd708c2fc09d2b319ba13418060fb4df325d1c2` | `1cd708c` — none gate | `reports/handoff/v3_none_gate_santiago.md` |
| Disagreement router | `origin/exp/v3-disagreement-ginestar` | `888d291e790d3a81566deb9961f6255dd87c3421` | `888d291` — evaluate per-client routing | `reports/handoff/v3_disagreement_gate_ginestar.md` |
| Family mapping | `origin/exp/v3-family-mapping-javier` | `02807fe27266ff4f343f9fb9eb6cdd9f44bdc160` | `02807fe` — improve cross-fitted identity | `reports/handoff/v3_family_mapping_javier.md` |
| Music specialist | `origin/exp/v3-music-streaming-christian` | `7bf01c7e4e6755894f4da3facbaf85a8f65cd63f` | `7bf01c7` — evaluate specialist | `reports/handoff/v3_music_streaming_christian.md` |
| Calibration | `origin/exp/v3-calibration-esteban` | `e2e534ebd2ab3edb65104c644aaa5d0017f44050` | `e2e534e` — evaluate calibration | `reports/handoff/v3_classwise_calibration_esteban.md` |
| Gym protection | `origin/exp/v3-gym-protection-laura` | `218bfcac4ed9495cd6367026b08372339b9f1cd6` | `218bfca` — evaluate gym gate | `reports/handoff/v3_gym_protection_laura.md` |
| Jaime blend | `origin/experiment/v3a-v2-blend` | `db21263dc6585fabf556c589cb1c4a0a9f34a642` | `a398d64` — OOF selection; `db21263` — blend handoff and unrelated additions | `reports/v3a_v2_blend.md` |

The reports and complete added predictive/selection source were read alongside
the three-dot diffs, logs and file inventories. No experiment was merged or
cherry-picked. Isolated `git archive` snapshots under ignored outputs supported
execution. Archive source bytes are verified separately from Git HEAD: a script
inside an archive can inherit the enclosing synthesis repository's HEAD, so the
wrapper's `snapshot_commit` is the authoritative execution source identifier.

### Complete changed-file inventory

| Branch key | Changed files relative to merge-base | Test additions / scope | Predictive and evaluation changes |
| --- | --- | --- | --- |
| NONE | report above; `scripts/run_v3_none_gate.py`; `src/transaction_forecasting/ubs/v3/none_gate.py`; `tests/test_ubs_v3_none_gate.py` | 4 parameterized cases: feature contracts, family preservation, logistic/tree output | New optional predictor and selection runner; no existing predictor or scorer changed |
| Router | report above; `scripts/run_v3_disagreement_gate.py`; `src/transaction_forecasting/ubs/v3/disagreement_gate.py`; `tests/test_v3_disagreement_gate.py` | 3 tests: probability contracts, second-layer coverage, error accounting | New optional router and runner; same scorer |
| Family | report above; `scripts/experiments/v3_family_mapping_javier.py`; `src/transaction_forecasting/ubs/v3/identity.py`; modified `tests/test_ubs_v3.py` | 3 parameterized own-label mutation tests, with fold membership fixed | New identity representation and runner; existing production source unchanged |
| Music | report above; `scripts/experiments/v3_music_streaming_christian.py`; `tests/test_v3_music_streaming_christian.py` | 2 tests: conservative override, fit-client rejection | Predictor lives in experimental runner; same scorer |
| Calibration | report above; `reports/handoff/v3_classwise_calibration_protocol_esteban.md`; `scripts/run_v3_classwise_calibration.py`; `src/transaction_forecasting/ubs/v3/calibration.py`; `tests/test_v3_classwise_calibration.py` | 5 tests: probability contracts, identity, deterministic search, outer-label isolation, perfect baseline | Optional decision correction and selection evaluator added; official metric unchanged |
| Gym | report above; `scripts/run_v3_gym_protection.py`; `src/transaction_forecasting/gym_protection.py`; `tests/test_gym_protection.py` | 16 cases: ID/probability contracts, threshold selection, gate-CV isolation, tampering, freeze sequencing | Optional override/selection runner; official metric unchanged |
| Jaime | report above; `scripts/run_v3a_v2_blend.py`; `src/transaction_forecasting/evaluation/probability_blend.py`; `tests/test_v3a_v2_blend.py`; `.vscode/settings.json`; `AGENTS.md`; `laracopilot-fluxfuel21`; `reports/handoff/jaime_unified_v2.md`; `reports/handoff/jaime_unified_v2.patch` | 19 cases: exact endpoints, IDs/probabilities, ties, freeze, cache integrity, VALID sequence and submission mocks | Probability helper under `evaluation/` does not alter scorer; extra metadata/docs/gitlink are unrelated |

**Flag:** Jaime's tip contains an unrelated Git link, commit
`cdad65d705844a05de5d928a98b86090eb98af6a`, plus editor/agent settings and an old
dashboard/tracking handoff and patch. The patch was read as text, never applied;
the Git link was not initialized. Those contents are not modelling evidence and
must not accompany any future predictive integration. No effect from them on
the blend execution was found. Contents of the external linked repository were
not audited; it is outside the seven predictive execution paths.

### Metric, labels and evaluator integrity

The following Git blob IDs are identical at the base and all seven branch tips;
per-commit path history also shows no intervening edits:

| Protected source | Git blob |
| --- | --- |
| `src/transaction_forecasting/evaluation/official.py` | `79abc7b03b012710042784a4203e350d6f1f10c6` |
| `src/transaction_forecasting/ubs/evaluation.py` | `c0e81021f4fa27d9cf2110d767b90855d8eea5b6` |
| `src/transaction_forecasting/ubs/data.py` | `7962cb2136013a807335214c13a4d2c32850a496` |

Every branch's tracked `data/` tree contains only four `.gitkeep` files. No
dataset, VALID labels or test labels were added or modified in Git. Local TRAIN
and VALID label SHA-256 values match V2's published fingerprints:

- TRAIN: `cab0ec46064348f970ffafdd4f42a12d41a8db660e9db835e7235851b556ff9f`.
- VALID: `979bd9b69253084fc9e2b1ef473c07a628a8b18091169051ba304cfed0fb5642`.

Transaction SHA-256 values:

- TRAIN: `ba0902b33b1181921ee5b1f8f445693201c0968ca52d70a92e12cbb114e7d70a`.
- VALID: `2c30941d49ca20f7b082526407f24529c5a704205610373eff952f4f444a3622`.

Transaction hashes also match the published references and are recorded before
and after reproduction. No test-label input exists in the supplied local files
or audited execution paths. Historical unrecorded author-side activity cannot
be proved absent by a Git diff; this audit establishes source behavior and the
inputs actually used here, not an unlimited historical guarantee.

All branch runners call the unchanged fixed-eight-class scorer. Source review
found no monkeypatch, alternative label set, dropped-client scoring, or rewritten
target in a candidate's official metric path. New OOF/selection code is an
evaluation-procedure change and is audited below; it is not a metric-definition
change. Oracle analysis here is separately and explicitly nondeployable.

## 3. Baselines

The original discovery synthesis/protocol and V2 full report/summary were read
before the seven handoffs. The V2/V3 runners, V2 implementation, all V3 modules,
UBS scoring adapter and official metric implementation were inspected.

| Reference | TRAIN pooled OOF Macro-F1 | VALID Macro-F1 | VALID accuracy | Interpretation |
| --- | --- | --- | --- | --- |
| V2 | 0.408883545420 | 0.391549455911 | 0.424 | Frozen fallback |
| V3-A | 0.459793826869 | 0.424111097737 | 0.461 | Identity-only research base |
| Full V3 = AB | 0.463395166332 | 0.398494213467 | 0.448 | Original OOF-selected arm; transfer failure |
| 50/50 V2 + full V3 | 0.451231679418 | 0.429901159739 | 0.470 | Original fixed diagnostic ensemble, not V2+A |

V2 is 75% small balanced CatBoost over 146 label-free history features plus
25% periodicity heuristic. CatBoost: 300 iterations, depth 4, learning rate .05,
seed 42, four CPU threads. The heuristic uses a TRAIN-only description map at
held-client inference; it rejects fitted clients.

V3-A adds 21 positive-family counts/shares/distinct-alias counts to the numeric
model, for 167 features. Hard description maps use unique-client presence,
smoothed positive-family lift, support >=5 and winning lift >=1.5. Every training
row's map excludes that client through five inner folds. Five outer stratified
client folds evaluate the complete base model. Final TRAIN features are also
cross-fitted; VALID receives the full-TRAIN map.

Full V3 adds 13 pooled recurrence and 182 family/exact-stream timeline features,
for 362 features. It retains the same 75/25 mixture. Its extra OOF gain over A
is only .003601339, while VALID loses .025616884. The previous .429901160
ensemble mixes **full V3**, not A, with V2; confusing these two blends would
misstate Jaime's experiment.

Official supports, in class order cloud/gym/insurance/mobile/music/software/
streaming/none: TRAIN `[190,190,214,191,198,195,225,597]`; VALID
`[89,121,99,104,93,104,97,293]`. There are 2,000 and 1,000 clients respectively.
The primary objective averages eight F1 values, including zero for an undefined
class. Accuracy is secondary. IDs are joined exactly, never used as features.

## 4. Seven experiment summaries

### 4.1 Santiago — dedicated NONE gate

Hypothesis: preserve A's seven-family argmax and learn positive-versus-NONE
separately. Base probabilities and mapped behavioral evidence come from the
unchanged identity model. The selected 27-feature gate uses probability geometry,
V2/A agreement, transaction/payment volume, recurrent descriptions, mapped-family
coverage, recency/span and recurrence summaries.

Six gate variants were compared: one normalized A score, probability-only and
compact logistic regression at C=.1/1, and a depth-2 tree with leaf minimum 30.
The winner is median imputation + standardization + unweighted logistic C=.1.
There are 61 thresholds from .20 to .80. Five second-layer folds use seed 43;
four inner gate folds choose each outer threshold. Selected outer thresholds
are .56/.56/.52/.53/.54; final cross-fitted TRAIN threshold is .53.

Its .485019029 OOF headline is out of sample for each row's base model and gate,
but **not fully nested end-to-end**: another gate-training row's base features
may depend on the gate-held client's label. Choosing among six candidates adds
selection optimism. The report's “standard stacked OOF” explanation does not
remove that dependency. No direct VALID fit or threshold retry was found.

VALID collapses to .357072946: 502 NONE predictions, 57 corrections versus 91
regressions, and every class F1 declines. Positive true positives fall 285→200.
**Reject the implemented gate.** The 5/5 positive gate-fold deltas are evidence
that ordinary TRAIN resplitting fails to reveal this transfer problem, not that
the gate is robust. Cost: base refits plus inexpensive logistic models; medium
implementation and validation complexity.

### 4.2 Ginestar — disagreement routing

Hypothesis: confidence geometry identifies which of V2/A to trust. The 27 inputs
are both probability vectors, maxima, margins, entropies and differences; NONE
probabilities are duplicated as explicit columns. Fit only the 342 TRAIN OOF
disagreements where exactly one base is correct; the other 189 both-wrong
disagreements are still included as evaluation failures. This target is a
correctness selector, not direct Macro-F1 optimization.

Logistic C=1, a depth-3 tree/leaf minimum 20, and histogram boosting
(75 rounds, learning rate .05, seven leaves, minimum leaf 20, L2=1) have fixed
hyperparameters. Five gate folds use seed 314159. Boosting wins at .457599 OOF,
below A, then .417294 VALID. Threshold .5 is fixed. Again, gate-layer client
exclusion is real but base refits are not nested inside the gate evaluation.
Only 37 VALID decisions change: 12 corrections, 13 regressions, 12 other errors.
**Reject the learned router.**

The no-training control selects the argmax of whichever model has the larger
maximum score, breaking ties toward A. It reaches approximately .441966 VALID
but .444519 OOF, a sign reversal against A. This control is complementary
diagnostic evidence with no parameter fit; it is not the OOF-selected learned
gate. Promoting it after this comparison would select on VALID. Its score is
not mislabeled as a VALID-tuned threshold result.

The original runner's f-string report template requires Python 3.12 syntax;
Python 3.11 raises a parse error before execution. The unchanged module is
usable on 3.11 and is replayed with independently verified base artifacts.

### 4.3 Javier — soft family identity

Hypothesis: hard unknown/known assignments discard graded description evidence.
Four arms use identical CatBoost settings: baseline; normalized hard identity;
normalized identity plus seven-family posterior mean/max/recurrent-mean features;
and that posterior arm with char-TF-IDF alias fallback at cosine .88.
The posterior uses prior strength 8 and distinct-client counts. Its 21 new
columns bring the total to 188. Five outer folds each contain five inner map
folds, so held-client feature/model isolation is properly rebuilt for the base
comparison. Arm selection itself uses the same outer OOF scores it reports.

Normalized identity is exactly a no-op. Probability wins at .480617 OOF versus
.478708 with char fallback, but VALID is .324374. It changes 427 clients,
correcting 89 and regressing 163; NONE rises 286→576. All eight F1s decline.
The source persists predictions before reading VALID labels, but the runner
lacks the stronger immutable source/input/freeze guards of Jaime/Laura/Esteban.
This run uses a fresh pinned snapshot to avoid stale-cache ambiguity.

**Additional source finding:** these are seven components of an eight-class
posterior, not probabilities renormalized conditional on being positive:

`p_k(d) = (n_k(d) + 8*pi_k) / (n_all(d) + 8)`.

For a known description, `sum_positive p_k = 1 - p_none`. The seven mean columns
therefore indirectly encode supervised NONE evidence. For unseen descriptions
all seven are zero, conflating missing evidence with missing positive mass.
Saying “no NONE column” is true syntactically but incomplete statistically.
This is legitimate TRAIN supervision, not leakage, and is a plausible contributor
to the NONE failure. It is not a proven causal explanation without an ablation.

**Reject the implemented posterior representation.** It does not disprove family
identity: the hard identity baseline remains the strongest supported change.

### 4.4 Christian — music/streaming specialist

Hypothesis: text plus recurrence supplies targeted identity missed by A. One
three-way logistic model (`music`, `streaming`, `other`) uses C=.5, balanced
weights, seed 42, 600 maximum iterations. Character TF-IDF uses 3–5 grams,
minimum document frequency 3 and at most 5,000 terms. Outgoing descriptions
are repeated up to four times per client. A second arm adds eight recurrence
summaries standardized within the fit partition.

Five client folds independently fit vocabulary, scaling, specialist and base.
There is no trained second-level model. Policy selection uses 16 threshold/margin
combinations plus no-op on the same OOF predictions: thresholds .65/.75/.85/.90,
margins .10/.20, and two specialists. It selects text+recurrence, .75/.20.
Overrides preserve A's NONE, music and streaming predictions, so the rule cannot
correct streaming→music errors that are already inside its own specialist pair.

OOF increases .007926 to .467720, with 37 changes: 21 corrections, 8 regressions,
8 different errors. VALID has **zero** overrides and exactly A's score. The
music/streaming maximum's 95th percentile drops .8445→.4387; only 4/1000 VALID
clients exceed .75, versus 222/2000 OOF. No threshold was lowered afterwards.
**Do not integrate a no-op.** The internal gain is selection evidence, not a
demonstrated specialist transfer. The recurrence summaries also pool currency
within a description, unlike the stricter V3 stream key; this is a feature-quality
limitation rather than a label leak.

### 4.5 Esteban — classwise decision correction

Hypothesis: a bounded correction to A's class scores improves equal-weight F1.
`q_c ∝ p_c exp(b_c)`; this is decision adjustment, not demonstrated probability
calibration. The direction is a centered/shrunk log true-count/raw-argmax-count
ratio with pseudocount 5. Three searched coefficients control this direction,
gym-versus-NONE and music-versus-streaming. Offsets are clipped to ±.25.

The fixed 125-point grid is regularized by .02 times mean squared scaled bias;
it requires .001 regularized gain and per-class loss guards. The calibration
layer has outer and inner leave-original-fold-out selection, with base OOF
fixed. The report correctly discloses that this is not nested base-model CV.
Final coefficients are `(1,-.15,-.075)`; NONE's weight is 1.284025 and gym's .804168.

Cross-fitted OOF is .469540322; .477712639 is the final inner selection score and
must not replace that headline. Every outer delta is positive, yet VALID falls
to .417236296. Accuracy improves .007 through 34 newly correct NONE clients,
while positive true positives drop 285→258. NONE predictions rise to 382 and
gym falls from 105 to 69. NONE hits its upper offset bound in every fold;
the final coefficient triple is selected in no outer fold. **Reject this correction.**
Expanding its bound or preserving only VALID-helpful coefficients would be tuning
on VALID. Runtime after cached probabilities is modest; statistical complexity
is larger than the three-coefficient description suggests because of selection.

### 4.6 Laura — gym protection

Hypothesis: a small confidence-dominant V2 override recovers gym without broader
changes. Start with A; replace with gym when V2 argmax=gym, A argmax differs,
V2 gym≥threshold and V2 gym≥A maximum. Search no-op plus 21 thresholds
.250–.750 at .025 increments. Ties prefer fewer changes, then no-op/higher threshold.
Earlier TRAIN-only dominance probes are disclosed, so the study is not pristine
preregistration.

The selected threshold .40 gives pooled OOF .460202323 (+.000408496), but
leave-one-original-fold-out threshold selection gives .458895310
(**−.000898517**). It makes two held-fold changes and both are regressions.
Two folds select no-op; no fold improves. This gate CV also retains indirect
dependence through cached base predictions, explicitly disclosed by the author.

The frozen VALID rule makes five changes, recovering three gym clients with
zero lost correct predictions, to approximately .427115231. The two other
changes remain wrong. This improves all affected class F1s on this sample,
but five clients cannot rescue failed TRAIN support. **Reject promotion.**
Only 13/74 TRAIN V2-only gym predictions are actual gym; the broader override
pool has very low precision. Minimal implementation cost does not imply a safe
generalization gain. The original CLI's path-key check is nonportable on Windows;
the predictive functions are replayed without weakening their contracts.

### 4.7 Jaime — global V2 + V3-A blend

The exact probabilities are the already blended eight-class V2 and identity-A
outputs, not raw CatBoost scores or full V3:
`P = (1-alpha)*P_V2 + alpha*P_A`.
Thus an interior alpha preserves the shared 25% periodicity heuristic and blends
the two 75% numeric components. It introduces no independent heuristic signal.

Eleven alphas 0,.1,...,1 use the same verified five-fold OOF probabilities.
Ties within .0001 prefer a pure model, then proximity to .5, then lower alpha.
The strict winner is 1.0; the tie rule is inactive. No other V2+A alpha is scored
on VALID in the runner. The original source also has a later submission phase,
which refits on TRAIN+VALID; it was **not run** for this synthesis.

| alpha, weight of A | TRAIN OOF Macro-F1 | Accuracy |
| --- | --- | --- |
| 0.0 | 0.408883545420 | .4230 |
| 0.1 | 0.417628165619 | .4315 |
| 0.2 | 0.435739710883 | .4485 |
| 0.3 | 0.445767764014 | .4575 |
| 0.4 | 0.452915330200 | .4645 |
| 0.5 | 0.456030008437 | .4690 |
| 0.6 | 0.455361430302 | .4680 |
| 0.7 | 0.452919955677 | .4650 |
| 0.8 | 0.453251995584 | .4655 |
| 0.9 | 0.455266480407 | .4665 |
| 1.0 | 0.459793826869 | .4710 |

The best interior score is .003763818432 below A. Frozen alpha=1 has zero
probability, class-frequency, calibration or prediction change relative to A.
It is a negative result for adding global V2 weight, and a useful confirmation
of the simpler base. The selected OOF score has eleven-way selection history;
there is no outer evaluation of alpha selection. The source does not support
an allegation that Jaime selected a VALID-best blend.

## 5. Reproduction status

Evidence labels used throughout:

- **REPRODUCED:** computations executed here from pinned predictive source;
  predictions independently rescored. A portable adapter is identified explicitly.
- **SOURCE-VERIFIED:** implementation confirms the stated mechanism, but no
  fresh execution of the particular statistic.
- **REPORT-ONLY:** author-stated result without local source/artifact reconstruction.
- **NOT VERIFIED:** unavailable evidence, including hidden leaderboard scores.

Every principal TRAIN OOF and VALID score in the master table is **REPRODUCED**.
The fresh predictions were independently rescored with the fixed evaluator,
including per-class F1 and changes against A. All match the author-reported
principal results to their published precision; no unexplained discrepancy remains.

| Execution | Status / method | OOF or selection seconds | VALID seconds |
| --- | --- | --- | --- |
| V2, A, full V3 and original ensemble | REPRODUCED; original base runner, all five outer folds and final refits | 1108.2 | 298.9 |
| NONE gate | REPRODUCED; original runner, base and gate refits | 948.7 | 261.4 |
| Family identity | REPRODUCED; original runner, all four OOF arms and frozen winner | 1013.6 | 204.7 |
| Music/streaming | REPRODUCED; original runner, both specialists and original policy search | 377.8 | 123.9 |
| Calibration | REPRODUCED; original runner on fresh base cache | 55.7 | 6.2 |
| Jaime blend | REPRODUCED; original runner on fresh base cache | 6.1 | 10.3 |
| Router, including fixed control | REPRODUCED via unchanged module and portable adapter | 7.0 combined | Included |
| Gym protection | REPRODUCED via unchanged module and portable adapter | 12.1 combined | Included |

These are elapsed wall seconds with overlapping jobs, not isolated performance
benchmarks. Cached studies inherit the base computation cost. The second-layer
models themselves are cheap; proper end-to-end nesting would multiply base fits.

One calibration VALID invocation began before base VALID provenance existed and
failed at preflight, before predictions or label scoring. Its failure record is
preserved; after the dependency completed, the same frozen calibration ran
successfully. This was an orchestration error, not a model or parameter retry.
The native gym path-key failure and router Python-version incompatibility are
documented separately from successful predictive-module reproduction.

The reproduction wrapper never requests `all` or `submission`. Base OOF/VALID
come from the unchanged V3 runner, which also reproduces V2 and historical full
V3 controls. The router/gym adapter verifies original source/input hashes,
exact seed-42 client fold memberships, cached argmax and endpoint metrics before
calling the pinned modules. It changes orchestration, not predictive code.
Source snapshots, logs, probabilities and predictions remain ignored.

The runtime environment was Windows, Python 3.11.9, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.9.1 and CatBoost 1.2.10. Matching fixed predictions despite
different author platforms is useful reproducibility evidence, not a new
independent statistical replicate.

Historical discovery-only statistics quoted later (659/707 target-family
presence, 291/293 NONE coverage, the .000015864 transductive-prior increment
and earlier next-event proxy findings) are **REPORT-ONLY** numeric evidence
from `reports/v3_discovery_synthesis.md`; their implementing mechanisms were
**SOURCE-VERIFIED**, but those older ablations were not rerun. None is counted
as a reproduced result of this seven-way round. The external ~.59 leaderboard
claim and unrecorded author-side search histories remain **NOT VERIFIED**.

## 6. Master metrics table

All VALID class F1 values below include all 1,000 clients. TRAIN figures are
pooled predictions, not averages of fold F1. Gym's main OOF entry is gate CV;
its selected-rule rescore is separately reported. The fixed router is a
predefined extra control within the router experiment, not an eighth workstream.

| Experiment | Branch | Commit | Base | TRAIN OOF F1 | VALID F1 | Delta V2 | Delta A | Accuracy | cloud | gym | insurance | mobile | music | software | streaming | none | Changed A | OOF to VALID | Leakage risk | Selection risk | Complexity | Reproduced? | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | `origin/integration/v3-discovery` | df9fe41 | df9fe41 | 0.408883545 | 0.391549456 | +0.000000000 | -0.032561642 | 0.424 | 0.450000 | 0.487273 | 0.429150 | 0.452675 | 0.169935 | 0.373984 | 0.250000 | 0.519380 | 315 | both lose | low direct | historical VALID selection | base model | REPRODUCED | Reference |
| A | `origin/integration/v3-discovery` | df9fe41 | df9fe41 | 0.459793827 | 0.424111098 | +0.032561642 | +0.000000000 | 0.461 | 0.481928 | 0.442478 | 0.427273 | 0.468085 | 0.323810 | 0.334802 | 0.306569 | 0.607945 | 0 | same prediction | low direct | historical VALID selection | base model | REPRODUCED | Reference |
| full | `origin/integration/v3-discovery` | df9fe41 | df9fe41 | 0.463395166 | 0.398494213 | +0.006944758 | -0.025616884 | 0.448 | 0.440000 | 0.351220 | 0.411765 | 0.456621 | 0.335079 | 0.361905 | 0.259542 | 0.571823 | 257 | reversal | low direct | historical VALID selection | base model | REPRODUCED | Reference |
| ensemble | `origin/integration/v3-discovery` | df9fe41 | df9fe41 | 0.451231679 | 0.429901160 | +0.038351704 | +0.005790062 | 0.470 | 0.496815 | 0.435146 | 0.444444 | 0.470085 | 0.298851 | 0.403587 | 0.298507 | 0.591772 | 197 | reversal | low direct | historical VALID selection | base model | REPRODUCED | Reference |
| NoneGate | `origin/exp/v3-none-gate-santiago` | 1cd708c | df9fe41 | 0.485019029 | 0.357072946 | -0.034476510 | -0.067038152 | 0.427 | 0.333333 | 0.365482 | 0.389189 | 0.400000 | 0.284091 | 0.305419 | 0.208000 | 0.571069 | 228 | reversal | indirect gate dependence | high | medium | REPRODUCED | Reject |
| Router | `origin/exp/v3-disagreement-ginestar` | 888d291 | df9fe41 | 0.457598934 | 0.417293829 | +0.025744373 | -0.006817269 | 0.460 | 0.459627 | 0.433628 | 0.429224 | 0.454148 | 0.315271 | 0.331839 | 0.300752 | 0.613861 | 37 | both lose | indirect gate dependence | medium | medium | REPRODUCED (adapter) | Reject |
| Family | `origin/exp/v3-family-mapping-javier` | 02807fe | df9fe41 | 0.480617046 | 0.324374444 | -0.067175011 | -0.099736653 | 0.387 | 0.405405 | 0.300000 | 0.339181 | 0.339869 | 0.112903 | 0.313725 | 0.291391 | 0.492520 | 427 | reversal | low direct | medium | medium | REPRODUCED | Reject |
| Music | `origin/exp/v3-music-streaming-christian` | 7bf01c7 | df9fe41 | 0.467719740 | 0.424111098 | +0.032561642 | +0.000000000 | 0.461 | 0.481928 | 0.442478 | 0.427273 | 0.468085 | 0.323810 | 0.334802 | 0.306569 | 0.607945 | 0 | gain vanishes | low direct | high policy selection | medium | REPRODUCED | No-op; reject addition |
| Calibration | `origin/exp/v3-calibration-esteban` | e2e534e | df9fe41 | 0.469540322 | 0.417236296 | +0.025686840 | -0.006874801 | 0.468 | 0.437500 | 0.378947 | 0.435644 | 0.465753 | 0.317949 | 0.339450 | 0.340426 | 0.622222 | 116 | reversal | indirect gate dependence | medium/high | low code; medium CV | REPRODUCED | Reject |
| Gym | `origin/exp/v3-gym-protection-laura` | 218bfca | df9fe41 | 0.458895310 | 0.427115231 | +0.035565775 | +0.003004133 | 0.464 | 0.484848 | 0.458874 | 0.427273 | 0.468085 | 0.325359 | 0.334802 | 0.306569 | 0.611111 | 5 | reversal | indirect gate dependence | high relative to 5 changes | low | REPRODUCED (adapter) | Reject promotion |
| Jaime | `origin/experiment/v3a-v2-blend` | db21263 | df9fe41 | 0.459793827 | 0.424111098 | +0.032561642 | +0.000000000 | 0.461 | 0.481928 | 0.442478 | 0.427273 | 0.468085 | 0.323810 | 0.334802 | 0.306569 | 0.607945 | 0 | same prediction | low direct | 11 OOF choices | low | REPRODUCED | Keep A; no blend gain |
| FixedRouter | `origin/exp/v3-disagreement-ginestar` | 888d291 | df9fe41 | 0.444519429 | 0.441966427 | +0.050416971 | +0.017855329 | 0.471 | 0.497041 | 0.475806 | 0.456897 | 0.472574 | 0.307692 | 0.403433 | 0.335766 | 0.586521 | 122 | reversal | low direct | high if promoted post-VALID | very low | REPRODUCED (adapter) | Diagnostic only |

Read this table with selection provenance, not as a VALID leaderboard. Positive
VALID deltas with negative OOF support are not accepted improvements. Equal
VALID scores for music/Jaime reflect identical predictions, not independent
confirmations of two distinct signals.

## 7. OOF vs VALID analysis

`delta_OOF` and `delta_VALID` subtract A on the same partition. A descriptive
generalization ratio is `delta_VALID/delta_OOF` only when delta_OOF>0. It is not
a calibrated forecast; tiny denominators and opposite signs make it unstable.
For negative/zero OOF gains it is intentionally marked N/A.

| Experiment | delta OOF | delta VALID | Generalization ratio |
| --- | --- | --- | --- |
| NoneGate | +0.025225202 | -0.067038152 | -2.658 |
| Router | -0.002194893 | -0.006817269 | N/A |
| Family | +0.020823220 | -0.099736653 | -4.790 |
| Music | +0.007925913 | +0.000000000 | +0.000 |
| Calibration | +0.009746495 | -0.006874801 | -0.705 |
| Gym | -0.000898517 | +0.003004133 | N/A |
| Jaime | +0.000000000 | +0.000000000 | N/A |
| FixedRouter | -0.015274397 | +0.017855329 | N/A |

Gym selected-rule rescore: 0.460202322721; this is not its gate-CV estimate.

The most consequential reversals are the NONE gate and soft family mapping.
The calibration reversal is smaller but repeats their movement toward NONE.
Music's selected gain disappears completely. Gym is positive only in its
selection rescore and five-client VALID diagnostic. The learned router loses
on both samples. Jaime is algebraically unchanged.

NONE, family posteriors and calibration each improve all five of their TRAIN
evaluation folds. Their failures therefore cannot be dismissed as one unlucky
TRAIN partition. The large negative NONE/family VALID bootstrap intervals
also argue against treating those losses as tiny sampling fluctuations. What
fails is the transfer of the learned relationship; these data do not identify
whether representation fit size, cohort shift or indirect stacking dependence
causes how much of each failure.

Fold means/stds are descriptive over five dependent folds. The gate-fold
partitions differ for NONE (seed 43) and router (314159), while base/family/
music/calibration/gym use the original seed-42 fold assignment. Do not compare
their fold rows as paired observations across differently partitioned models.
No fold standard error is treated as an independent-replicate confidence interval.

| Model | Partition seed | F1 fold 1 | 2 | 3 | 4 | 5 | Mean | Sample SD | Mean delta A | Delta SD | Wins A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 42 | 0.386089 | 0.414131 | 0.423242 | 0.426194 | 0.387149 | 0.407361 | 0.019453 | -0.050347 | 0.017203 | 0/5 |
| A | 42 | 0.422837 | 0.449722 | 0.476690 | 0.474125 | 0.465166 | 0.457708 | 0.022165 | +0.000000 | 0.000000 | 0/5 |
| NoneGate | 43 | 0.483701 | 0.491928 | 0.483970 | 0.467980 | 0.496805 | 0.484877 | 0.010951 | +0.025455 | 0.012465 | 5/5 |
| Router | 314159 | 0.461393 | 0.421597 | 0.434154 | 0.492578 | 0.475377 | 0.457020 | 0.029147 | -0.002387 | 0.013644 | 2/5 |
| Family | 42 | 0.428227 | 0.497871 | 0.488516 | 0.502560 | 0.479366 | 0.479308 | 0.029914 | +0.021600 | 0.017063 | 5/5 |
| Music | 42 | 0.428075 | 0.448691 | 0.487843 | 0.484988 | 0.478599 | 0.465639 | 0.026142 | +0.007931 | 0.005848 | 4/5 |
| Calibration | 42 | 0.439861 | 0.456373 | 0.482779 | 0.488648 | 0.466468 | 0.466826 | 0.019794 | +0.009118 | 0.006482 | 5/5 |
| Gym | 42 | 0.421257 | 0.449722 | 0.476690 | 0.474125 | 0.462247 | 0.456808 | 0.022577 | -0.000900 | 0.001320 | 0/5 |
| Jaime | 42 | 0.422837 | 0.449722 | 0.476690 | 0.474125 | 0.465166 | 0.457708 | 0.022165 | +0.000000 | 0.000000 | 0/5 |
| FixedRouter | 42 | 0.415240 | 0.452515 | 0.452407 | 0.454754 | 0.439204 | 0.442824 | 0.016598 | -0.014884 | 0.012214 | 1/5 |

OOF true NONE prevalence is 29.85%, VALID 29.30%, but A's predicted NONE rate
is 15.20% versus 28.60%. A stable label prior therefore does not imply stable
decision scores. The full V3 rate changes 15.20%→43.10%. Learning an unconditional
OOF correction toward NONE is especially dangerous under this change.

## 8. Per-class analysis

The following cells are **VALID F1 deltas versus A**, not conditional-positive
F1 or precision. Positive values can result from fewer false positives without
recovering more true clients; the change accounting must accompany them.

| Experiment | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | -0.031928 | +0.044795 | +0.001877 | -0.015410 | -0.153875 | +0.039182 | -0.056569 | -0.088565 |
| NoneGate | -0.148594 | -0.076996 | -0.038084 | -0.068085 | -0.039719 | -0.029383 | -0.098569 | -0.036876 |
| Router | -0.022300 | -0.008850 | +0.001951 | -0.013937 | -0.008539 | -0.002963 | -0.005817 | +0.005917 |
| Family | -0.076522 | -0.142478 | -0.088091 | -0.128216 | -0.210906 | -0.021076 | -0.015179 | -0.115425 |
| Music | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| Calibration | -0.044428 | -0.063531 | +0.008371 | -0.002332 | -0.005861 | +0.004648 | +0.033856 | +0.014277 |
| Gym | +0.002921 | +0.016397 | +0.000000 | +0.000000 | +0.001549 | +0.000000 | +0.000000 | +0.003166 |
| Jaime | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| FixedRouter | +0.015114 | +0.033329 | +0.029624 | +0.004489 | -0.016117 | +0.068632 | +0.029197 | -0.021424 |

No class is solved. A's relative strengths are NONE (.607945), cloud (.481928)
and mobile (.468085), but even these leave substantial errors. Streaming
(.306569), music (.323810) and software (.334802) are the main F1 bottlenecks.
Gym (.442478) is a regression from V2 (.487273), and insurance is essentially
flat versus V2.

The family and NONE candidates damage every class on VALID. Their titles do
not describe demonstrated specialization. Music's specialist improves no VALID
class because it takes no action. Calibration improves streaming/NONE and
slightly insurance/software while damaging gym/cloud; it trades class errors
and favors accuracy over the actual objective. Gym protection's increase is
real on these five clients, but not supported out of sample within TRAIN.

A correctly predicts only 21/97 streaming clients; it predicts streaming only
40 times. Streaming→music (24 clients) is its largest single streaming error,
but 52 other streaming clients fail elsewhere. Music has only two errors toward
streaming and many toward insurance/software/NONE. The bottleneck is not a
symmetric two-label confusion, and Christian's preserve-music/streaming gate
cannot resolve the biggest streaming→music confusion by construction.

| TRAIN OOF model | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | 0.435185 | 0.454545 | 0.478664 | 0.426778 | 0.238095 | 0.412281 | 0.352078 | 0.473441 |
| A | 0.463964 | 0.469526 | 0.474438 | 0.450704 | 0.395062 | 0.429224 | 0.444934 | 0.550499 |
| NoneGate | 0.457547 | 0.496454 | 0.486957 | 0.440506 | 0.394667 | 0.441748 | 0.468966 | 0.693309 |
| Router | 0.466819 | 0.481236 | 0.491018 | 0.428571 | 0.380952 | 0.417234 | 0.430913 | 0.564047 |
| Family | 0.466231 | 0.478261 | 0.494929 | 0.459821 | 0.437811 | 0.440758 | 0.461197 | 0.605928 |
| Music | 0.465753 | 0.466513 | 0.469854 | 0.453461 | 0.437647 | 0.435185 | 0.462845 | 0.550499 |
| Calibration | 0.462963 | 0.477612 | 0.457983 | 0.455206 | 0.395990 | 0.442353 | 0.438710 | 0.625506 |
| Gym | 0.460497 | 0.467416 | 0.474438 | 0.450704 | 0.395062 | 0.429224 | 0.444934 | 0.548889 |
| Jaime | 0.463964 | 0.469526 | 0.474438 | 0.450704 | 0.395062 | 0.429224 | 0.444934 | 0.550499 |

## 9. Prediction-change analysis

“Correction” means A wrong→candidate correct; “regression” means A correct→wrong;
“other error” means wrong→different wrong. These are paired client counts, not
independent samples or direct additive Macro-F1 contributions.

| Model | OOF changed | OOF corrections | OOF regressions | OOF other error | VALID changed | VALID corrections | VALID regressions | VALID other error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NoneGate | 207 | 136 | 37 | 34 | 228 | 57 | 91 | 80 |
| Router | 141 | 48 | 45 | 48 | 37 | 12 | 13 | 12 |
| Family | 333 | 149 | 99 | 85 | 427 | 89 | 163 | 175 |
| Music | 37 | 21 | 8 | 8 | 0 | 0 | 0 | 0 |
| Calibration | 142 | 71 | 30 | 41 | 116 | 40 | 33 | 43 |
| Gym | 2 | 0 | 2 | 0 | 5 | 3 | 0 | 2 |
| Jaime | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FixedRouter | 146 | 30 | 62 | 54 | 122 | 47 | 37 | 38 |

VALID cells: corrections / regressions / wrong-to-wrong, grouped by **true** class.

| Model | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NoneGate | 0/16/11 | 0/14/12 | 0/11/9 | 0/20/9 | 0/9/12 | 0/7/12 | 0/8/15 | 57/6/0 |
| Router | 0/3/2 | 0/1/1 | 1/1/2 | 1/4/1 | 0/2/1 | 0/1/2 | 0/1/3 | 10/0/0 |
| Family | 7/17/17 | 2/25/32 | 2/20/14 | 2/31/17 | 1/28/25 | 7/13/26 | 6/5/37 | 62/24/7 |
| Music | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| Calibration | 1/6/5 | 0/14/5 | 1/4/3 | 0/4/7 | 0/3/5 | 1/2/9 | 3/0/8 | 34/0/1 |
| Gym | 0/0/0 | 3/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/1 | 0/0/1 | 0/0/0 |
| Jaime | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| FixedRouter | 2/0/5 | 12/3/2 | 6/0/4 | 5/4/2 | 0/4/4 | 11/2/6 | 4/2/7 | 7/22/8 |

Changed VALID decisions by **predicted** class: count leaving that A class / count entering that candidate class. Diagonal unchanged decisions are excluded.

| Model | cloud | gym | insurance | mobile | music | software | streaming | none |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NoneGate | 24/2 | 30/1 | 36/1 | 60/0 | 35/1 | 24/0 | 13/1 | 6/222 |
| Router | 5/0 | 1/1 | 4/3 | 9/3 | 8/1 | 6/2 | 4/0 | 0/27 |
| Family | 33/15 | 58/12 | 55/6 | 92/10 | 92/6 | 47/24 | 12/26 | 38/328 |
| Music | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| Calibration | 8/2 | 36/0 | 20/2 | 19/3 | 17/2 | 13/4 | 3/7 | 0/96 |
| Gym | 1/0 | 0/5 | 0/0 | 0/0 | 1/0 | 0/0 | 0/0 | 3/0 |
| Jaime | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| FixedRouter | 5/8 | 6/28 | 8/20 | 15/17 | 19/4 | 15/21 | 7/7 | 47/17 |

Every actual-class transition, including small cells omitted from narrative,
is retained in ignored `analysis/analysis.json`. The table above aggregates
only class labels/counts and exposes no client-level data. For music and Jaime
there are no VALID transitions to analyze.

## 10. Pairwise complementarity

| A model | B model | Agree /1000 | Disagree | Both correct | A only correct | B only correct | Both wrong | Error Jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2 | A | 685 | 315 | 334 | 90 | 127 | 449 | 0.674 |
| V2 | NoneGate | 560 | 440 | 282 | 142 | 145 | 431 | 0.600 |
| V2 | Router | 722 | 278 | 346 | 78 | 114 | 462 | 0.706 |
| V2 | Family | 466 | 534 | 249 | 175 | 138 | 438 | 0.583 |
| V2 | Music | 685 | 315 | 334 | 90 | 127 | 449 | 0.674 |
| V2 | Calibration | 643 | 357 | 324 | 100 | 144 | 432 | 0.639 |
| V2 | Gym | 690 | 310 | 337 | 87 | 127 | 449 | 0.677 |
| V2 | Jaime | 685 | 315 | 334 | 90 | 127 | 449 | 0.674 |
| V2 | FixedRouter | 807 | 193 | 381 | 43 | 90 | 486 | 0.785 |
| A | NoneGate | 772 | 228 | 370 | 91 | 57 | 482 | 0.765 |
| A | Router | 963 | 37 | 448 | 13 | 12 | 527 | 0.955 |
| A | Family | 573 | 427 | 298 | 163 | 89 | 450 | 0.641 |
| A | Music | 1000 | 0 | 461 | 0 | 0 | 539 | 1.000 |
| A | Calibration | 884 | 116 | 428 | 33 | 40 | 499 | 0.872 |
| A | Gym | 995 | 5 | 461 | 0 | 3 | 536 | 0.994 |
| A | Jaime | 1000 | 0 | 461 | 0 | 0 | 539 | 1.000 |
| A | FixedRouter | 878 | 122 | 424 | 37 | 47 | 492 | 0.854 |
| NoneGate | Router | 761 | 239 | 369 | 58 | 91 | 482 | 0.764 |
| NoneGate | Family | 645 | 355 | 306 | 121 | 81 | 492 | 0.709 |
| NoneGate | Music | 772 | 228 | 370 | 57 | 91 | 482 | 0.765 |
| NoneGate | Calibration | 823 | 177 | 388 | 39 | 80 | 493 | 0.806 |
| NoneGate | Gym | 767 | 233 | 370 | 57 | 94 | 479 | 0.760 |
| NoneGate | Jaime | 772 | 228 | 370 | 57 | 91 | 482 | 0.765 |
| NoneGate | FixedRouter | 680 | 320 | 341 | 86 | 130 | 443 | 0.672 |
| Router | Family | 573 | 427 | 300 | 160 | 87 | 453 | 0.647 |
| Router | Music | 963 | 37 | 448 | 12 | 13 | 527 | 0.955 |
| Router | Calibration | 877 | 123 | 426 | 34 | 42 | 498 | 0.868 |
| Router | Gym | 958 | 42 | 448 | 12 | 16 | 524 | 0.949 |
| Router | Jaime | 963 | 37 | 448 | 12 | 13 | 527 | 0.955 |
| Router | FixedRouter | 881 | 119 | 423 | 37 | 48 | 492 | 0.853 |
| Family | Music | 573 | 427 | 298 | 89 | 163 | 450 | 0.641 |
| Family | Calibration | 619 | 381 | 315 | 72 | 153 | 460 | 0.672 |
| Family | Gym | 569 | 431 | 298 | 89 | 166 | 447 | 0.637 |
| Family | Jaime | 573 | 427 | 298 | 89 | 163 | 450 | 0.641 |
| Family | FixedRouter | 533 | 467 | 289 | 98 | 182 | 431 | 0.606 |
| Music | Calibration | 884 | 116 | 428 | 33 | 40 | 499 | 0.872 |
| Music | Gym | 995 | 5 | 461 | 0 | 3 | 536 | 0.994 |
| Music | Jaime | 1000 | 0 | 461 | 0 | 0 | 539 | 1.000 |
| Music | FixedRouter | 878 | 122 | 424 | 37 | 47 | 492 | 0.854 |
| Calibration | Gym | 879 | 121 | 428 | 40 | 36 | 496 | 0.867 |
| Calibration | Jaime | 884 | 116 | 428 | 40 | 33 | 499 | 0.872 |
| Calibration | FixedRouter | 804 | 196 | 403 | 65 | 68 | 464 | 0.777 |
| Gym | Jaime | 995 | 5 | 461 | 3 | 0 | 536 | 0.994 |
| Gym | FixedRouter | 883 | 117 | 427 | 37 | 44 | 492 | 0.859 |
| Jaime | FixedRouter | 878 | 122 | 424 | 37 | 47 | 492 | 0.854 |

For V2 versus A, there are 315 VALID disagreements: V2 alone is correct on 90,
A alone on 127, and both are wrong on 98 disagreement rows. This leaves useful
unique V2 signal despite its lower aggregate score. V2-only correct cases include
22 gym, 14 software, 11 insurance, 11 mobile and 11 streaming; only one is music.
A's largest unique benefits are NONE and music. These are retrospective true-class
counts; actual true class is unavailable to a deployable router.

The learned gate does not extract this complementarity successfully. Its final
fit sees only 342 decisive TRAIN rows for 27 inputs, versus 217 decisive VALID
disagreements; the correctness relationship changes materially by class.
Both-wrong cases supply no supervised choice target but still consume one third
of VALID disagreements. A weak predictor can have useful unique corrections
without making a useful blend: the negative global blend and learned router
results demonstrate exactly that distinction.

To separate genuinely new answers from rearrangements of V2/A, the next table
counts correct predictions **only where both V2 and A are wrong**. These are
diagnostic recoveries, not a fitted combination or additive expected gain.

| Experiment | New correct beyond V2/A union | Positive families | NONE |
| --- | --- | --- | --- |
| NONE gate | 47 | 0 | 47 |
| Learned router | 0 | 0 | 0 |
| Family posterior | 70 | 17 | 53 |
| Music specialist | 0 | 0 | 0 |
| Calibration | 26 | 3 | 23 |
| Gym | 0 | 0 | 0 |
| Jaime | 0 | 0 | 0 |

**Family posteriors contain the most novel complementary experimental answers**,
despite being the worst standalone model. Their 17 new positive-family answers
are cloud 5, gym 1, insurance 1, mobile 1, music 1, software 5 and streaming 3.
The 53 new NONE answers dominate the total. This is a reason to investigate
the representation and its transfer failure, not to deploy a gate trained to
select these VALID successes. V2 remains the most useful established complement
to A; the fixed router reveals a subset of that existing information and adds
no new top-1 answers beyond the two bases.

## 11. Oracle diagnostic analysis

**DIAGNOSTIC ORACLE — NOT DEPLOYABLE.** VALID labels are used only to ask whether
any model's existing top-1 label is correct. The oracle chooses truth when it
is available; if all members fail, it retains the first listed model's label.
It never invents a true class absent from every member's predictions.

Accuracy is the exact achievable top-1 selector ceiling. Macro-F1 is reported
for this explicit fallback oracle, **not claimed to be the maximum possible
Macro-F1** over all assignments of wrong labels. A separate loose Macro-F1 upper
bound assumes each class avoids all false positives except those forced by
unanimous wrong predictions; those simultaneous per-class minima need not be
jointly attainable. This distinction prevents an arbitrary fallback from being
mislabeled as a solved F1 optimization.

| Oracle members | Fallback | Accuracy ceiling | Oracle F1 at fallback | Loose F1 upper bound | Recoverable clients | Extra vs first | All fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A, V2 | A | 0.551 | 0.519729298 | 0.546154414 | 551 | 90 | 449 |
| A, Gym | A | 0.464 | 0.427118699 | 0.427616185 | 464 | 3 | 536 |
| A, FixedRouter | A | 0.508 | 0.473114527 | 0.482197497 | 508 | 47 | 492 |
| A, V2, Gym | A | 0.551 | 0.519729298 | 0.546154414 | 551 | 90 | 449 |
| A, V2, FixedRouter | A | 0.551 | 0.519729298 | 0.546154414 | 551 | 90 | 449 |
| A, Family | A | 0.550 | 0.491748713 | 0.548283856 | 550 | 89 | 450 |
| A, V2, Family | A | 0.621 | 0.571711218 | 0.639846605 | 621 | 160 | 379 |
| A, V2, Calibration | A | 0.577 | 0.537168629 | 0.571382920 | 577 | 116 | 423 |
| Jaime, NoneGate, Router, Family, Music, Calibration, Gym | Jaime | 0.580 | 0.515877552 | 0.585419853 | 580 | 119 | 420 |
| A, V2, NoneGate, Router, Family, Music, Calibration, Gym, Jaime | A | 0.638 | 0.583731983 | 0.657963831 | 638 | 177 | 362 |

All-model all-fail true classes: {'cloud': 37, 'gym': 48, 'insurance': 40, 'mobile': 37, 'music': 57, 'software': 46, 'streaming': 61, 'none': 36}.

An oracle gain demonstrates available answers, not learnable routing boundaries.
The all-experiment oracle especially benefits from bad NONE-heavy candidates
that happen to recover true NONE clients. That does not support deploying those
models. Their unique family recoveries and all-fail class counts matter more
than the oracle headline alone.

The V2/A **hard top-1 routing** Macro-F1 ceiling is at most .546154 on this VALID
sample, even under the loose bound. Merely choosing between those two answers
cannot reach .59 here. This bound does not constrain a probability mixture that
can choose a third class, a new predictor, or an unknown leaderboard cohort.
Adding the failed family arm makes another 70 clients recoverable; across all
nine predictions, 362 clients still fail, including 326 positives and 36 NONE.
Music (57/93 all-fail) and streaming (61/97) remain particularly unresolved.
These shared failures support seeking new identity/ranking evidence rather
than treating oracle diversity as a deployable ensemble recipe.

## 12. Leakage and selection-bias audit

| Check | Finding | Consequence |
| --- | --- | --- |
| Official metric, data labels, loader | Protected blobs and data hashes match; same complete-client eight-class scoring | No identified evaluator or label tampering |
| Time cutoff | Shared loaders and public V3 entry points reject missing/future timestamps | No identified post-cutoff features |
| Own-client family mapping | Five inner folds inside every base outer fit; explicit overlap rejection | Direct self-label encoding path excluded |
| Soft family posterior | Includes NONE implicitly through omitted posterior mass; unknown is zero | Representation/coverage confounding, not label leakage |
| Second-layer nesting | NONE/router/calibration/gym reuse fixed base OOF; base training can include gate-held labels for other rows | Their gate-CV headline is not fully nested end-to-end |
| Specialist / alpha selection | Same OOF labels choose policy/alpha and report best score | Selection optimism; no nested selection estimate |
| Family arm selection | Proper base cross-fitting; best arm chosen on its reported OOF | Sound feature isolation, remaining selection optimism |
| VALID freeze | Source puts prediction persistence before semantic label scoring; fixed controls distinguished | No identified new VALID threshold/alpha fitting |
| Historical VALID exposure | V2/discovery informed hypotheses; router report's “untouched” wording is too strong | No experiment has a research-wide untouched holdout |
| Freeze authentication | New protocol and result often committed together; local hashes are integrity checks, not independent timestamp attestations | Historical order source-verifiable, not independently certified |
| Refit/test | Jaime has a separate historical TRAIN+VALID submission phase; synthesis does not run it | VALID-scored predictions must remain pre-refit |
| Bootstrap | Fixed prediction vectors, repeated VALID, seven-way comparison | Conditional uncertainty, no search/training correction |

A gate-held row H is absent from its own base fit. However, a different OOF row
J used to train the gate can have base features produced by a model trained on
H's label. Therefore `y_H → base_for_J → features_J → gate → prediction_H` is
an indirect path. An independently reshuffled gate split does not break it.
This does not establish that the observed OOF gains are entirely caused by that
dependence; the separate VALID failures already reject their promotion.

A clean end-to-end estimate must hold H outside every base/map/transform fit
used by the gate-selection pipeline, construct inner OOF only within the
outer-fit clients, then score H once. Nested model selection and disjoint
calibration fitting are supported by the primary
[scikit-learn nested-CV documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
and [calibration API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
The specific indirect dependency above is an inference from this repository's
data flow, not a claim that either reference audited these branches.

### Paired uncertainty

10,000 paired client bootstrap replicates, seed 42, percentile 95% intervals,
unstratified sampling, the same resampled clients for both predictions, all eight
classes retained. Positive deltas favor the right-hand model. Training and
selection are not rerun inside bootstrap; familywise comparison is not corrected.

| Contrast | Point delta | Paired 95% interval |
| --- | --- | --- |
| FixedRouter minus A | +0.017855329 | [+0.000192, +0.035544] |
| Gym minus A | +0.003004133 | [+0.000000, +0.006817] |
| Jaime minus A | +0.000000000 | [+0.000000, +0.000000] |
| FixedRouter minus Gym | +0.014851197 | [-0.002358, +0.032198] |
| ensemble minus A | +0.005790062 | [-0.016868, +0.028313] |
| A minus V2 | +0.032561642 | [+0.002930, +0.062429] |
| Family minus A | -0.099736653 | [-0.131157, -0.068583] |
| NoneGate minus A | -0.067038152 | [-0.089777, -0.044886] |
| Calibration minus A | -0.006874801 | [-0.022339, +0.008422] |

Identical A/Jaime predictions give `[0,0]` for their difference; this is algebraic
identity, not certainty about hidden-set performance. Gym's interval reaches
zero and depends on five changed clients. The fixed router's lower endpoint
is only +.000192 and does not correct for its retrospective prominence among
multiple controls; its adverse TRAIN evidence remains. A versus V2 has a
positive interval consistent with its five positive original fold contrasts.
Intervals are uncertainty descriptions, not mechanical tests based on overlap
between separately estimated model intervals.

## 13. What V3 taught us

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| A. Family identity is the dominant missing signal | Supported among tested changes | A−V2: +.050910281 OOF / +.032561642 VALID; all five original folds improve |
| B. General temporal modelling is weak | Supported for tested aggregates/rules, not all temporal learning | Full−A: +.003601339 OOF / −.025616884 VALID; historical next-date proxies transfer poorly |
| C. Recurrence is useful mainly as evidence strength | Plausible, partly supported | V2 periodicity contributes modestly; hard next-stream systems fail; specialist recurrence gain is small and does not transfer |
| D. NONE versus positive is the major remaining challenge | Important but not sufficient | Repeated NONE transfer failures; many A positive errors remain wrong-family decisions |
| E. Family ranking is a major remaining challenge | Supported | Broad candidate coverage in discovery, multiple candidate families, low music/streaming/software F1 |
| F. Macro-F1 requires class-specific calibration | Not established | Tested correction improves OOF but loses VALID; equal weighting alone does not prove a particular correction |
| G. V2 and A justify routing | Complementarity yes; tested routing no | 90 V2-only VALID wins; learned routing and global OOF blend fail to exploit them |
| H. Gains are all noise and V3 should be abandoned | Rejected | Large replicated identity gain; small new gains are weak or reversed, which is different from identity being useless |

The strongest lesson is to distinguish **family identity, evidence reliability,
positive/NONE status and family ordering**. Client targets are not labels for
each historical merchant. More detailed target co-occurrence does not automatically
produce better semantic merchant identity.

## 14. Why current models still fail

V3-A has 539 VALID errors. Of 707 positives, it correctly identifies 285, sends
110 to NONE and sends **312 to the wrong positive family**. It misses 117 of
293 true NONE clients. Thus positive-family ranking accounts for more of A's
raw errors than positive↔NONE mistakes (312 versus 227). Macro-F1 weights make
these counts nonadditive as metric losses, but they rule out “just fix NONE” as
a complete answer.

The hard map labels a description by association with the client's future
target. Multiple recurring families coexist within clients, common aliases
are weakly pure, and mapping support is not stream truth. Discovery found
target-family presence in 659/707 positive clients under broad mapped history,
but mapped family evidence also in 291/293 NONE clients. Candidate presence is
therefore useful for recall and nearly useless as a standalone NONE gate.

TRAIN has 147,459 transactions and VALID 73,898. Median events/client are 72 and
73; outgoing card counts are 93,464 and 46,474. Distinct descriptions are 1,475
and 1,432. Only **0.9704%** of VALID outgoing rows have descriptions unseen in
TRAIN, versus roughly .50–.57% in ordinary OOF. This is a real difference but
too limited to assert that vocabulary novelty alone explains specialist score
collapse. Volume summaries also look broadly similar; these coarse
marginals cannot exclude conditional family/label relationship shifts.

At least three mechanisms remain confounded: model/map training sample size,
cross-fitted training versus full-map inference, and cohort/conditional shift.
OOF-to-VALID score changes demonstrate transfer failure but do not identify its
cause. Current evidence does not justify calling any one mechanism proven.

## 15. Most plausible path toward a large improvement

The approximate external leaderboard **.59 is NOT VERIFIED**. No access to its
predictions, split composition, evaluator trace or submission history was supplied.
It is a gap to investigate, not a valid local target or proof of another team's method.

| Explanation for an apparent .59 versus our local .42 gap | Evidence-based rank | Supporting / limiting evidence |
| --- | --- | --- |
| Better semantic family identity and official-target ranking | Highest modelling priority | Identity is our only large supported improvement; broad candidate presence but many wrong-family choices remain |
| More reliable NONE decision jointly with family ranking | High relevance, difficult transfer | 227 positive↔NONE errors; tested gates/calibration fail, so higher-capacity gating alone is not supported |
| Local/hidden cohort or split difference | Plausible, unverified for hidden split | Several observed OOF→VALID shifts; hidden distribution unavailable; cannot equate VALID and leaderboard numerically |
| Better recurrence reconstruction tied to the actual target | Possible, moderate evidence gap | History can contain useful recurrence; tested earliest-date/proxy objectives fail, event ground truth absent |
| Richer semantic canonicalization or external domain knowledge | Plausible, untested | Simple normalizer is a no-op; tested lexical clustering/char fallback fail; this does not rule out reliable semantic anchors |
| Class threshold tuning alone | Low for a large transferable gain | Bounded calibration reverses, conservative specialist never fires, gym gains three clients |
| Causal unlabeled/transductive learning | Low direct support, bounded future possibility | Discovery priors add only .000015864; semantic learning could differ but has not been demonstrated |
| Pseudo-labeling | Unsupported | Teacher errors/conditional shift can be reinforced; no successful evidence here |
| Our VALID overfitting | Known risk, incomplete gap explanation | Repeated selection usually makes local scores optimistic; it does not by itself explain a higher external score |
| Leakage-like exploitation by others | No evidence | No accusation or inferred explanation is justified from an aggregate number |

The most plausible substantive improvement is better semantic family evidence
and official-target family scoring **after** establishing that representations
transfer across fit sizes/cohorts. An oracle alone does not promise .59. A new
architecture must reduce shared errors, not merely rearrange the existing top-1
answers. No numerical hidden-leaderboard uplift is forecast from these data.

## 16. Integration compatibility matrix

These are structural compatibility assessments, not measured additive gains.
N=NONE gate, R=router, F=family posterior, M=music specialist, C=calibration,
G=gym protection, B=Jaime/global blend.

| Pair | Classification | Reason / required order |
| --- | --- | --- |
| N + R | PARTIALLY REDUNDANT | Both use V2/A scores and alter final selection; one may undo the other's NONE choice |
| N + F | UNKNOWN | F changes the base distribution and implicitly NONE evidence; existing N freeze is invalid on it |
| N + M | LIKELY COMPLEMENTARY structurally | NONE decision then positive specialist acts on disjoint decision stages; neither transfers as implemented |
| N + C | HIGHLY REDUNDANT | Both push A's OOF decisions toward NONE; likely compound observed failure |
| N + G | PARTIALLY REDUNDANT | G can override NONE into gym; ordering changes the result |
| N + B | UNKNOWN | Nontrivial blending changes gate inputs; alpha=1 is exactly neutral |
| R + F | UNKNOWN | Router correctness target and confidence geometry change with the base representation |
| R + M | PARTIALLY REDUNDANT | Both can replace the same positive-client A decisions; need overlap audit after fresh OOF |
| R + C | PARTIALLY REDUNDANT | Calibration changes the confidence comparisons and router features |
| R + G | HIGHLY REDUNDANT | G is a specialized confidence-dominant V2-vs-A route |
| R + B | HIGHLY REDUNDANT in purpose | Both select/mix V2/A evidence, though hard and soft decisions differ; alpha=1 is neutral |
| F + M | PARTIALLY REDUNDANT | Both strengthen text/family identity, especially music/streaming, using different representations |
| F + C | LIKELY COMPLEMENTARY structurally | Representation then offsets is well-defined, but frozen offsets cannot transfer to new probabilities |
| F + G | UNKNOWN | Upstream mapping may alter gym/NONE boundaries; no incremental gym evidence survives CV yet |
| F + B | LIKELY COMPLEMENTARY structurally | Global mixing can hedge an upstream representation, but old alpha selection does not apply |
| M + C | PARTIALLY REDUNDANT | Both redistribute music/streaming decisions; specialist gate depends on base argmax |
| M + G | LIKELY COMPLEMENTARY structurally | Separate destination classes but some clients overlap; ordering still needs specification |
| M + B | UNKNOWN | Blend changes which clients are eligible for specialist overrides |
| C + G | INCOMPATIBLE as frozen policies | C suppresses gym while G boosts it; conditions learned against raw A are no longer the same |
| C + B | PARTIALLY REDUNDANT | Both alter probability geometry; C must be reselected after blend |
| G + B | PARTIALLY REDUNDANT | Both restore V2 influence; globally blended score dominance differs from raw A |

Music and Jaime have exactly A's VALID predictions, so their measured VALID
complementarity is zero. Structural compatibility does not rescue that fact.
No combination score was tuned or used as a deployable candidate in this synthesis.

## 17. Recommended next architecture / experiments

**Run three experiments at most, in this priority order.** Each is a hypothesis
test with a stop condition. P0 is a prerequisite for interpreting later OOF
improvements. There is no V3.1 architecture approved for integration now.

### P0 — isolate map-size-induced transfer failure

- **Hypothesis:** the size/distribution difference between inner training maps
  and inference maps explains a material part of soft-posterior/NONE instability.
- **Exact baselines:** frozen A and Javier's frozen probability arm, each using
  the existing 300/depth4/.05/seed42 model, 75/25 heuristic mixture and outer
  seed-42 client folds. Include their ordinary full-map inference unchanged.
- **Exact modification:** on each outer held-client set, compare full outer-fit
  map inference to the average predictions obtained using the five corresponding
  inner-fit maps. Keep the fitted classifier, history features, heuristic and
  all parameters fixed. No gate or offset. Repeat the contrast with outer-fit
  training sizes 800, 1,200 and 1,600, using deterministic class-stratified subsets
  and the same held clients. This changes map inference size, not labels or targets.
- **Required evidence/features:** original hard/soft columns, per-map support,
  unknown share, positive posterior mass, NONE score quantiles and class counts.
- **Leakage-safe evaluation:** held clients excluded from every map and classifier;
  subset construction and contrast fixed before execution. All diagnostics and
  selection are TRAIN-only. No reused base OOF is used to claim nested gating.
- **Primary success metric:** paired held-client eight-class Macro-F1 contrast,
  plus predefined transfer diagnostics. Require ≥.010 pooled gain over the
  relevant full-map arm, positive contrast in ≥4/5 outer folds, and ≥50% reduction
  in its fit-size-related NONE-rate distortion without any class losing >.02 F1.
- **Stop/falsification:** if matching map size neither stabilizes scores nor
  improves held-client performance, reject map-size mismatch as the main cause;
  prioritize cohort/conditional-shift analysis. Do not lower these thresholds
  after seeing results. Stabilizing scores without F1 improvement is diagnostic,
  not a promoted model.
- **Expected upside/cost/risk:** potentially removes a common failure mechanism;
  no promised VALID gain. Roughly 30 base fits plus map predictions, estimated
  1–3 CPU-hours on this workstation, with caching and exact timing recorded.
  Main risk is changing bias through ensembling rather than correcting mismatch;
  record per-map and averaged effects to distinguish them. Direct leakage risk low.

### P1 — semantic identity independent of the client's future label

- **Hypothesis:** a small TRAIN-only reviewed set of unambiguous merchant-family
  anchors improves official-target family identification more than additional
  client-target posteriors.
- **Exact baseline:** unchanged A, or a single P0 winner only if its success
  criterion was met and frozen before this experiment; retain A as paired control.
- **Exact modification:** replace only the hard description-to-family assignment
  for predeclared high-precision semantic anchors; unknown/ambiguous names retain
  the baseline learned map. Preserve the 21 identity columns, classifier and
  75/25 mixture. No extra temporal features, model capacity or calibration.
- **Required features:** TRAIN descriptions, MCC, direction/type, currency-aware
  amount ranges used only to audit consistency; a versioned alias rationale with
  no client IDs or VALID-derived rules. No generic `premium plan` hard assignments.
- **Leakage-safe evaluation:** create anchors without target labels and without
  inspecting VALID names/errors; double-review semantics. If any learning uses
  target labels, redo it wholly within each outer fit. Existing task-wide VALID
  familiarity remains disclosed. Score all eight classes and unseen aliases.
- **Primary success metric/minimum:** ≥.015 pooled OOF Macro-F1 versus the frozen
  baseline, ≥4/5 positive folds, and positive-family accuracy improvement ≥.02
  with NONE F1 loss no worse than .01. Confirm across one prespecified second seed.
- **Stop/falsification:** reject if anchors cover too few previously ambiguous
  clients, improve only conditional-positive metrics, or fail either seed. No
  expanded alias dictionary derived from held-out errors in the same evaluation.
- **Expected upside/cost/risk:** identity has the strongest evidence and could
  reduce shared family errors. Cost roughly half a day of annotation/review plus
  30–90 minutes of model computation. Risk: incorrect semantic assumptions in
  synthetic descriptions. Direct leakage low if label-blind; human selection bias
  medium. This tests semantic identity, not unrestricted canonicalization.

### P2 — conditional family scorer using the official client target

- **Hypothesis:** a shared family-scoring model better compares a client's
  competing families than the current flat classifier, without making a new NONE gate.
- **Exact baseline:** the frozen successful identity representation from P0/P1,
  with raw A also reported. If neither succeeds, baseline remains A.
- **Exact modification:** generate all seven family candidates per client, with
  current count/share/alias evidence plus family-specific pre-cutoff recency,
  support and amount-stability features. Fit a low-capacity shared scorer to
  the **official positive-client target family**, not the next repeated merchant
  proxy. Use it only to choose among positive families when the baseline predicts
  positive; preserve baseline NONE decisions. No gate/blend/specialist stack.
- **Leakage-safe evaluation:** split by client before expanding candidates; fit
  every supervised map inside outer and inner client partitions. Any scorer
  parameter selection occurs in inner client folds; score each outer client once.
  Keep all seven candidates even if evidence is zero, exposing coverage failure.
- **Primary success metric/minimum:** ≥.015 complete eight-class held-client
  Macro-F1 and ≥.025 positive-client family accuracy, positive deltas in ≥4/5
  folds, NONE predictions identical by construction. Conditional ranking scores
  are auxiliary; no proxy metric substitutes for official F1.
- **Stop/falsification:** stop if the scorer only shifts false positives among
  families, fails a second prespecified seed, or requires a VALID-defined gate.
- **Expected upside/cost/risk:** directly targets 312 wrong-positive-family
  errors while avoiding the failed NONE correction. Estimated 1–2 development
  days and 1–3 CPU-hours. Risk is weak target-to-history attribution and increased
  variance; leakage risk medium unless candidate rows stay grouped by client.

A hierarchical NONE-first system, generic mixture of experts, broader router,
pseudo-labeling and causal transductive features are deferred paths, not additional
P3/P4 workstreams. Their expected upside is unquantified, compute/selection cost
is higher, and current evidence is weaker. They become worthwhile only if the
three tests above establish transferable evidence and a remaining specific need.

## 18. Explicit rejected directions

- Do not integrate the selected NONE gate, posterior identity or calibration.
- Do not integrate the music specialist's zero-action VALID rule.
- Do not promote gym protection from three recovered VALID clients after failed CV.
- Do not swap the learned router for its VALID-best fixed control and call that
  a TRAIN-selected result.
- Do not test unreported Jaime alphas on VALID; alpha=1 is the actual answer.
- Do not restore the full temporal V3 arm because it won original OOF.
- Do not combine individually failed components and assume cancellation or addition.
- Do not replace official labels with historical next-merchant proxy labels.
- Do not chase .59 with leaderboard probing, ID shortcuts or unverified leakage theories.

## 19. Risks

Repeated VALID use and TRAIN OOF model selection are already sunk limitations.
Fresh folds on the same studied TRAIN are useful diagnostics but cannot recreate
an untouched labelled cohort. Five folds and ~90–120 VALID positives per class
leave substantial sampling uncertainty. Seed and platform stability is not the
same as cohort stability. Bootstrap conditions on fixed predictions and ignores
retraining uncertainty. Frozen runtime hashes demonstrate integrity, not author
intent or historical chronology. Ordinary class probabilities are mixture scores,
not established calibrated posterior confidences.

The soft-map complement-to-NONE argument is exact algebra for known descriptions;
its role in the measured failure is an inference. The map-size hypothesis is
also unproved. The next round is designed to distinguish causes rather than
retrospectively explain every error with a convenient story.

## 20. Exact next-step plan

1. Review this synthesis and retain the frozen A/V2 references and evaluator.
2. Before any new fits, write the P0 contrasts, subsets, failure criteria and
   output schema; pin source/data hashes. Do not incorporate branch code into
   production as part of that protocol.
3. Execute P0 entirely on TRAIN; publish negative results as well as positives.
   Report per-class scores, score distributions, map support/unknown shares and
   paired changes, separating sample-size from map-inference contrasts.
4. Proceed to P1 only with its semantic review protocol frozen. Proceed to P2
   only after the identity input is frozen; no parallel sweep of alternative gates.
5. If one candidate meets its preregistered TRAIN criteria, prefer a genuinely
   new labelled cohort for confirmation. If unavailable, a single frozen reused
   VALID diagnostic must retain that limitation and cannot reopen parameter search.
6. Only after a successful complete-pipeline OOF test consider a separate V3.1
   integration task. No production replacement, merge or submission is authorized
   by the research recommendation itself.

## 21. Reproduction commands

From the synthesis root in the installed project environment:

```powershell
git fetch --all --prune
git status
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -10
git branch -a

# Fresh immutable snapshots; no checkout/merge/cherry-pick of experiment code.
python scripts/reproduce_v3_seven_way.py base
python scripts/reproduce_v3_seven_way.py none family music
python scripts/reproduce_v3_seven_way.py calibration blend
python scripts/reproduce_v3_seven_way.py router gym
python scripts/analyze_v3_seven_way.py

python -m pytest -q
ruff check .
ruff format --check .
python -m pre_commit run --all-files
```

The wrapper records exact executable, arguments, snapshot commit, source hashes,
input hashes, exit code and elapsed seconds. Re-running into a frozen result
directory is deliberately rejected by author scripts. Base must finish before
calibration/blend/router/gym. The latter two use the portable diagnostic adapter
described above. Running names concurrently is optional; wall times with
contention are not performance benchmarks.

Author-native commands remain in each pinned handoff, and can be executed in a
compatible snapshot/environment. No source/evaluator edit is necessary to
reproduce the predictive computations. For router-native CLI, use Python 3.12;
gym-native provenance path keys require a consistent platform/path spelling.
This synthesis does not alter those author files to make them pass.

Quality checks used the repository `.venv` executables:

- `python -m pytest -q`: **90 passed** (87 existing plus three diagnostic tests).
- `ruff check .`: **passed**.
- `ruff format --check src scripts tests`: **passed**, 68 files formatted.
- `python -m pre_commit run --all-files`: **passed**, including all six staged
  research files. No test or evaluator rule was weakened.
- `ruff format --check .` with standalone Ruff 0.16.8 **crashed** in its diagnostic
  renderer: `Annotation range 0..514 is beyond the end of buffer 512`. The root
  command's failure is retained as a tooling limitation; its precise triggering
  file was not established. Scoped source checks and the repository-pinned Ruff
  pre-commit formatter pass. Unrelated files were not edited to suppress it.

Targeted tests against all seven pinned snapshots also passed: NONE 4, router 3,
family 10 (seven inherited), music 2, calibration 5, gym 16 and blend 19, totaling
59 cases. The initial snapshot harness omitted the parent of three temporary
directories; creating that parent and rerunning the unchanged tests resolved
those setup errors. Original failure logs remain alongside the successful runs.
The new tests verify client alignment/change accounting, oracle recoverability
and equivalence of the bootstrap's fixed-eight-class F1 kernel to the official
scorer. No model-training artifact is part of a test fixture in this commit.

Local artifacts: `outputs/metrics/v3_seven_way/{snapshots,runs,audit,analysis}`.
No datasets, prediction tables, model artifacts, caches or copied branch source
are included in the research commit.

## 22. Final ranked conclusions

1. **Retain V3-A as the research base.** Cross-fitted hard identity remains the
   only large improvement with consistent OOF and VALID evidence.
2. **Investigate representation/score transfer before adding decision layers.**
   Multiple apparently stable TRAIN improvements fail in the same direction.
3. **Preserve V2's unique information for research.** It recovers 90 A errors on
   VALID, especially gym/software, but neither the learned router nor the global
   blend demonstrates a reliable way to use it.
4. **Target family ranking, not only NONE.** A's largest raw error group is wrong
   positive-family selection. Music/streaming/software remain weak.
5. **Treat the fixed router and gym gain as diagnostics.** They identify possible
   useful decisions, with weak or adverse OOF support and reused-VALID selection risk.
6. **Do not mistake unchanged results for complementary components.** Jaime's
   alpha=1 and the specialist's zero overrides are the same A predictions on VALID.
7. **Choose ONE MORE TARGETED ROUND, beginning with P0.** The evidence supports
   continued identity research, not immediate seven-component integration and
   not abandonment of V3.
