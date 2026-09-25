# V4 integration audit

Integration branch: `integration/v4-synthesis`. Frozen baseline:
`051ce64a7cf0ab999f8aacb81fa405d5fa0257cf` (`origin/baseline/v4-frozen`).

## Evidence levels and initial state

**VERIFIED** means inspected code, an executed check, or independently rescored
local probabilities; its scope is stated. **REPORTED** means author evidence
that was not rerun. **DIAGNOSTIC** includes privileged or proxy measurements.
**HYPOTHESIS** is an explanation not established experimentally.

Initial branch and HEAD exactly matched the requested branch and baseline. No
tracked local changes existed. Seven cached remote refs exist, all with extra
commits and merge-base equal to the frozen SHA. None is MISSING/INCOMPLETE.
`git fetch origin --prune` failed because `.git/FETCH_HEAD` is read-only;
`git ls-remote` also failed because outbound GitHub access is unavailable.
Thus branch tips are **cached**, not certified as the latest remote tips.
Pre-existing inaccessible temporary directories were preserved. No branch was
merged, no checkout was performed, and main was not modified.

## Ledger

V = VERIFIED; R = REPORTED; D = DIAGNOSTIC. All scores below are official
eight-class Macro-F1 unless a proxy is explicitly named. Different corruption
protocols must not be compared as if they were the same benchmark.

| Branch / author | Hypothesis | TRAIN OOF | Corrupt/stress | VALID reused | Per-class | Runtime | Leakage risk | Stability | OOF probs | Verdict |
|---|---|---:|---|---:|---|---|---|---|---|---|
| exp/v4-shift-corruption-ginestar | Paired merchant degradation exposes fragile identity/grouping | A .459794 | V A .452061/.432375/.373384 mild/medium/severe | No model selection/evaluation on VALID labels | R severe cloud −.200228, streaming −.162725, music −.157120 | V 1219.18 s; R 1781.3 s full suite | Low: views after client split; labels untouched | 14 tests pass; SHA-keyed randomness; one corruption seed | Per-fold probabilities; pooled labels | **STRONG** infrastructure |
| exp/v4-direct-robust-javi | Whole-description + chars + MCC with weighted corruption augmentation | V .507420 | V local medium .478119, severe .459898; R severe .461383 | R .076353 | R 989/1000 NONE; two classes never predicted | V 80.47 s selected recipe; R ~15 min grid | Fold-local vocabulary/IDF; no direct label leakage found | **Hash-seed-dependent corruption**; not fully deterministic | V regenerated 2000 × 8 | **NEGATIVE** |
| exp/v4-merchant-intelligence-christian | Text/behavior retrieval and weak family summaries reduce identity loss | R B .482502; C .480770 | D top-8 recovery .794 masked, .060 fragmentation; not downstream F1 | R B .337778; C .329612 | OOF gain concentrated in NONE; VALID positive hits 285→174 | R 877.9 s; RAM not measured | Label maps cross-fit; unsupervised index includes outer held histories (transductive) | B wins 3/5 folds; no unseen-index control | Expected exports absent locally | **MIXED**; reject predictor |
| exp/v4-soft-candidate-santiago | Soft eight-family scorer resolves ambiguity using nested OOF evidence | V CatBoost .542662; linear .499181 | V event-thinning .519199 vs A .454978 (rescored) | V saved .400662 | OOF all classes improve; VALID only insurance/streaming F1 improve | V 63.16 s heads + one evidence fold; full original runtime unavailable | Nested generators exclude outer hold; 8 candidates; no ID feature | OOF gain all 5 folds; transfer regression; runner hash discrepancy | V 2000 × 8 with sources/candidates | **MIXED**; strongest TRAIN candidate |
| exp/v4-survival-laura | Causal pseudo-cutoffs learn recurrence/hazard and NONE mass | V hazard support .423917; best hybrid .457216 | No common merchant-corruption score | R hazard support .311002; hybrid median .389100 | Proxy AUROC ~.614; not official family accuracy | V 200.78 s, cached baseline | Client groups across cutoffs; features < cutoff; labels future within observed TRAIN | No winning variant; annual mirror lacks training support | V regenerated all 20 temporal/hybrid variants | **NEGATIVE** |
| exp/v4-self-supervised-esteban | Label-free SVD + ridge denoising supplies robust representations | V .439100 | R raw .3886 vs denoised .3879; protocols differ from Ginestar | R .3219 | R all eight F1 values in handoff; OOF independently rescored | V 30.08 s frozen head; R 37 s full experiment; CPU | Transductive label-free encoder; client-held downstream head; auxiliary probe splits events | Denoising gain .0020 in L2; no downstream justification | V regenerated 2000 × 8 | **NEGATIVE** |
| product/v4-agent-demo-jaime | Real predictor, evidence tools and conditional investigation | Not a competing model | Not applicable | Same V3-A model | Scores and limitations visible | Current build/HTTP blocked by dependencies | Original TRAIN+VALID policy replaced by TRAIN-only; no TEST labels | Unit tests pass; live HTTP/frontend not verified here | Predictor runner artifacts | **PROMISING** product |

## Source and experiment audit

### Ginestar

Ported only `corruption.py`, `corruption_evaluation.py`, their tests and runner.
The suite preserves every non-description field, labels, timestamps and client;
clean is an exact deep copy. TRAIN-only dictionaries precede held-out views.
Whole streams anchored by outgoing payments include matching refunds/transfers.
Opaque masking preserves groups; temporal aliases split groups; explicit
collisions require equal MCC/type/currency/direction. Generic/missing sentinels
can still create incidental cross-context collisions in downstream description
grouping: disclosed limitation, not altered retrospectively.

This is inference stress, not augmentation. No severity was changed. Severe
unseen-name frequency is much larger than real VALID/TEST input diagnostics.
The integrated runner writes its report inside the output directory, preserving
the author's reviewed handoff, and supports the integration branch.
The final wrapper enumerates protected files from the baseline Git tree, so new
integration modules do not become false baseline-change failures after commit.
That source-guard-only correction followed the full run; its executed wrapper
is preserved in ignored `stress/executed_runner.py`. Corruption algorithms,
parameters and model code are byte-identical to the evaluated versions.

### Javi

Verified ComplementNB(alpha=1) and logistic(C=1), six representations, clean
fits and two augmentation modes. Fit vocabulary/IDF/generic counts inside outer
TRAIN. Normalized views give total weight one/client. Reproduced only the frozen
NB/R4/normalized recipe, not the 20-arm selection grid. Clean/medium reproduce;
severe fold 3 differs. A two-process control with PYTHONHASHSEED 0/1 establishes
different tied generic vocabularies and corrupted-view hashes: `Counter.update`
on a set plus `most_common` lacks deterministic tie-breaking. Do not patch this
rejected branch and relabel the revised experiment as the original result.
The handoff's statement that VALID is independent is incorrect in project context.

### Christian

Read fingerprints, text/MCC/numeric blocks, behavior/text KNN and bounded graph
unions. Cross-cluster compatibility is checked across all pairs, limiting graph
chaining; zero incompatible pairs is a construction invariant, not identity
ground truth. One weak-label vote/client, supervised self-client rejection and
label-independent inner folds are present. Nine focused tests pass. No downstream
rerun: local OOF/index artifacts are absent, graph build is costly and no candidate
is selected for promotion. Reported aggregates remain REPORTED.

No material candidate-recall gain: @3 exact and merchant both .862438, versus
.8639 when exact vacancies are filled by a prior. Fragmentation recovery fails
despite low abstention. No evidence supports adding this resolver to the ranker
or survival model in this synthesis. Embeddings/index are not copied to Git.

### Santiago

Exactly eight deterministic candidate rows/client including NONE; official
client target defines binary relevance. Family maps and baseline probabilities
are nested inside outer TRAIN; metadata validates disjoint generators. Scorers
exclude IDs, string candidates and targets from features. Refit all five CatBoost
heads: probability differences from stored values ≤1.12e−16. Regenerated all
features for the 400 held clients of fold 1: maximum difference **0**.

All source/model/TRAIN hashes match the cache except the experiment runner hash.
The cause of that mismatch is unresolved; do not call the entire historical
artifact chain bit-identical. Full generator reproduction for every inner fold
was not repeated. Candidate availability 100% is tautological with eight rows;
positive target evidence recall .950107 OOF is different. Rank accuracy .563 is
accuracy, not Macro-F1 .542662. An omniscient eight-candidate selector is not a model.

### Laura

Reproduced all temporal and predefined 75/25 hybrid arms, using original branch
functions and baseline OOF with exact fold-client checks. 48,522 snapshots,
2,000 clients; per-client snapshot weight sums to one. Three effective cutoffs
(January/April/July 2025); earlier three have no history. Features exclude target
columns and discard official-cutoff recency/due fields before recalculation.
Targets use [cutoff, cutoff+90d); no client crosses folds. NONE is accumulated
survival, not a separate tuned gate. Competition is symmetric within a bucket,
not a hard earliest-stream rule. This does not establish calibrated independence
between streams or reliable official family forecasts. No structural integration.

### Esteban

Pretraining rejects challenge labels. 80k unlabeled events plus TRAIN histories,
1,690 unique descriptions, 32 SVD dimensions, ridge denoiser; no GRU/TCN/Transformer
was trained and none is needed here. Reproduced the frozen clean-embedding-plus-
stats head. Full-TRAIN encoder makes downstream OOF transductive, which is
disclosed; its encoder is not outer-fold-local. The gap probe is event-stratified,
not client-grouped, and is only a weak auxiliary diagnostic. The runner loads
VALID labels early via `load_ubs_data`, although inspected model selection uses
TRAIN scores only; our reproduction reads TRAIN/unlabeled only. No embeddings
are integrated or versioned.

### Jaime

Preserved selected API, frontend, evidence, bounded tool router, scripts and
pitch assets; did not copy the branch's historical-report changes. Adapted
preparation to consume the final runner's exact serialized TRAIN-only model.
Updated source lock, runtime fit-scope display and stale case/refit claims.
Eight tools are implemented; only a conditional deterministic router is present.
No LLM or measured economic impact is claimed. API/build reproduction is blocked
by missing FastAPI/npm dependencies, recorded separately from passing unit tests.

## Traceability and reproduction

The full original handoffs and their small committed aggregate JSONs are
preserved unchanged under `reports/handoff/v4_*`. Branch SHA/commits, changed
files, verified checks and final evidence are in `v4_final_decision.json`.
Only Jaime adds dependencies (`demo` extra plus pinned frontend lockfile); six
research branches reuse existing numerical/ML dependencies. Javi's reported
Python 3.14 is outside the project's supported range; this reproduction uses
Python 3.12.0, NumPy 2.3.5, pandas 2.3.3, sklearn 1.9.1, CatBoost 1.2.10.

```powershell
python scripts/audit_v4_branches.py --snapshots --tests
python scripts/experiments/v4_synthesis_checks.py ranker
python scripts/experiments/v4_synthesis_checks.py direct
python scripts/experiments/v4_synthesis_checks.py svd
python scripts/experiments/v4_synthesis_checks.py survival
python scripts/experiments/v4_complementarity.py
```

The ranker check needs its original ignored nested feature artifacts. Without
them, execute its original OOF command inside the exported source snapshot with
an absolute `--data-dir`, then point the audit at that cache; do not fabricate
missing probabilities. Exported source snapshots remain ignored and do not
change the integration branch. Model imports/tests execute each branch in its
own subprocess. Existing V1/V2/V3 runners and the official evaluator are unchanged.
