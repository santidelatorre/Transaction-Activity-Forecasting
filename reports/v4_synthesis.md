# V4 synthesis — retain V3-A, TRAIN-only

## 1. Executive summary

**FINAL_MODEL: V3-A**, version `v4-synthesis-v3a-train-only-1`.
**FINAL_REASON:** none of the six research lines establishes a transferable,
stable improvement sufficient to replace the frozen baseline. V4's deliverable
is a defensible recipe, reusable stress infrastructure and an adapted product;
there is **no claimed predictive gain**. Delta versus baseline is exactly zero.

The strongest new individual on TRAIN is Santiago's soft candidate scorer:
Macro-F1 **0.542662209**, versus baseline **0.459793827**. Its saved, independently
rescored reused VALID result is **0.400662**, below **0.424111098**. A fixed equal
average does not beat it on TRAIN or missing-history stress. No model weights,
thresholds or parameters were selected from VALID in this integration. Previously
reported reused VALID is explicitly part of the promotion-rejection context;
this is not an untouched model-selection exercise.

## 2. Frozen baseline and scope

Baseline SHA `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`; branch remains
`integration/v4-synthesis`. Eight classes, official target, fixed-eight-class
Macro-F1, pre-2026-01-01 history, 90-day horizon. IDs only align/group clients.
No TEST labels, evaluator modification, whole-branch merge or main integration.

Original V3 research synthesis/protocol, relevant NONE-gate/seven-way V3 handoffs,
README promotion, runner, model, official evaluator, submission tests and all
seven V4 handoffs were read. Frozen V3-A is candidate A: 167 history/identity
features, CatBoost 300/depth4/.05/balanced/seed42, five cross-fit mapping folds,
fixed 75% numeric plus 25% periodicity. Historical B/AB models do not choose A.
The retained implementation still fits historical diagnostic arms internally;
avoiding that overhead is an optional future optimization, not part of this change.

## 3. Audit of seven branches

| Author | Verdict | Integration decision |
|---|---|---|
| Ginestar | STRONG infrastructure | Port unchanged corruption algorithms/evaluator/tests; adapt runner output/branch support |
| Javi | NEGATIVE | Reject predictor; clean OOF succeeds, VALID collapses; corruption has hash-seed instability |
| Christian | MIXED | Useful retrieval diagnostics; reject downstream model and structural composition |
| Santiago | MIXED | Best TRAIN model; keep research artifacts, reject promotion after transfer failure |
| Laura | NEGATIVE | Causal proxy implementation verified; no official-target improvement |
| Esteban | NEGATIVE | SVD is adequate simple control; denoiser/deep escalation unsupported |
| Jaime | PROMISING product | Port compatible product and adapt to TRAIN-only final model; HTTP/build verification blocked |

All seven cached tips contain additional commits with the correct merge-base.
Full ledger, code/leakage review, handoffs, dependency and reproduction scope:
[integration audit](v4_integration_audit.md). Exact branch heads/commits/files are
also in [machine-readable decision](v4_final_decision.json).

## 4. Complete metric comparison

V = VERIFIED reproduction/rescoring, R = REPORTED, D = DIAGNOSTIC. A dash means
unavailable/not evaluated, never zero. Author-specific stress levels differ.

| Model | Official TRAIN OOF | TRAIN stress | Reused VALID | Delta VALID vs A |
|---|---:|---|---:|---:|
| V3-A | V .459793827 | Common suite: see section 5 | V .424111098 | .000000000 |
| Javi frozen NB/R4/normalized | V .507419937 | V medium .478119212, severe .459898496; R severe .461382746 | R .076353284 | −.347757813 |
| Christian B additive | R .482502095 | D known-stream retrieval only | R .337777577 | −.086333521 |
| Christian C replacement | R .480769622 | D known-stream retrieval only | R .329611703 | −.094499395 |
| Santiago raw family | V .254798 | V .268526 event thinning | V .192719 | −.231392 |
| Santiago logistic | V .499181 | V .483360 event thinning | — | — |
| Santiago CatBoost | V .542662209 | V .519199 event thinning | V .400662 | −.023449 |
| Fixed A/ranker average | V .540059768 | V .506080747 event thinning | Not evaluated | — |
| Laura hazard support | V .423917074 | — | R .311002 | −.113109 |
| Laura hybrid median-gap | V .457216168 | — | R .389100 | −.035011 |
| Laura hybrid seasonal/support | V .455288820 | — | R .418457 | −.005654 |
| Esteban SVD + stats | V .439100310 | R denoised .3879 vs raw .3886; different heads/views | R .3219 | approximately −.1022 |

All 20 reproduced survival/temporal/hybrid arms, complete fixed-class metrics,
prediction distributions and confusion matrices for available OOF are in the
decision JSON. Christian aggregates and absent artifacts remain REPORTED; its
full expensive graph/model experiment was not repeated. The 20-arm Javi search
and all SVD head-selection alternatives were not repeated. This synthesis does
not reinterpret unexecuted experiments as measured results.

## 5. Shift/corruption findings

**VERIFIED:** Ginestar's pooled A clean/mild/medium/severe scores reproduce .459793827,
.452061182, .432375381, .373383822. Full five-fold reproduction and precise
status/timing are recorded in the generated verification appendix below.
Severe drop is .086410005 (18.79% relative). V2 severe .394071435 is more resilient
than A in this extreme, while A leads on clean/mild/medium. This does not select
a severity-specific route or change blend weights.

Severe class losses are largest for cloud (.200228), streaming (.162725), music
(.157120). NONE predictions rise 304→1077 against fixed support 597. The isolated
opaque stream mask leaves history-control unchanged but damages identity A;
fragmentation particularly harms description-grouped history features. These
ablations are not an additive causal decomposition.

Unsupervised author diagnostics: severe unseen descriptions .7767 versus VALID
.0117 and TEST .0099; median same-currency within-stream amount CV severe .2728
versus VALID .4487 / TEST .5442. The stress suite is a **robustness simulation**,
not a calibrated UBS generator or proof of the cause of the real cohort gap.
Only one corruption seed is studied. No retrospective parameter adjustment.

## 6. Per-class comparison

All F1, fixed order. Baseline and ranker OOF/VALID are independently rescored.

| Class | A OOF | Ranker OOF | A VALID | Ranker VALID |
|---|---:|---:|---:|---:|
| cloud | .463964 | .515222 | .481928 | .400000 |
| gym | .469526 | .518337 | .442478 | .413146 |
| insurance | .474438 | .604255 | .427273 | .474747 |
| mobile | .450704 | .513447 | .468085 | .393258 |
| music | .395062 | .480198 | .323810 | .270833 |
| software | .429224 | .527964 | .334802 | .330097 |
| streaming | .444934 | .498896 | .306569 | .317241 |
| none | .550499 | .682977 | .607945 | .605974 |

Final model inherits A, so its per-class delta is zero. Baseline VALID prediction
counts: 77, 105, 121, 131, 117, 123, 40, 286; TRAIN counts: 254, 253, 275, 235,
207, 243, 229, 304. Exact confusion matrices and precision/recall are preserved
in JSON, not inferred from rounded tables.

## 7. OOF stability

| Fold | A | Ranker CatBoost | Christian B (R) |
|---|---:|---:|---:|
| 1 | .422837 | .497450 | .469847 |
| 2 | .449722 | .576839 | .518279 |
| 3 | .476690 | .539670 | .464929 |
| 4 | .474125 | .547994 | .493583 |
| 5 | .465166 | .547751 | .459112 |

Ranker gain appears in all five folds and its descriptive paired bootstrap
interval [+ .062780, + .102493] (R) is positive. That interval does not correct
model selection or refit uncertainty. Baseline mean fold score .457708 is not
the pooled score .459794. Fold standard deviations are descriptive, not
independent confidence intervals. Javi's severe repeatability additionally fails
across hash seeds; the deterministic shared suite does not have this failure.

## 8. VALID comparison

Final TRAIN-only baseline evaluation reproduces .424111097737, accuracy .461.
VALID was already reused repeatedly. Existing ranker probabilities are rescored;
no logistic VALID attempt, average VALID score, new gate or corrective threshold
is tried. Javi/Christian/Laura/Esteban VALID numbers are author reports, not fresh
independent evaluations. A fresh baseline evaluation checks reproducibility,
not independence. No leaderboard improvement is claimed.

## 9. Complementarity

One aligned table contains 2,000 unique TRAIN clients, official targets, folds
and eight ordered finite probabilities for every available principal model.
Raw IDs/probabilities remain ignored in `outputs/metrics/v4_synthesis/aligned_train_oof.csv`.
Baseline probabilities equal the original fold A artifacts. Aggregated matrices
are included in the decision JSON.

| Candidate vs A | Disagreement | Error correlation | Corrects A | Regresses A |
|---|---:|---:|---:|---:|
| Santiago CatBoost | .2735 | .619278 | 289 | 105 |
| Santiago logistic | .1810 | .737598 | 185 | 82 |
| Javi | .2910 | .567138 | 298 | 144 |
| Esteban | .3860 | .471425 | 245 | 281 |
| Laura hazard support | .3120 | .531771 | 202 | 264 |
| Laura hybrid median-gap | .1510 | .780161 | 103 | 116 |

Christian B reportedly corrects 175 and regresses 100, disagreement .181; its
OOF files are absent locally, so it is not silently inserted into the aligned
table. Per-class corrected/regressed counts, probability correlations,
Jensen–Shannon distances and pairwise oracles are available in JSON.
**DIAGNOSTIC oracle** chooses a correct model when either is right, otherwise A:
it uses labels, is not deployable and is not an optimized Macro-F1 upper bound.
Error diversity alone does not identify a valid inference-time selector.

## 10. Ensemble/stacking experiments

Exactly one new composition: the predeclared equal average of A and strongest
TRAIN ranker. OOF .540059768 is below ranker .542662209; event-thinning .506080747
is below .519199. Rejected on TRAIN evidence, without opening average VALID.
No weight search, calibration, new features or threshold adjustment.

**Regularized stacking was not executed.** The only credible new finalist has
unresolved feature-size/cohort transfer; other branches are weaker or collapse.
There is no pair supported for promotion. A valid additional stack assessment
would require nested OOF ranker predictions inside each outer fit. The saved
inner *features* are insufficient for that additional supervised layer. Merely
re-splitting globally generated ranker OOF would allow indirect outer-label
dependence through the base fits; it is not used for selection or a claimed
improvement. The cost of another nesting level is not justified by the rejected
average and existing transfer failures. No stack weights/stability claims exist.

## 11. Structural integrations attempted

None. Merchant evidence does not materially improve candidate recall and
fragmented-alias recovery is poor. Survival improves a temporal proxy but loses
the official target. Neither establishes a specific defect it can reliably
correct in the ranker. Combining them would be speculation, not an evidence-led
composition. Shared stress infrastructure and product adapters are integrations,
not new predictive features.

## 12. Negative results

Clean OOF improvements do not establish transfer. Javi predicts 989/1000 NONE
on reported VALID. Christian's OOF gain is mostly NONE while positive-family
hits fall on VALID. The soft ranker has complete candidate availability but
still misranks families and shifts NONE decisions. Survival's annual mirror
lacks historical support; its best hybrid remains below baseline. SVD denoising
barely changes representation distance and adds no useful head performance.
A deep model was not trained. These failures remain visible, not merged away.

## 13. Leakage and provenance audit

Official evaluator unchanged; pre-cutoff loaders, disjoint client folds,
cross-fit mappings, no ID feature, no TEST labels. All author source changes
were inspected without merging. Santiago's generator/scorer nesting is valid;
one runner hash mismatch remains disclosed despite exact model-source hashes,
feature regeneration and scorer reproduction. Christian and SVD OOF are
transductive in their unsupervised components; those claims cannot be called
inductive unseen-client evidence. SVD's auxiliary gap probe is not client-CV.

VALID is secondary reused evidence. No VALID fit, blend/threshold selection or
oracle route is added. The challenge documentation available locally does not
clearly authorize final VALID-label refit; external confirmation could not be
retrieved. Consequently submission and demo remain **TRAIN-only**. The old
TRAIN+VALID V3 runner is preserved historically but is not the V4 entry point.

## 14. Final model decision

`FINAL_MODEL = V3-A`; `FINAL_REASON = retain the best supported frozen predictor;
no convincing transferable V4 gain`; `REJECTED_ALTERNATIVES = Javi direct,
Christian B/C, Santiago ranker/linear/raw and equal average, Laura temporal/hybrids,
Esteban SVD/denoising; stacking and structural compositions not justified`.

TRAIN winner and deployment recommendation are explicitly different. Retaining
baseline has zero delta and does not claim that complexity improved the model.

## 15. Final architecture

Official histories → frozen V2 historical representation + cross-fitted family
identity → unchanged V3-A CatBoost/periodicity scores → eight-class argmax.
No postprocessing gate. `ubs/v4.py` supplies recipe metadata, checked probabilities,
client separation and submission contract; `scripts/run_ubs_v4.py` provides OOF,
VALID and submission phases. Source/data/environment hashes guard artifact reuse.

## 16. Submission recipe

```powershell
python scripts/run_ubs_v4.py --phase valid
python scripts/run_ubs_v4.py --phase submission
```

The same serialized TRAIN model supplies both phases. Submission must contain
exactly 1,000 unique sample IDs, exact two-column schema/order, allowed labels,
no NaNs and the sample's row order. Artifact path/hash/status are in the
verification appendix and `v4_final_decision.json`. No CSV is uploaded.

## 17. Product/demo status

Jaime's predictor adapter, evidence/recurrence views, API, React frontend,
conditional tool router and pitches are ported selectively. Preparation consumes
the final runner artifact; it no longer fits TRAIN+VALID. Version and fit scope
come from metadata; sample stories are selected anew without TEST labels.

The adapter checks live scores against the runner for the same client. Bundle
parity and tests are recorded below. HTTP/API and frontend build cannot be
claimed verified here: FastAPI is absent and npm dependencies could not be
downloaded. Original author HTTP/smoke success is REPORTED only. No screenshot,
video or live end-to-end UI result is invented. [Demo instructions](../DEMO_README.md).

## 18. Jury mapping

Only the five supplied criteria are used; no weights or extra criteria invented.

| Criterion | Actual evidence and boundary |
|---|---|
| Technical Functionality & AI | Reproducible official pipeline, nested OOF audit, source/model hashes, stress suite and contract tests; no V4 predictive gain |
| User Experience | Real prediction/evidence frontend and API source; author demo evidence exists, current HTTP/build still unverified |
| Agentic Depth | Eight allowlisted read tools, conditional selection, budget and stopping decisions; deterministic policy, no connected LLM |
| Originality & Fun | Investigation of degraded identity and candidate ambiguity; no claim that CatBoost itself is novel |
| Potential & Market Impact | Hypothesis: support review of recurring commitments with evidence; no measured financial/customer benefit |

## 19. Limitations and integration readiness

Repeated TRAIN/VALID research induces selection optimism. Synthetic stress does
not reproduce behavior/amount/cohort shifts or provide hidden TEST performance.
Weak client labels are not stream identities; probabilities are uncalibrated.
One stress seed; incomplete artifact provenance in the ranker runner; no full
Christian reproduction; no extra seeds/deep models/combinatorial search.
Peak RAM/GPU measurements are absent unless explicitly attributed to authors.

Git metadata is read-only and GitHub is unreachable, so fetch, final commit and
push cannot be completed in this environment. No PR is opened and no main merge
is performed. The deliverable is the intentional working-tree change, auditable
reports and ignored local artifacts; it is not falsely described as a published
branch. Formatter/pre-commit/tooling limitations are recorded below.

## 20. Reproduction commands

```powershell
python scripts/audit_v4_branches.py --snapshots --tests
python scripts/experiments/v4_synthesis_checks.py ranker
python scripts/experiments/v4_synthesis_checks.py direct
python scripts/experiments/v4_synthesis_checks.py svd
python scripts/experiments/v4_synthesis_checks.py survival
python scripts/experiments/v4_complementarity.py
python scripts/experiments/v4_shift_corruption.py --external-inputs --output-dir outputs/metrics/v4_stress_new
python scripts/run_ubs_v4.py --phase oof --output-dir outputs/metrics/v4_oof_new
python scripts/run_ubs_v4.py --phase valid
python scripts/run_ubs_v4.py --phase submission
python scripts/prepare_product_demo.py
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pre_commit run --all-files
python scripts/quality_gate.py
```

Use `.venv/Scripts/python.exe` in this environment. Original branch reproduction
commands and expected output paths are in their preserved handoffs. For each new
full run, use a fresh output directory. The ranker replay requires its original
nested feature cache or regeneration from its exported source. Outputs, datasets,
models, embeddings, node_modules, caches and credentials must remain outside Git.

<!-- VERIFIED_APPENDIX -->

## Verification appendix (generated from local evidence)

Final VALID: **0.4241110977365116**; TRAIN OOF: **0.45979382686863396**.
Submission: `outputs\metrics\ubs_v4_final\submission_v4.csv`; SHA-256 `cd726195b0e6d7443ca161b446da6cdf8f07ac3b23df2d06f9ce57a3fa1ed7ee`.
Validated: 1,000 rows/unique IDs, exact schema/order/classes and no NaNs.

**VERIFIED complete common stress reproduction:**

| View | Macro-F1 | Drop vs clean |
|---|---:|---:|
| clean | 0.459793826869 | 0.000000000000 |
| mild | 0.452061181537 | 0.007732645332 |
| medium | 0.432375380867 | 0.027418446001 |
| severe | 0.373383822058 | 0.086410004811 |

Runtime: 1219.2 s, CPU; no peak RAM measurement.

Quality checks and product parity:

```json
{
  "tests": {
    "pytest": {
      "exit_code": 0,
      "passed": 116,
      "skipped": 2,
      "reason_for_skips": "FastAPI not installed; opt-in HTTP real-data test not enabled. Separate adapter parity executed."
    },
    "ruff_check_all": {
      "exit_code": 0,
      "status": "PASS",
      "warnings": "Pre-existing inaccessible temporary directories"
    },
    "ruff_format_sources": {
      "exit_code": 0,
      "status": "PASS",
      "python_files": 88
    },
    "ruff_format_all": {
      "exit_code": 101,
      "status": "BLOCKED",
      "reason": "Ruff panics while traversing pre-existing inaccessible temporary directories"
    },
    "pre_commit": {
      "exit_code": null,
      "status": "BLOCKED",
      "git_fetch_exit_code": 128,
      "reason": "Fresh pinned-hook initialization cannot reach GitHub; stale old cache also unusable"
    },
    "http_frontend": {
      "status": "BLOCKED",
      "reason": "FastAPI absent; package install failed; npm offline cache lacks required dependencies"
    },
    "quality_gate": {
      "status": "FAIL",
      "path": "outputs/metrics/quality_gate/20260925T005848282995Z/report.json",
      "failed_checks": [
        {
          "name": "Lint/format",
          "status": "FAIL",
          "reason": "ruff format --check --no-cache .: exit 101"
        }
      ]
    },
    "branch_syntax": {
      "christian": {
        "parsed_python_files": 67,
        "status": "PASS"
      },
      "esteban": {
        "parsed_python_files": 67,
        "status": "PASS"
      },
      "ginestar": {
        "parsed_python_files": 68,
        "status": "PASS"
      },
      "jaime": {
        "parsed_python_files": 76,
        "status": "PASS"
      },
      "javi": {
        "parsed_python_files": 67,
        "status": "PASS"
      },
      "laura": {
        "parsed_python_files": 67,
        "status": "PASS"
      },
      "santiago": {
        "parsed_python_files": 67,
        "status": "PASS"
      }
    },
    "final_wrapper_targeted": {
      "passed": 19,
      "skipped": 2,
      "exit_code": 0,
      "scope": "corruption, product integration guards and V4 runner"
    },
    "git_publication": {
      "commit_sha": null,
      "push_completed": false,
      "git_add_exit_code": 1,
      "reason": "Permission denied creating .git/index.lock"
    },
    "changed_file_hygiene": {
      "file_count": 58,
      "files_above_5mb": [],
      "forbidden_artifacts": [],
      "trailing_whitespace": []
    }
  },
  "product": {
    "level": "VERIFIED",
    "bulk_score_parity_clients": 1000,
    "adapter_and_agent_clients": 8,
    "agent_steps": [
      3,
      6,
      4,
      4,
      6,
      6,
      6,
      6
    ],
    "artifacts_unchanged": true,
    "unknown_client_rejected": true,
    "fit_scope": "TRAIN",
    "submission_sha256": "cd726195b0e6d7443ca161b446da6cdf8f07ac3b23df2d06f9ce57a3fa1ed7ee",
    "api_http_verified": false,
    "frontend_build_verified": false,
    "limitation": "Missing FastAPI/npm dependencies; no HTTP or browser test"
  }
}
```

Final commit SHA: **null**; push: **not completed**. Read-only `.git` and unavailable remote prevent publication.
