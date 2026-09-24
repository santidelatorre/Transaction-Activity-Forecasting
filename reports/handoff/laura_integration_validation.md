# Team Handoff — Integration / Validation / Quality Gate

## 1. Scope

This handoff documents the integration/validation contribution on `features/laura-v2-integracion`, based on the committed gate, project evaluators and loaders, V1 runner, configuration, tests, and recorded validation evidence. The role is to provide repeatable evidence for human review across seven parallel workstreams. The contribution adds a local gate; it does not change V1 modeling or automate integration. No training or `--evaluate` was run for this handoff.

## 2. Contribution summary

Commit `614f769c5fcece506d9f0aae61b3c811bddcf8e8` adds a configured local quality gate, its tests and user documentation. It runs tests and Ruff, inventories branch/worktree changes, can explicitly run the existing V1 runner, independently scores saved validation predictions, validates the test submission, records provenance and compares reviewed reports. It refuses `main`; it never merges. The configured `0.271` is explicitly approximate and cannot trigger a blocking metric result.

## 3. Files contributed

| path | purpose | integration_value |
|---|---|---|
| `scripts/quality_gate.py` | Local gate, report generation, provenance, hygiene and optional evaluation | Gives feature branches a common evidence format and explicit severity |
| `configs/quality_gate.toml` | Reference, test targets, baseline placeholder and configurable thresholds | Makes policy reviewable without editing gate logic |
| `tests/test_quality_gate.py` | Dependency-light gate unit tests | Exercises policy edges and guards against accidental training/main behavior |
| `docs/QUALITY_GATE.md` | Usage, output and known limitations | Gives contributors a shared operating procedure |

These files are the contribution in the branch’s only commit ahead of `origin/main`. The shared official evaluator, UBS data loader/validator, V1 feature builder and runner, project dependencies and CI are existing infrastructure, not Laura’s contribution.

## 4. Quality gate architecture

`python scripts/quality_gate.py` is the default entry point. It reads `configs/quality_gate.toml`, refuses detached HEAD and branch `main`, compares against the locally available `origin/main` reference, inventories committed/staged/unstaged/non-ignored untracked paths, runs configured targeted tests followed by the full pytest suite and Ruff checks, and writes JSON/Markdown reports plus JUnit XML beneath ignored `outputs/metrics/quality_gate/<run-id>/`. It does not fetch, install dependencies, train, or merge. The report includes a warning that the reference is local and was not refreshed.

`--evaluate` is explicit and potentially expensive: it invokes `scripts/run_ubs_baseline.py` with the existing V1 protocol and isolated output paths. It hashes inputs before and after, validates fresh artifacts, and records runtime/provenance. `--candidate REPORT --baseline REPORT` rechecks saved evidence without training, requiring current provenance and intact artifact hashes. `--baseline REPORT` supplies the reviewed V1 gate report. Reports are evidence for a feature branch → gate → human review → PR → possible integration; PASS does not integrate anything.

## 5. Checks implemented

| check | purpose | PASS | WARN | FAIL |
|---|---|---|---|---|
| Tests / JUnit | Run configured targeted tests and full `tests`; parse XML and compare test nodes/failures when a baseline report exists | Commands succeed and usable evidence exists | Missing pytest/XML/baseline, skipped tests, or tests removed from full suite | Nonzero tests, XML failures/errors, malformed XML, or new failures against baseline |
| Ruff | Run `ruff check --no-cache .` and `ruff format --check --no-cache .` | Both succeed | Ruff unavailable | Either command fails |
| Performance / provenance | Score validation outputs and compare compatible saved reports | Comparable evidence within thresholds | Missing/unverified/mismatched evidence; approximate score only | Macro-F1 drop beyond blocking threshold against comparable verified baseline |
| Per-class / distribution | Compare each class F1 and predicted share | No configured warning exceeded | Class F1 or share shift exceeds configured warning; zero predictions for a supported class | No separate class/distribution-only FAIL rule |
| Submission | Reuse official and UBS checks on output CSV | Exact schema, IDs, allowed labels, coverage, count and order accepted | Evidence missing | Invalid submission |
| Data integrity / leakage | Exercise V1 loader contracts during explicit evaluation | Loader accepts cutoff, partition and label-alignment contracts | Feature-level safety cannot be proven automatically | Loader rejects cutoff/partition/label contract violation |
| Git hygiene | Inventory changed paths and sizes; flag generated/sensitive names and notebooks | No prohibited/oversized paths detected (PR scope still WARN) | Untracked suspicious files, notebooks, symlinks, local reference uncertainty, or review needed | Tracked/staged generated or sensitive path, or file over blocking size |
| Runtime / memory | Compare recorded full-run wall time; state memory limitation | Same machine and runtime ratio ≤ threshold | No comparable timing, ratio above threshold, or memory unmeasured | No runtime-only FAIL rule |
| Gate execution | Produce report and aggregate severities | No check above PASS | Any WARN and no FAIL | Any FAIL |

Overall severity is maximum of component severity (`PASS < WARN < FAIL`), with exit codes `0`, `1`, and `2` respectively. A default run normally remains WARN because it does not evaluate performance and the configured baseline report is empty; missing evidence is not silently promoted to PASS.

## 6. Performance regression policy

All values below are configurable engineering decision thresholds, not UBS rules, statistical significance, leaderboard guarantees, or scientific evidence.

| metric | warning_threshold | blocking_threshold | reasoning | limitations |
|---|---:|---:|---|---|
| Macro-F1 absolute drop | >0.005 | >0.010 | Tolerate small movement; flag larger regression; block only a clearly configured drop | Applies only if baseline artifacts exist and data, protocol and environment match |
| Per-class F1 absolute drop | >0.05 | none | Highlight potentially hidden class harm | Warning only; no confidence intervals or support-sensitive test |
| Predicted class-share absolute change | >0.10 (10 percentage points) | none | Flag a large distribution shift | Not inherently a model failure; review class support and predictions |
| Supported class with zero predicted validation examples | warning | none | Surface a class coverage collapse | Valid output format remains distinct from model quality |
| Runtime ratio | >1.5x | none | Flag a materially slower run on the same recorded machine | Wall time includes candidate runner/final refit; hardware-normalized benchmark and peak memory are absent |

Metrics are produced by the official fixed-eight-label scorer: Macro-F1, per-class precision/recall/F1/support/predicted count and confusion matrix. The gate independently scores `model_prediction` from V1’s validation error CSV against `valid_labels.csv`, then checks agreement with saved summary Macro-F1. The approximate `0.271` in config is only displayed as context when candidate metrics exist; it is not reproducible evidence, and cannot cause FAIL. No reviewed baseline report/artifact is present in the inspected branch state, so baseline Macro-F1 is **NOT VERIFIED**.

## 7. Submission validation

The gate reads the candidate CSV with string values and calls the existing UBS `validate_submission`, which in turn enforces the official contract: exactly `client_id,predicted_next_recurring_merchant`, expected client coverage and count, unique IDs, valid/nonempty IDs, labels among the eight allowed classes, and sample order. It also loads the UBS split and calls the shared official scorer for validation metrics. Thus a malformed submission is an integration blocker, while a schema-valid submission with weak Macro-F1 or a missing predicted class is a model-quality concern. Test predictions are validated but never scored against hidden labels. There is no separate new submission validator attributed to this contribution.

## 8. Leakage and validation integrity

### Automatic safeguards

- The UBS loader rejects transactions at or after cutoff `2026-01-01`, invalid/duplicate rows, client overlap among train/validation/test, and label/transaction misalignment.
- V1 validation fits the feature builder on train transactions/labels and transforms validation separately. Text TF-IDF/scaling/imputation fit on training rows in the V1 estimator path.
- Client IDs are used for indexing/alignment; the inspected model inputs do not include raw IDs as predictors.
- Evaluation provenance hashes the six expected data files and relevant source/config/test Python/TOML files, records commit, protocol, Python and selected package versions, and detects input changes during a fresh run.

These verify specific contracts only; they do not prove arbitrary feature code safe.

### Existing V1 warnings

- The supervised description-lift map is learned from training labels and is then applied to the same training clients while fitting the model. That is training self-label influence and deserves review/cross-fitting; it is not direct evidence of held-out validation labels entering features.
- V1 repeatedly selects models, heuristic bias and possible ensemble using the same validation set; the winning validation score is selection-biased and should not be described as an unbiased generalization estimate.
- The final train+validation refit is for test prediction. Its metrics must not be scored on those same validation labels as held-out evidence.

### Confirmed leakage

No direct train/validation leakage violation was confirmed by this review. This is not a claim that leakage is absent.

### Human review required

Review every changed feature’s timestamp availability, fit scope, labels, IDs, normalization and text mapping. The gate performs no static proof or independent sandboxing; changing the evaluator or tests can also weaken evidence. Validation reuse for model selection requires an honest holdout or cross-validation plan if an unbiased estimate is needed.

## 9. Git and artifact hygiene

The gate inventories committed paths versus its local reference, staged paths, unstaged paths and non-ignored untracked paths. It does not fetch `origin/main`; stale remote-tracking state is therefore possible and is always a warning. Generated/data locations (`data/`, `outputs/`), caches, common serialized/model/output extensions and `.venv` paths are flagged. A changed tracked path in those categories is FAIL; an untracked item is WARN. Sensitive names/directories (including `.env*`, credential/secret/password/token names, SSH key names and common key extensions) are not opened; tracked/staged instances are FAIL and untracked instances WARN. Size uses changed Git blob and working-file sizes where available: >5 MiB WARN, >25 MiB FAIL. Notebook files receive an output-review warning. Symlinks/out-of-root paths are not followed and are WARN. Deleted paths are reported but deletion alone is not prohibited.

Git filename scanning cannot prove that ordinary source files contain no embedded credentials. Human diff review remains required. It does not delete or sanitize any path, and ignored files (including `.venv`) are outside the non-ignored untracked inventory.

## 10. Reproducibility and provenance

Fresh evaluation records current commit, hashes of the six data inputs and relevant `.py`/`.toml` sources plus `pyproject.toml`, non-output protocol settings, Python and selected core package versions, machine identifiers, runtime and hashes of summary/error CSV/submission/generated config. Inputs are rehashed after evaluation. Saved reports require artifact hashes to match and candidate rechecking requires current provenance to match. Baseline comparison requires equal data hashes, protocol and environment; runtime comparison additionally requires matching machine metadata. This is why modified artifacts, changed data/config/code/environment/commit can invalidate a comparison.

This supports reproducibility checks, not authentication or trust: hashes show bytes match a claimed report but do not prove who ran it, that a report is honest, or that the source/evaluator is secure. Reports supplied as baseline must be reviewed by the team. Package inventory is limited, and machine/runtime equality does not eliminate nondeterminism.

## 11. Validation status

| component | status | evidence | remaining_work |
|---|---|---|---|
| Gate unit tests | VERIFIED | `.venv` Python 3.11.16; `pytest -q tests/test_quality_gate.py`: 19 passed | Re-run on future changes |
| Targeted repository tests | VERIFIED | Configured five-file target list: 49 passed | Re-run after integration changes |
| Full pytest suite | VERIFIED | `pytest`: 56 passed | Re-run after future changes |
| Ruff on gate Python files | VERIFIED | `ruff check scripts/quality_gate.py tests/test_quality_gate.py` passed | Full-repository lint/format was not rerun after final two E501 fixes |
| Ruff format on gate Python files | VERIFIED | `ruff format --check` on the two gate Python files passed | Full-repository format was not rerun after final fixes |
| Unit/mocked gate scenarios | PARTIALLY VERIFIED | Tests cover threshold edges, artifact tampering, sensitive paths, `main` refusal and no-training default behavior | Does not establish end-to-end performance evidence |
| Real UBS evaluation / dataset | NOT VERIFIED | UBS data directory was unavailable in the inspected environment; no `--evaluate` run | Run explicitly in a data-ready environment if authorized |
| V1 baseline and Macro-F1 ≈0.271 | NOT VERIFIED | Config baseline report path is empty; no reviewed gate baseline report/artifacts | Generate and review a baseline report on the intended V1 commit |
| Real candidate comparison / submission | NOT VERIFIED | No real validation artifacts or candidate-vs-baseline run | Requires data, baseline and candidate artifacts |
| Leakage absence / runtime comparison | NOT VERIFIED | Static review and unit tests only; no live run or proof for arbitrary features | Human review; comparable run on same machine for runtime |
| Handoff documents | VERIFIED after creation | File existence, required headings, and `git diff --check` | Commit/push must be reported separately |

Historical executed checks before the last two string-only E501 formatting fixes: quality-gate tests 19 passed, targeted tests 49 passed, full suite 56 passed; full-repository Ruff had earlier reported issues and was not rerun after the targeted fixes. Do not infer a current full-project Ruff PASS from targeted Ruff success.

## 12. Known limitations

- No reproducible baseline report or real `0.271` artifact is available; approximate config value is nonblocking context only.
- No live `--evaluate`, real dataset scoring, candidate comparison or real submission validation was performed for this handoff.
- Default mode runs tests/static hygiene, not performance evaluation, so performance/submission/provenance evidence is WARN/missing.
- A local `origin/main` reference may be stale; the gate deliberately does not fetch.
- Test-removal detection needs a full suite and baseline test manifest. Without baseline evidence, new-failure comparison is unavailable.
- Gate code executes branch-controlled project scripts in the current environment; it is not a security sandbox. Filename scanning is not secret-content detection.
- Leakage review is not automated beyond existing loader contracts and provenance.
- Per-class and distribution thresholds are warning-only; runtime has no blocking threshold; memory is not measured.
- Provenance records selected packages, not a complete lock/freeze or authenticated execution attestation.
- Validation counts above are prior executed evidence; full Ruff status after the last fixes remains unverified.

## 13. How the other 6 members should use this

Run the default gate before opening a PR. It gathers test, lint and Git-hygiene evidence without training. For a model comparison, coordinate an explicitly authorized `--evaluate` run and use reviewed baseline/candidate reports with the same data, protocol and environment. Save report directories with their artifacts. Read each WARN/FAIL and review the changed source; never interpret the gate as automatic approval.

## 14. PR integration checklist

- [ ] Correct feature branch; review full diff
- [ ] Targeted tests; full tests when appropriate
- [ ] Ruff check and format
- [ ] Comparable Macro-F1 plus per-class metrics and prediction distribution reviewed
- [ ] Submission schema/IDs/labels/coverage/order valid
- [ ] Leakage and validation-use review recorded
- [ ] Baseline/candidate provenance and artifacts compatible
- [ ] No datasets, secrets, oversized artifacts, caches or unrelated changes
- [ ] Runtime reviewed when comparable baseline exists
- [ ] Human integration approval

## 15. Recommendations by team area

| team_area | integration_risk | recommended_action | priority |
|---|---|---|---|
| TEMPORAL / RECURRENCE | Post-cutoff transactions or future recurrence intervals | Derive features strictly before cutoff; record interval/window definitions and run gate | HIGH |
| TEXT / MERCHANTS | Label-informed merchant mappings can self-influence training rows | Document fit scope; test train-only/cross-fitted mappings and compare per-class metrics | HIGH |
| FEATURE ENGINEERING | New aggregates may use validation/test distribution or identifiers | State feature availability time and fit data for each feature; inspect diff manually | HIGH |
| MODELS / TUNING | Repeated validation selection inflates winning score | Keep protocol fixed for gate comparison; use train-fold CV/untouched holdout for unbiased estimate | HIGH |
| ERROR ANALYSIS | Overall score hides rare-class collapse | Attach class metrics/confusion and investigate class with zero predicted examples | MEDIUM |
| EXPERIMENT TRACKING | Stale/mismatched artifacts yield incomparable claims | Preserve reports and artifacts; compare only matching hashes/protocol/environment | HIGH |
| INTEGRATION / VALIDATION | Gate WARN may be mistaken for approval | Treat status as evidence severity; resolve reasons and obtain human PR review | HIGH |

## 16. Commits worth reviewing

| commit_hash | description | recommendation |
|---|---|---|
| `614f769c5fcece506d9f0aae61b3c811bddcf8e8` | Add V2 integration quality gate (four files; sole branch commit ahead of `origin/main`) | REVIEW BEFORE INTEGRATING |

No separate handoff commit is included in this table until it exists. V1 runner/evaluator code is inherited project behavior, not this contribution.

## 17. Dependencies and conflicts

The gate intentionally reuses the existing V1 runner, `configs/ubs_v1.toml`, UBS data loader/submission validator and official eight-class scorer. Feature/model branches may alter source/config/tests and thereby change provenance; that is expected but makes old artifacts stale. Changes to evaluator, split, labels, validation error output schema, test collection or runner config can invalidate comparison and need coordination with integration. Gate policy config is separate from model config. Avoid simultaneous edits to the same gate/config files; contributors should keep work on their feature branch and provide artifacts/reports to integration.

## 18. Recommended V2 integration process

`feature branch → python scripts/quality_gate.py → inspect JSON/Markdown evidence and diff → resolve FAIL and review WARN → human review → PR → possible integration`.

For performance, first establish a reviewed V1 report, then explicitly run `python scripts/quality_gate.py --evaluate --baseline <path-to-reviewed-v1-report>` for the candidate. A saved candidate can later be rechecked with `python scripts/quality_gate.py --candidate <candidate-report> --baseline <v1-report>`. These are the actual implemented CLI forms; `--evaluate` runs the configured V1 candidate search and final train+validation refit and is not lightweight.

## 19. What NOT to do

- Do not merge automatically or equate PASS with integration approval.
- Do not integrate only because Macro-F1 rises; inspect classes, distribution, data integrity and runtime.
- Do not compare incompatible splits, data, protocol, environments or stale artifacts.
- Do not ignore severe class regressions, even where the overall metric improves.
- Do not claim leakage is absent without evidence.
- Do not trust unreviewed reports just because their hashes are internally consistent.
- Do not add datasets, secrets, large outputs or model artifacts.
- Do not treat approximate `0.271` as a reproducible baseline unless a reviewed run proves it.

## 20. Executive summary for V2 integration AI

Laura’s contribution is a local, non-merging quality gate in commit `614f769`.
Run `python scripts/quality_gate.py` on a feature branch for targeted/full tests,
Ruff and Git hygiene; it writes JSON/Markdown/JUnit evidence under ignored
outputs. `--evaluate` explicitly runs the existing V1 pipeline; saved report
comparison requires matching provenance and intact artifacts. PASS means checks
have evidence, WARN means evidence is incomplete or review is needed, and FAIL
means a blocker was demonstrated; the worst check sets overall status. Unit,
targeted and full pytest passed in the recorded Python 3.11 environment. Ruff
passed for the two gate Python files after final fixes; full-repository Ruff was
not rerun then. No real UBS evaluation or submission was run. The approximate
0.271 is not a verified baseline. Review leakage and all WARNs manually; hashes
support reproducibility, not authentication. Human PR review remains required.
