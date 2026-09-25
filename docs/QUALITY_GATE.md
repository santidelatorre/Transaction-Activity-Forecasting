# Local V2 quality gate

Run on a feature branch using the project's existing Python 3.11+ environment.
The gate never fetches, checks out, merges, commits, pushes, installs packages or
deletes project files. It refuses `main` and detached HEAD. No pipeline code is
changed. A human remains responsible for integration.

```bash
python scripts/quality_gate.py
```

The default runs targeted tests, then the full suite, then read-only Ruff checks.
It inventories committed changes against the locally fetched `origin/main`, plus
staged, unstaged and nonignored untracked paths. It does **not** train. Without
performance evidence it returns WARN, not PASS. It never refreshes origin/main
automatically; update that reference separately when authorized.

## Explicit evaluation

```bash
python scripts/quality_gate.py --evaluate
```

This can be expensive: it runs the existing V1 candidate search **and final
train+validation refit**, using `configs/ubs_v1.toml`. It does not invent a faster
evaluation split or silently change models. Earlier blocking checks prevent
training. Missing data/dependencies produce WARN without installing anything.

Each invocation creates `outputs/metrics/quality_gate/<run-id>/`, already ignored
by Git. A generated `candidate.toml` changes only output destinations. Outputs
include isolated runner artifacts, submission, normalized validation predictions,
pytest XML, `report.json` and `report.md`. Reports record commands, thresholds,
Git changes, file sizes, metric details and reasons. Raw subprocess output is
not echoed. Investigate a failed command separately using its recorded arguments.

The gate independently scores `model_prediction` from the runner's validation
error CSV against `valid_labels.csv`, never against its embedded `actual` values.
It reuses official eight-class scoring and both official/UBS submission checks.
The report contains per-class precision/recall/F1 and the confusion matrix.
Test-set predictions are only validated for compatibility, never scored.

## Baseline and saved candidate evidence

The configured 0.271 score is approximate and cannot trigger an automatic metric
FAIL. For a blocking comparison supply a reviewed V1 **gate report**, including
its artifacts. Generating this report on an appropriate baseline branch is a
separate, explicitly authorized operation; the gate does not switch branches.

```bash
python scripts/quality_gate.py --evaluate --baseline outputs/metrics/quality_gate/<v1-run>/report.json
python scripts/quality_gate.py --candidate outputs/metrics/quality_gate/<candidate-run>/report.json --baseline outputs/metrics/quality_gate/<v1-run>/report.json
```

Saved candidate review reruns tests and rescoring, not training. It requires the
same current commit, source/config/test hashes, data hashes and environment as
the saved run. Editing code after evaluation makes it stale. Preserve runner
artifacts with the report; changed/missing artifact hashes cannot be accepted as
verified evidence. Ordinary legacy runner outputs without a gate manifest are
not automatically trusted. Hashes support reproducibility, not authentication:
only supply reports reviewed by your team, not untrusted JSON claiming a score.

Baseline and candidate must match data hashes, the non-output configuration and
Python/package versions. Differences produce WARN and disable blocking score
thresholds. This conservative comparison intentionally requires human review
when V2 changes model settings. V1 baseline commit and origin/main PR reference
are distinct: main can advance without redefining V1. Baseline reports retain
their original commit and per-class metrics. Both reports' test results identify
new failures; removed tests are flagged only after a full candidate suite.

## Rules and thresholds

Configure `configs/quality_gate.toml`; no thresholds are leaderboard guarantees.

| Signal | WARN | FAIL |
|---|---|---|
| Macro-F1 drop | >0.005 absolute | >0.010 against comparable verified baseline |
| Per-class F1 drop | >0.05 absolute | No automatic class-only blocker |
| Prediction share change | >10 percentage points | No distribution-only blocker |
| Missing predicted class | Supported validation class receives no predictions | Invalid class values fail submission validation |
| Full-run runtime | >1.5x baseline on matching machine/environment | No automatic time blocker |
| Changed file size | >5 MiB | >25 MiB |
| Tests | Missing tooling, skipped tests, missing baseline evidence | Test failure/collection error |
| Submission | Missing evidence | Invalid CSV/schema/IDs/classes/order |
| Provenance | Missing/stale saved evidence | Inputs change during a fresh run |

Tracked datasets, model artifacts, caches, temporary outputs and sensitive
filenames are blockers. Untracked suspicious files require review. Changed
tracked blobs are size-checked even if the working copy is smaller. Credential
files are never opened; symlinks are not followed by the hygiene scanner.
Filename checks cannot detect arbitrary embedded secrets. Notebooks are flagged
for human output review. The gate does not delete files or sanitize your branch.

PASS means a specific check has evidence. WARN means incomplete evidence or a
review concern. FAIL means a demonstrated blocker. Overall severity is the
maximum; exit codes are 0=PASS, 1=WARN, 2=FAIL. The report ends with overall status
and explicit reasons. Missing checks cannot silently become PASS.
Missing/empty pytest XML is WARN; malformed XML or recorded test failures are
FAIL even if the subprocess returns zero. Sensitive parent directories are
rejected as well as sensitive filenames. Candidate metrics become available
for comparison only after metric consistency and submission validation succeed.

## Leakage limitations

The existing loader explicitly rejects timestamps at/after 2026-01-01 and client
overlap across partitions. V1 fits preprocessing on training clients during
validation scoring and retains client IDs as an index rather than a feature.
Successful data loading verifies those data contracts, **not arbitrary feature
code**. The gate does not pretend static keyword searches prove leakage safety.

Human review remains required for changed features/models/evaluation:

- Does any feature use post-cutoff observations, held-out labels, raw IDs, or
  preprocessing fitted on validation/test?
- Training description mappings include each training client's own label. This
  is an inherited self-label-influence concern, not demonstrated held-out leakage.
- Validation selects models and heuristic/ensemble settings; its winning score
  is selection-biased, not an unbiased estimate of future performance.
- Final train+validation refit is for test prediction only. Never score that
  refit on validation as if validation were held out.
- Changes to the evaluator, test collection or configured runner need review;
  execution of branch code is not an independent security sandbox.

These inherited/review limitations stay WARN rather than claiming leakage is
absent. Actual cutoff/client-overlap violations rejected by the loader fail a
fresh pipeline run. Peak memory is not measured in this lightweight version and
is explicitly WARN; runtime is wall time, not a hardware-normalized benchmark.

## Gate development tests

```bash
python -m pytest -q tests/test_quality_gate.py
```

The gate's standard-library unit tests can also run without modeling packages:

```bash
python3 -B -m unittest discover -s tests -p test_quality_gate.py
```

This fallback does not validate the forecasting pipeline or replace the full
suite. The gate itself requires the project's documented Python 3.11+ runtime.
