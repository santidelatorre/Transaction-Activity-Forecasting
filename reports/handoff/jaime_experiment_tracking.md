# Team handoff — Experiment Tracking / Tooling

## 1. Scope

Implemented a standalone, standard-library experiment logger for the team's parallel runs. It records a UUID, UTC timestamp, Git commit and dirty-tree flag, model/version, features, hyperparameters, required classification metrics, optional V1 comparison, compute time, result, notes, risk notes, and Python version. Persistence uses SQLite WAL and a configurable busy timeout.

Out of scope: model/feature changes, model selection, changing the V1 pipeline, training runs, leaderboard submissions, and integration into the shared training runner. No files on `main` were changed; work is on `feature/jaime-experiment-tracking`.

## 2. Baseline

Reference supplied in the task: **Macro-F1 V1 = 0.2710243**, accuracy **0.2660**, **1,000 validation clients**, valid submission, and 28 automated tests passed for the established V1.

The repository's official protocol uses the provided train/valid split and the `target_next_recurring_merchant` target at cutoff `2026-01-01`. It reports macro-F1 over the eight official labels and accuracy; class-level F1 is also available. This work did not train or score a model, and did not independently reproduce the supplied V1 score. The local `data/raw/ubs_2026/` dataset and generated V1 submission are absent, so the official V1 runner and submission-file validation could not be completed here.

## 3. Hypotheses investigated

1. **Concurrent logger calls can persist without losing rows.** Motivation: seven teammates may run experiments simultaneously. Tested with 40 calls through an eight-worker `ThreadPoolExecutor`, each opening its own SQLite connection; all 40 rows and 40 unique IDs were present. This checks concurrent threads in one process; a multi-process stress run remains untested.
2. **The logger preserves the metadata needed to compare runs reproducibly.** Motivation: results need to be attributable to code, model, features, parameters, and validation scores. Tested a round trip through SQLite for required fields, per-class F1, optional V1 delta, compute time, notes, and risks. The metric values in this unit test are synthetic fixtures and are not model results.

No forecasting or feature hypotheses were tested.

## 4. Experiments performed

The identifiers below label tooling QA cases; neither is a forecasting experiment. `N/A` means no model metric was measured, not a zero score.

| experiment_id | change | macro_f1 | delta_vs_v1 | accuracy | affected_classes | result |
|---|---|---:|---:|---:|---|---|
| TRACK-QA-001 | 40 concurrent SQLite writes | N/A | N/A | N/A | N/A — no predictions | PASS: 40 rows, 40 unique IDs |
| TRACK-QA-002 | SQLite metadata/metric round trip | N/A | N/A | N/A | N/A — no predictions | PASS: required fields and computed baseline delta persisted |

No measured score is available for comparison against V1. The test fixture's synthetic metric values are intentionally excluded from this results table.

## 5. What worked

- SQLite WAL plus a 10-second configured busy timeout retained all **40/40** concurrent test records without duplicate IDs.
- Metadata, metrics JSON, notes, risk notes, compute seconds, and baseline delta round-tripped in the focused persistence test.
- The complete repository test suite passed: **39 passed**. Ruff check, Ruff format check, and Python bytecode compilation passed for the logger and its tests.

These are tooling results only; they do not improve or re-measure model quality.

## 6. What did NOT work

- No forecasting experiment was run, so there are no new Macro-F1, accuracy, per-class F1, or class-impact results. Do not treat logger QA as a model improvement.
- V1 was not rerun end-to-end: the configured `data/raw/ubs_2026/` files are not present locally. No generated submission is available to validate in this worktree.
- The concurrency test covers threads in one process, not simultaneous independent Python processes. SQLite locking/WAL is designed for this case, but the team should run a multi-process stress check before relying on heavy concurrent writes across hosts or network filesystems.

## 7. Main findings

1. A standard-library-only SQLite logger can capture the team's core experiment metadata without introducing MLflow/W&B dependencies.
2. A single SQLite database retained 40 concurrent thread-driven inserts in the focused check.
3. A supplied baseline score can be recorded alongside a run; the logger calculates `delta_vs_baseline` rather than requiring manual arithmetic.
4. A `git_dirty` flag matters: a commit hash alone cannot prove that uncommitted code was not used.
5. This work contains no evidence that any model or feature change beats V1; the current best measured result remains the supplied V1 reference.

## 8. Recommendations for V2

**MUST**
- Reuse `ExperimentLogger` for every candidate scored on the official validation split; pass the V1 reference only for directly comparable scores.
- Save the same metric definitions, client split, model settings, features, and commit with each result.

**SHOULD**
- Require a short experiment question in `description` and a truthful outcome in `result`; record compute time and leakage/other risks when known.
- Verify that `git_dirty` is false for a final candidate, or document exactly what uncommitted changes were used.

**COULD**
- Add a small report/export command and run a multi-process stress test if team usage requires it.

**AVOID**
- Claiming model gains from the logger QA fixtures or comparing runs that use different validation protocols.
- Adding a tracking service dependency before the local SQLite workflow proves insufficient.

## 9. Code worth integrating

| Path | Purpose | Dependencies | Ready? |
|---|---|---|---|
| `src/transaction_forecasting/experiment_tracking.py` | `ExperimentLogger`, validation, Git metadata, SQLite WAL writes | Python standard library only | Yes, for local repository runs; multi-process stress check remains recommended |
| `tests/test_experiment_tracking.py` | Metadata round-trip and concurrent-write regression coverage | `pytest` (development dependency) | Yes |
| `.gitignore` | Excludes `outputs/experiments/*.sqlite3*` (database, WAL, and SHM files) | None | Yes |

Import with `from transaction_forecasting.experiment_tracking import ExperimentLogger`; instantiate with `logger = ExperimentLogger()` and call `logger.log_experiment(...)`.

## 10. Commits worth integrating

`29213ff`
Add SQLite experiment tracking utility and ignore local database artifacts.
CHERRY-PICK RECOMMENDED as the logger foundation.

`d97df2b`
Capture experiment outcomes and validate concurrency, with focused tests.
CHERRY-PICK RECOMMENDED after `29213ff` (or cherry-pick both in order).

No debugging-only or model-experiment commits were created.

## 11. Dependencies / conflicts

- No new runtime library is required; storage and Git metadata use Python's standard library and the Git CLI.
- The focused tests use the repository's existing `pytest` development dependency.
- No other teammate's branch or model code is required. The logger is not wired into `scripts/run_ubs_baseline.py`, which avoids changing V1 behavior but means model runners must opt in.
- SQLite serializes writers to one local database. Avoid putting the file on an unreliable network filesystem; keep each experiment's training/validation protocol in its description/notes or a future structured field.

## 12. Risks

- **Leakage:** the logger cannot detect leakage; experiment owners must document and audit feature timing and validation use.
- **Reproducibility:** Git commit, dirty flag, model settings, feature list, metrics, and Python version are captured. Package lock state, data hashes, hardware, and random seeds are not automatically captured unless included in hyperparameters/notes.
- **Concurrency:** same-process thread writes passed; multi-process and multi-host load have not been stress-tested. SQLite WAL expects a local filesystem.
- **Compatibility:** changes to the SQLite schema in future versions will need a migration strategy if an old database is retained.
- **Dirty working tree:** the recorded dirty flag reflects the whole repository, including unrelated local modifications.
- **Performance:** logger overhead was not benchmarked; the intended workload is one short insert per completed experiment, not per prediction or training batch.
- **Evidence:** there is no new model score and no local dataset/submission to validate. Do not infer performance from this branch.

## 13. Recommended next experiment

Run the unchanged V1 pipeline once on the official local train/valid files, using the official validation protocol, and log the resulting Macro-F1, accuracy, per-class F1, elapsed training/evaluation time, commit, dirty flag, and dataset/config identifiers. First restore `data/raw/ubs_2026/`; compare its score to the supplied **0.2710243** only if the data and protocol match. This checks the logger's end-to-end integration without mixing in a model change.

## 14. Executive summary for integration AI

- This branch adds a standard-library SQLite experiment logger; it makes no forecasting changes.
- Threaded write QA persisted 40/40 records; metadata round-trip passed.
- Repository suite: 39 passed; Ruff and compile checks passed.
- No new model metrics were measured; V1 remains the supplied reference at Macro-F1 0.2710243.
- The V1 runner and submission validator need the absent local UBS dataset/submission to be rerun.
- Integrate the logger and focused tests if a lightweight local tracker is useful.
- Log protocol, seed, data/config identity, class metrics, runtime, risks, and Git dirty state for each V2 candidate.
- Do not integrate synthetic test metrics as model results or claim a V2 gain from this work.
- Review `29213ff` (logger foundation), then `d97df2b` (outcome fields and concurrency tests).
- Multi-process stress testing and future SQLite schema migrations remain follow-up work.
