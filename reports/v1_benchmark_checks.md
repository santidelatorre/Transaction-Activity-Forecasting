# V1 benchmark — execution checks

Date: 2026-09-24. Working branch: `feature/v1-benchmark`.

## Predictive reproduction

- Historical V1 source: `0199a8b7c8b2c790a9d3447b156f2088708c2864`.
- Frozen V2 source: `96409b7940a991fbda5235b85ffeb40652a4087b`.
- Both executed from isolated detached worktrees; no predictor was edited.
- V1: Macro-F1 **0.2710242658492452**, accuracy **0.266**.
- V2: Macro-F1 **0.3915494559105542**, accuracy **0.424**.
- Both submissions match the SHA-256 values in the earlier V2 integration report byte for byte.
- Shared evaluator independently rescored fresh predictions, aligned by exact client ID set.
- Both submissions passed the shared strict schema, vocabulary, coverage and order validator.
- Real input checks: no post-cutoff transactions, exact duplicates or client overlap across splits.
- 246 V1 errors corrected; 88 V1 correct predictions lost; 178 both correct; 488 both wrong.
- Figure labels/layout inspected visually for all three generated PNGs.

## Tests

V1/current branch: `python -m pytest -q` — **37 passed**, one cache-permission warning.
The first sandboxed attempt passed 34 tests and had three temporary-directory permission errors;
rerunning with authorized access resolved those setup errors.

V2 isolated historical snapshot: **79 passed, 1 failed** out of 80 collected tests.
The failing test is
`QualityGateTests.test_default_end_to_end_does_not_train_and_writes_warn_report`.
It expects WARN (exit 1), but `scripts/quality_gate.py:610` explicitly refuses detached HEAD and
returns FAIL (exit 2). This snapshot is deliberately detached to preserve the working branch.
An earlier attempt additionally encountered Git's worktree ownership check; a process-local
`safe.directory` exception for the exact worktree resolved that environmental issue.

This is a documented test-context incompatibility, not an all-tests-pass claim. No V2 tests,
quality-gate behavior or branch references were altered to hide it. The V2 final report's historical
79-pass claim is separate from this fresh 80-test run. The predictor and submission tests passed.

## Source and artifact preservation

- Original `src/`, baseline runner and configs have no changes on this branch.
- Historical worktrees contain the exact frozen code; copied data and outputs are ignored by Git.
- Original submission files were not overwritten. Audit submissions live under
  `outputs/metrics/v1_benchmark/{v1,v2}/submission.csv`.
- Per-client comparison stays ignored at `outputs/metrics/v1_benchmark/paired_validation.csv`.
- Versioned evidence contains only aggregate statistics, feature names and hashes.
- `scripts/emergency_submission.py` remains untracked and unmodified by this task.
- No submission was uploaded; no PR or merge was performed.

## Limitations

Both models reused official validation during selection. Their paired difference is a reproduced
local comparison, not an independent estimate of hidden leaderboard performance. Historical
ablations are attributed to the integration report, not claimed as rerun here. Runtime measurements
include concurrent execution and different workloads (V1 selection versus V2 fixed fit/refit).

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_v1_benchmark.py --version v1
.\.venv\Scripts\python.exe scripts/reproduce_v1_benchmark.py --version v2
.\.venv\Scripts\python.exe scripts/analyze_v1_baseline.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

Ruff check and format passed for all 43 current-branch Python files. All pinned pre-commit hooks
passed: Ruff, Ruff format, trailing whitespace, YAML and added-large-files. The original cache had
a missing manifest; a fresh ignored `.pre-commit-cache-v1-benchmark` resolved it.
