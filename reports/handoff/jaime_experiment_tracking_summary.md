# Jaime — Experiment Tracking / Tooling

- **Best result:** V1 remains the best measured result available: Macro-F1 `0.2710243`, accuracy `0.2660` on 1,000 validation clients (provided baseline). No new model result was measured.
- **Delta vs V1:** N/A; this branch tested tooling, not a forecasting change.

## 3 key findings

1. SQLite WAL retained all 40 concurrent thread-driven test writes.
2. The logger round-trips Git state, model/config/features, class metrics, notes, risks, runtime, and optional V1 delta.
3. All 39 repository tests passed; the UBS data and generated submission are absent, so the official V1 run was not reproduced here.

## 3 recommendations

1. **MUST:** Log every V2 candidate on the same official validation protocol and record per-class F1.
2. **SHOULD:** Include data/config identifiers, seed, runtime, risk notes, and require a clean Git tree for final results.
3. **COULD:** Add a multi-process stress test and compact SQLite export when needed.

- **Relevant commits:** `fe85154` logger foundation; `cd19c44` outcome fields and concurrency tests.
- **Main risk:** no multi-process stress test yet; SQLite WAL should stay on a local filesystem.
- **What V2 should do:** use this logger for a clean, unchanged V1 rerun first, then compare single, controlled model experiments against Macro-F1 `0.2710243`.
