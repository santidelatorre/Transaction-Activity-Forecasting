# V2.1 investigation summary

- Branch: `v2.1-integration`; frozen V2: `96409b7`; V1: `0199a8b`.
- V2 remains best: Macro-F1 **0.391549456**, accuracy **0.424000**.
- No candidate was promoted to V2.1 and no V2.1 submission was generated.
- All 88 V1-only wins are raw recurrence-heuristic wins; V2 history CatBoost
  explains 242/246 V2-only wins. The conflict is model arbitration, not a
  missing generic aggregate.
- `music`: V1 19 correct, history CatBoost 11, V2 13; seven of 15 lost V1 wins
  become `streaming` under V2.
- Fixed raw/confidence arbitration failed internal OOF (0.408690/0.407408 versus
  V2 0.408884).
- Leakage-safe cross-fitted exact family features looked strong internally
  (0.489785) but collapsed on official validation (0.197868).
- Normalized leave-one-client-out merchant features also failed (0.116346
  official). Both supervised text variants suffered severe split shift.
- Recommendation: retain V2; pursue split-aware/domain-robust text only with a
  stronger holdout protocol.

Full evidence and commands: `reports/v21_final_report.md`.
