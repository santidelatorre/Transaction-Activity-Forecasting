# V3 try: stream-aware blend over V2

**Branch:** `esteban-v3-models-try`
**Base:** `5884ddd` (V2)
**Leaderboard context:** crypto 0.5964 · Ulmans 0.4240 · our V2 0.3915

## Best so far (VALID)

| Model | Macro-F1 | Δ vs V2 |
| --- | ---: | ---: |
| V2 baseline | 0.3915 | — |
| StreamV3 85/15 | 0.4023 | +0.0108 |
| **StreamV3 65% V2 + 35% hist+due CatBoost** | **0.4134** | **+0.0219** |
| Santiago oracle ceiling | 0.7694 | — |

Per-class at 0.4134: cloud 0.44 · gym 0.473 · insurance 0.441 · mobile 0.46 · music 0.198 · software 0.419 · streaming 0.296 · none 0.581

## Failed paths

Family_* in CatBoost, client TF-IDF/SVD, soft due logit boost, keyword pickers, pure stream argmax — all ≤0.36 or collapse.

## Gap to crypto 0.59

Oracle 0.77 if the correct due-stream family is chosen. Need better **disambiguation among candidate families** (stream-level text, aliases, pseudo-cutoffs), not more client-level classifiers.

## Reproduce

```bash
python scripts/run_ubs_v3.py --v2-blend 0.65
```
