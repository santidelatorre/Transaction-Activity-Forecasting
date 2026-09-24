# V3 try: stream-aware blend over V2

**Branch:** `esteban-v3-models-try`
**Base:** `5884ddd` (V2)
**Leaderboard context:** crypto ~0.5964 · Ulmans **accuracy** 0.4240 (= V2) · Ulmans V2 Macro-F1 **0.3915**

## Best so far (VALID)

| Model | Macro-F1 | Δ vs V2 |
| --- | ---: | ---: |
| V2 baseline (Ulmans) | 0.3915 | — |
| StreamV3 65% V2 + 35% hist+due CatBoost | 0.4134 | +0.0219 |
| **StreamV3Max + soft music/streaming text (α 0.60 / 0.05)** | **0.4319** | **+0.0403** |
| Santiago oracle ceiling | 0.7694 | diagnostic only |

See `esteban_v3_max.md` for the freeze recipe and discovery mining notes.

## Reproduce

```bash
python scripts/run_ubs_v3_max.py --music-alpha 0.60 --streaming-alpha 0.05
```
