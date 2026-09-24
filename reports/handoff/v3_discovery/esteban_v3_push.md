# V3 push: Astra identity × StreamV3Max stack

**Branch:** `feature/esteban-v3-push`
**Parents:** `esteban-v3-models-try` + ideas from `integration/v3-discovery` (Astra)

## Comparison

| Model | Macro-F1 | Accuracy | Notes |
| --- | ---: | ---: | --- |
| Ulmans V2 | 0.3915 | 0.424 | board “0.424” is accuracy |
| Astra V3-A (reported) | 0.4241 | 0.461 | cross-fit identity features |
| Astra 50/50 full+V2 (reported) | 0.4299 | 0.470 | OOF preferred full; VALID liked ensemble |
| StreamV3Max (ours) | 0.4319 | 0.463 | soft music/streaming text |
| **StreamV3Push** | **0.4445** | **0.476** | equal stack vmax+A+v3 |

Δ vs Ulmans V2: **+0.0530** Macro-F1.

## Recipe

Equal probability average of:

1. `StreamV3MaxModel` (StreamV3 + music α=0.60 / streaming α=0.05 text soft boost)
2. `IdentityV3Model` (Astra V3-A: history + cross-fitted `identity_*` + 75/25 heuristic)
3. `StreamV3Model` (history + due-stream CatBoost @ 0.65 V2 blend)

## What failed / weaker

- Kitchen-sink CatBoost (history+identity+due+text): ~0.37
- Soft MS text on Astra A alone: usually hurts vs A
- V2-none hard gate: ≤0.428
- Astra’s AB temporal blocks: already rejected by Astra; we use A only

## Reproduce

```bash
python scripts/run_ubs_v3_push.py
```
