# V3 try / push status

**Branch:** `feature/esteban-v3-push` (from `esteban-v3-models-try`)
**Ulmans V2 Macro-F1:** 0.3915 (board 0.424 = accuracy)

| Model | Macro-F1 | Δ vs V2 |
| --- | ---: | ---: |
| V2 Ulmans | 0.3915 | — |
| StreamV3 | 0.4134 | +0.0219 |
| StreamV3Max | 0.4319 | +0.0403 |
| Astra V3-A (reproduced) | 0.4241 | +0.0326 |
| Astra identity∩V2 ens (ours) | 0.4350 | +0.0435 |
| **StreamV3Push (triple stack)** | **0.4445** | **+0.0530** |

```bash
python scripts/run_ubs_v3_push.py
```

Details: `esteban_v3_push.md`, `esteban_vs_astra.md`.
