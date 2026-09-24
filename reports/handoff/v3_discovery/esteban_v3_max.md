# V3 max: StreamV3 + soft music/streaming text

**Branch:** `esteban-v3-models-try`
**Base:** `5884ddd` (main / Ulmans V2)

## Clarification on “Ulmans 0.424”

The public board figure **0.4240 is V2 accuracy**, not Macro-F1.
Official Ulmans V2 Macro-F1 is **0.391549**. Santiago’s oracle **0.769** is a diagnostic ceiling (uses VALID labels to assign among candidates), not a submission.

## Best so far (VALID, try branch)

| Model | Macro-F1 | Accuracy | Δ vs V2 MF1 |
| --- | ---: | ---: | ---: |
| V2 (Ulmans freeze) | 0.3915 | 0.4240 | — |
| StreamV3 65/35 | 0.4134 | ~0.44 | +0.0219 |
| Soft MS shared α=0.35 | 0.4278 | 0.459 | +0.0362 |
| StreamV3Max music α=0.60 / streaming α=0.10 | 0.4305 | ~0.46 | +0.0389 |
| **StreamV3Max music α=0.60 / streaming α=0.05** | **0.4319** | **0.463** | **+0.0403** |
| Santiago oracle ceiling | 0.7694 | — | diagnostic |

Per-class at 0.4319: music ~0.365 · streaming ~0.282 · none ~0.584

## Recipe

1. Fit `StreamV3Model(v2_blend=0.65)` (history+due CatBoost blended with V2).
2. Fit Javier-style `StreamTextFamilyModel` (char TF-IDF + LR on weak train streams; positive families only).
3. Soft-add into V3 blend logits: `music += 0.60 * P_text(music)`, `streaming += 0.05 * P_text(streaming)`.
4. Softmax; argmax. **Do not** touch `none` with text.

Alphas were selected on VALID (reported, not OOF-frozen).

## Discovery mining

| Branch | Takeaway used / not |
| --- | --- |
| Javier stream-family | **Used** — music/streaming text soft boost |
| Santiago oracle | Ceiling / due-stream features already in V3 |
| Ginestar next-date | Temporal closeness tried; pure substitution small; not in freeze |
| Jaime pseudocutoffs | Ranker alone ≤0.16 MF1 — not used as classifier |
| Christian normalization | Tiny / mixed on V2 — not used |
| Laura stream-system | Ran: A~0.28, D~0.057 — too none-heavy alone |
| Esteban pretrain | Prior ablation Δ≈0 — not used |

## Reproduce

```bash
python scripts/run_ubs_v3_max.py --music-alpha 0.60 --streaming-alpha 0.05
```

Search artifacts: `outputs/metrics/v3_max/` (gitignored).
