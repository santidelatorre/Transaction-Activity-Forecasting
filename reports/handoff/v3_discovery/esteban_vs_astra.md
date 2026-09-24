# Astra `integration/v3-discovery` vs Esteban StreamV3Max

## Headline scores (official VALID Macro-F1)

| Source | Model | Macro-F1 | Accuracy |
| --- | --- | ---: | ---: |
| Ulmans / main V2 | history 75% + periodicity 25% | **0.3915** | 0.424 |
| Astra synthesis | V3-A identity-only (cross-fit family counts) | 0.4241 | 0.461 |
| Astra synthesis | 50/50 full-V3 + V2 (VALID-best, not OOF-selected) | 0.4299 | 0.470 |
| Esteban try | StreamV3 due-stream CatBoost blend | 0.4134 | ~0.44 |
| Esteban try | **StreamV3Max** (V3 + soft music/streaming text) | **0.4319** | **0.463** |

StreamV3Max is slightly above Astra’s best measured ensemble on Macro-F1; Astra ensemble is slightly higher on accuracy.

## What Astra concluded (useful)

1. **Main missing signal = family identity without own-label leakage** → cross-fitted `identity_{family}_{count,share,aliases}` into CatBoost.
2. Temporal / pooled-recurrence blocks **hurt** VALID when stacked on A (OOF looks good, VALID drops).
3. Prefer iterating on **A**, not full AB. Ensemble A/V2 is interesting but CI vs A alone spans zero.
4. Keep V2 as fallback; none is still hard.

## What we already had that Astra did not freeze

- Due-stream features from Santiago’s detector into a second CatBoost (StreamV3).
- Javier char-TFIDF soft logit boost **only** on music/streaming (StreamV3Max).
- Explicit refusal to let text rewrite `none`.

## Hybrid directions under test (`feature/esteban-v3-push`)

- Soft MS text on Astra A / ensemble.
- Probability stacks (arith/geo) of vmax × Astra × V2.
- V2-none gate + positive from identity/vmax.
- Kitchen-sink CatBoost: history + identity + due-stream + text.
- Margin arbitration / temperature sharpening.

Reproduce Astra A locally via `IdentityV3Model` in `ubs/v3_identity_model.py`.
