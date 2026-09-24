# V3 try: stream-aware blend over V2

**Branch:** `esteban-v3-models-try`
**Base:** `5884ddd` (V2)
**Leaderboard context:** crypto 0.5964 · Ulmans 0.4240 · our V2 0.3915

## Best so far

| Model | Macro-F1 VALID | Δ vs V2 |
| --- | ---: | ---: |
| V2 baseline | 0.3915 | — |
| Stream override only | 0.3930 | +0.0015 |
| **StreamV3 (85% V2 + 15% hist+due-stream CatBoost)** | **0.4023** | **+0.0108** |
| Santiago oracle ceiling (diagnostic) | 0.7694 | — |

Music F1: 0.17 → **0.206** · Streaming: 0.25 → **0.309** · none: 0.519 → **0.539**

## What we tried (and killed)

| Idea | Result |
| --- | --- |
| CatBoost + classic `family_*` lift features | Collapse ~0.19 (all-none bias) |
| Client TF-IDF / SVD into CatBoost | Collapse ~0.08–0.14 |
| Soft due-weight logit boost | 0.36 (hurts) |
| Keyword+lift stream picker | ≤0.36 |
| Pure stream argmax | 0.17 |
| Unlabeled pretrain (prior discovery) | Δ≈0 on B→C |

## Architecture of the winner

1. Exact `(client, description)` streams + train-only family map (Santiago).
2. Per-family due-stream features (weight, lift, regularity, days-to-next).
3. CatBoost on V2 history features **plus** those due features.
4. Probability blend **0.85 × V2 + 0.15 × stream-CatBoost** (grid on VALID for this try branch).

## Gap to 0.59

Oracle shows **0.77** if we pick the right candidate family. Current detector maps only ~136 descriptions and leaves multi-family ambiguity. Closing crypto’s gap needs a much better **family disambiguation** among due streams (text at stream level, alias graphs, or pseudo-cutoffs), not another generic client classifier.

## Reproduce

```bash
python scripts/run_ubs_v3.py --data-dir data/raw/ubs_2026 --v2-blend 0.85
python scripts/experiments/esteban_v3_search.py
```
