# V3 PushEmbed: unlabeled pretrain × StreamV3Push

**Branch:** `feature/esteban-v3-push`
**Model:** `StreamV3PushEmbedModel` in `ubs/v3_pretrain.py`

## Headline

| Model | Macro-F1 | Accuracy | Δ vs Push |
| --- | ---: | ---: | ---: |
| Ulmans V2 | 0.3915 | 0.424 | — |
| StreamV3Push | 0.4445 | 0.476 | — |
| **StreamV3PushEmbed (w=0.85)** | **0.4570** | **0.484** | **+0.0125** |

Δ vs Ulmans V2: **+0.0655** Macro-F1.

## Recipe

1. Fit `StreamV3PushModel` (equal stack vmax + identity-A + stream V3).
2. Fit char TF-IDF (3–5) + TruncatedSVD(32) on up to 200k unlabeled pretrain descriptions.
3. Client bag = unique non-noise descriptions joined; embed → L2 → multinomial LR.
4. Blend: `0.85 * P_push + 0.15 * P_embed`, renormalize, argmax.

Push weight selected on VALID grid (`0.85` / `0.90` / `0.95` all beat Push; `0.85` best).

## Reproduce

```bash
python scripts/run_ubs_v3_pretrain.py --push-weight 0.85
```

## Jury demo

```bash
python scripts/demo_subscription_foresight.py metrics
python scripts/demo_subscription_foresight.py tour --client C000000
```

See `esteban_demo_README.md`, `esteban_pitch_1min.md`, `esteban_pitch_qa.md`.
