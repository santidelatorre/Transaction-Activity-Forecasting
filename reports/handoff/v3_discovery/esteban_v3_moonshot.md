# Moonshot log: path toward ~0.70 Macro-F1

**Branch:** `feature/esteban-v3-push`
**Best deployable now:** StreamV3PushEmbed **~0.4570** (Push × unlabeled pretrain embeds)
**Previous:** StreamV3Push **0.4445**

## What we learned (hard)

1. **Santiago abstaining oracle ≈ 0.77** with mapped due candidates (recall ~0.65).
2. **Signature merchants are not pure:** even `urban gym` / `audio streaming` only co-occur with their family ~30–38% of the time. Winner-take-all mapping is polluted by multi-label merchants (`digital plus`, `premium plan`).
3. **Soft multi-map + recurrent** raises diagnostic ceilings (~0.78–0.92 depending on lift) but every Push restrict/mix **hurt** VALID (best mix 0.4415 < 0.4445).
4. **Disambiguation among candidates is barely above chance** with due_weight / days_to_next (~33% among multi-family covered). Stream geometry alone does not pick the true family.
5. **Target = next recurring merchant family in 90d.** Keyword earliest-projection matches true **~80% when true is already in the keyword set** (oracle-conditioned), but the deployable rule precision is only ~31–37%, so overrides destroy Push.
6. **Learned rankers, Dirichlet, bigrams, desc multi-hot, set-matching, pretrain embeds** — none beat Push in VALID searches so far.

## Scoreboard (VALID Macro-F1)

| Model / idea | MF1 | Notes |
| --- | ---: | --- |
| Ulmans V2 | 0.3915 | board 0.424 = accuracy |
| StreamV3Max | 0.4319 | soft MS text |
| StreamV3Push | 0.4445 | prior champion |
| **StreamV3PushEmbed** | **~0.4570** | Push 85% + pretrain embed LR 15% |
| Moonshot restrict/diri | ≤0.442 | no beat |
| Softmap mixes | ≤0.4415 | no beat |
| Precise temporal alone | ~0.30 | high recall path, low precision |
| Precise oracle ceiling | ~0.56 | coverage only ~39% |
| Candidate oracle | ~0.77 | uses labels among cands |

## Why 0.70 is hard (not hopeless)

0.70 sits between Push and the candidate oracle. It needs a **disambiguation signal stronger than merchant identity + cadence**, because identity is ~35% pure. Plausible remaining bets:

- Unlabeled pretrain for better merchant clustering / sequence models (in flight).
- Pseudo-cutoff self-supervision (Jaime) to train a true next-date ranker without VALID labels.
- Cross-fitted description multi-hot inside identity-style CatBoost (not kitchen-sink dumped).
- None detector as its own binary problem (289/293 none clients still have mapped recurrents in horizon — huge FP).

## Reproduce searches

```bash
python scripts/run_ubs_v3_pretrain.py --push-weight 0.85
python scripts/experiments/esteban_v3_moonshot.py
python scripts/experiments/esteban_v3_softmap.py
python scripts/experiments/esteban_v3_next.py
```

Artifacts under `outputs/metrics/v3_*` (gitignored).
