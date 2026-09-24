# Discovery handoff: unlabeled pretrain (10k clients)

**Branch:** `discovery/esteban-pretrain`
**Base:** `5884ddd07a19b2e19b44b4fd55afdf4eb6dcf749` (V2 integration)
**Code:** `scripts/experiments/esteban_pretrain.py`, `src/transaction_forecasting/ubs/pretrain_priors.py`
**Metrics:** `outputs/metrics/v3_discovery/esteban_pretrain/`

## Question

Does the official **unlabeled_pretrain** partition (10 000 clients, no labels) add incremental predictive signal on top of train histories for the current V2 system?

## Rule compliance

- Priors fitted **without labels** (recurrence rates, periods, gap/amount CV, MCC/type/currency modes, weekly/biweekly/monthly prevalence).
- No pseudo-labeling from V2.
- No validation labels in prior fitting or γ selection.
- No `client_id` features; no large grid (γ ∈ {0, 0.05, …, 0.25} on **train OOF only**).
- CatBoost recipe and V2 component probabilities are **identical** across A/B/C; only the prior *source* and the per-client blend weight change.

## What was learned from the population

Priors cover **1 475** train descriptions → **2 673** with unlabeled (+1 198).
Unlabeled covers **274 / 407 (67%)** of validation descriptions that are unseen in train.

Population tables (see `priors_train_plus_unlabeled.csv`, `top_recurrent_descriptions_C.csv`) expose:

| Signal | Finding |
| --- | --- |
| Global recurrence | Many everyday merchants (salary, ATM, grocery, coffee) have high `P(≥2 appearances)` — **not** subscription-specific. |
| Periodicity | Monthly/quarterly mass dominates among multi-hit streams; weekly/biweekly are rarer. |
| Stability | Gap CV and amount CV priors exist per description with Laplace smoothing (α=5). |
| Rare/unseen | Pretrain fills vocabulary holes on valid, but those descriptions are often generic billing strings. |

So the 10k file **is** informative as a **population lexicon**; that does not automatically translate into Macro-F1.

## Ablation design

| Arm | Definition |
| --- | --- |
| **A** | Frozen V2: 75% history CatBoost + 25% periodicity heuristic |
| **B** | Same V2 components; heuristic weight = `0.25 + γ · trust(priors_train)` |
| **C** | Same V2 components; same frozen γ; `trust(priors_train+unlabeled)` |

`trust` is unsupervised: recurrence lift vs global rate, gap stability, and short-history period match.

### Failed approaches (documented, not used for the headline)

1. **CatBoost + prior features** — train OOF looked strong (~0.44–0.46) but **valid Macro-F1 collapsed to ~0.06–0.15** (almost all `none`). Even two features (`mean/max P(recurrent)`) broke generalization via train/valid description-mix shift.
2. **Anti-none logit nudge** — train OOF selected **β = 0** (any β > 0 hurt).

These failures matter: naive “add unsupervised features” **harms** the current system.

## Final VALID results (frozen γ = 0.25 from train OOF)

| Arm | Macro-F1 | Accuracy |
| --- | ---: | ---: |
| **A** V2 baseline | **0.3915** | 0.424 |
| **B** train priors only | **0.3978** | — |
| **C** train + unlabeled | **0.3978** | — |

| Contrast | Δ Macro-F1 |
| --- | ---: |
| A → B (train priors help adaptive blend) | **+0.0063** |
| **B → C (incremental value of 10k unlabeled)** | **0.0000** |
| A → C | +0.0063 |

### F1 by class (VALID)

| Class | A | B | C | Δ(C−B) |
| --- | ---: | ---: | ---: | ---: |
| cloud | 0.4500 | 0.4596 | 0.4596 | 0 |
| gym | 0.4873 | 0.4873 | 0.4873 | 0 |
| insurance | 0.4291 | 0.4400 | 0.4400 | 0 |
| mobile | 0.4527 | 0.4508 | 0.4508 | 0 |
| music | 0.1699 | 0.1722 | 0.1733 | **+0.0011** |
| none | 0.5194 | 0.5197 | 0.5187 | −0.0010 |
| software | 0.3740 | 0.3725 | 0.3725 | 0 |
| streaming | 0.2500 | 0.2805 | 0.2805 | 0 |

- **Classes benefited B→C:** music (tiny)
- **Classes hurt B→C:** none (tiny)
- **Prediction flips B→C:** 1 client
- **Music / streaming:** unlabeled does **not** move streaming; music +0.001 F1 only.

### Prediction distribution (A = B ≈ C at selected γ)

`cloud:71 gym:154 insurance:148 mobile:139 music:60 none:223 software:142 streaming:63` (A; B/C nearly identical)

### Forced-γ VALID grid (reporting only; not used for selection)

At every γ ∈ [0, 0.25], **Δ(C−B) ∈ [−0.001, +0.0007]** — unlabeled never delivers a meaningful gain over train-only priors.

Train OOF grid favored larger γ (0.25 → OOF MF1 0.4189 vs 0.4089 at γ=0), which is why B/C beat A slightly — that lift comes from **train** population stats, not from the extra 10k.

## Incremental value of the 10 000 unlabeled clients

**Quantified answer:** under a clean A/B/C design that keeps the V2 model fixed and only swaps the prior source,

\[
\Delta_{\text{Macro-F1}}(B \to C) = 0.0000
\]

Unlabeled **expands description coverage** (274/407 valid-unseen strings) and densifies priors, but those denser priors **do not improve** the frozen V2 decision rule beyond train-only priors. Mean `trust_periodicity` even drops slightly (B 0.152 → C 0.137).

## Conclusion

**PRETRAIN DOES NOT IMPROVE THE CURRENT SYSTEM**

More precisely:

- Unlabeled pretrain is a useful **lexicon / coverage** resource.
- It is **not** a major (or even small) Macro-F1 source for the current V2 pipeline when isolated cleanly.
- Train-only unsupervised priors can give a **small** adaptive-blend lift over vanilla V2 (+0.006 Macro-F1); that must not be attributed to the 10k file.

## Reproduce

```bash
python scripts/experiments/esteban_pretrain.py \
  --data-dir data/raw/ubs_2026 \
  --output-dir outputs/metrics/v3_discovery/esteban_pretrain
```

## Tests / lint

- `pytest tests/test_pretrain_priors.py tests/test_ubs_v2.py`
- `ruff check` + `ruff format --check` on touched files
