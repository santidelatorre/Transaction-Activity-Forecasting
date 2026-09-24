# V4 self-supervised representations — Esteban

**Branch:** `exp/v4-self-supervised-esteban`
**Base:** `main` @ `051ce64a7cf0ab999f8aacb81fa405d5fa0257cf`
**Base V3-A:** TRAIN OOF Macro-F1 `0.459794`, VALID Macro-F1 `0.424111`

## Decision

Keep V3-A. The clean SVD client vector beats label-free stats (0.4391 vs 0.1498 TRAIN OOF) but loses to V3-A on OOF (0.4391 vs 0.4598) and on the one VALID pass (0.3219 vs 0.4241). Reject the ridge denoiser: drift gain is 0.0020 and raw-corrupt OOF (0.3886) matches denoised OOF (0.3879). Do not start a GRU or Transformer.

Phase 2 (GRU / temporal CNN / Transformer) was not trained. PyTorch is not installed, and the brief requires a real phase-1 signal before any sequence model.

## Pretraining (no challenge labels)

Char TF-IDF (3–5) + TruncatedSVD fit on up to 25k unique descriptions drawn from an 80k-row unlabeled sample plus TRAIN histories. A ridge map sends a 35% character-dropped embedding back to the clean SVD. `client_id` is not a feature. VALID labels were not used to pick the encoder or the clean-view head (selection is max TRAIN OOF among stats, clean embeddings, and clean embeddings + stats).

- SVD components: `32`
- Descriptions used in the text fit: `1690`
- Parameter bytes (SVD + denoiser + IDF): `215432`
- Device: CPU only (`arm`), torch=`False`
- Runtime: `37.0` s

### Task metrics

- Next-gap probe accuracy `0.5391` vs majority `0.5288` (lift `0.0103`, n=`12000`).
- Drift L2 raw-corrupt `0.4510` vs denoised `0.4490` (gain `0.0020`).
- Contrastive cosine same-client `0.9769` vs cross-client `0.1244` (margin `0.8525`).

## Downstream TRAIN OOF Macro-F1

| View | Macro-F1 |
| --- | ---: |
| stats | 0.1498 |
| clean_embed | 0.4323 |
| clean_embed_plus_stats ** | 0.4391 |
| denoised_corrupt | 0.3879 |
| raw_corrupt | 0.3886 |

Chosen on OOF only: `clean_embed_plus_stats` (0.4391).

## Frozen VALID

Macro-F1 `0.3219`, accuracy `0.3280`. V3-A VALID is `0.4241`.

Per-class F1 (chosen, clean VALID):

- cloud: 0.4424
- gym: 0.3422
- insurance: 0.2411
- mobile: 0.3814
- music: 0.2184
- software: 0.3203
- streaming: 0.2928
- none: 0.3367

## Corruption

Same linear head, trained on clean features, scored on corrupted VALID inputs.

- denoised_corrupt: VALID Macro-F1 `0.0858`
- raw_corrupt: VALID Macro-F1 `0.0815`

- denoised_corrupt: TRAIN OOF Macro-F1 `0.3900` (head refit on that view)
- raw_corrupt: TRAIN OOF Macro-F1 `0.3883` (head refit on that view)

A low reconstruction error is not treated as success. The stability gain and the Macro-F1 drop under corruption are the checks that matter.

## Integration

OOF probabilities: `outputs/metrics/v4_selfsupervised_esteban/oof_probabilities.csv` (gitignored).
Summary JSON sits beside that file.

```bash
python scripts/experiments/v4_selfsupervised_esteban.py
```
