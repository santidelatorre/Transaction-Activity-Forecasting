# Subscription Foresight — demo README

Jury-facing demo for Ulmans PushEmbed (Macro-F1 **0.457**).

## Quick path (no model fit)

```bash
python scripts/demo_subscription_foresight.py metrics
python scripts/demo_subscription_foresight.py clients --pack valid
python scripts/demo_subscription_foresight.py explain --client C000000
python scripts/demo_subscription_foresight.py tour --client C000000
```

`predict` needs a warm cache (or `--fit-if-needed`, slow):

```bash
# full VALID cache (~4–7 min fit once)
python scripts/demo_subscription_foresight.py warm
# smoke / first 30 VALID clients (covers C000000)
python scripts/demo_subscription_foresight.py warm --max-valid 30
python scripts/demo_subscription_foresight.py predict --client C000000
python scripts/demo_subscription_foresight.py tour --client C000000
```

Cache lands in `outputs/demo/` (gitignored).

## Tools (agentic spine)

| Tool | CLI | Purpose |
| --- | --- | --- |
| `show_metrics` | `metrics` | Frozen scoreboard + honesty notes |
| `list_clients` | `clients` | Sample VALID/TRAIN ids |
| `explain_streams` | `explain` | Recurring/due timeline narrative |
| `predict_client` | `predict` | Next family + probabilities |
| `warm` | `warm` | Fit TRAIN → cache VALID probs |

Implementation: `src/transaction_forecasting/ubs/demo_agent.py`.

## Reproduce the score

```bash
python scripts/run_ubs_v3_pretrain.py --push-weight 0.85
```

## Pitch artifacts

- 1-min script: `reports/handoff/v3_discovery/esteban_pitch_1min.md`
- Q&A ↔ jury criteria: `reports/handoff/v3_discovery/esteban_pitch_qa.md`
- Model handoff: `reports/handoff/v3_discovery/esteban_v3_pretrain.md`

## Leakage / honesty

- Predictions use only pre-cutoff transactions.
- `--reveal-label` is display-only for internal QA.
- Board **0.424** = V2 accuracy; we report Macro-F1.
- Oracle **~0.77** is diagnostic, not our submission.
