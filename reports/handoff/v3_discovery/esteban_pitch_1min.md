# Subscription Foresight — 1-minute Expert Jury pitch

**Team:** Ulmans · UBS 2026 challenge
**Product name:** Subscription Foresight
**Metric to say out loud:** Macro-F1 **0.457** (not the board’s 0.424 — that is V2 **accuracy**)

## Spoken script (~55–60s)

Banks don’t just need fraud alerts — they need to know **which recurring bill hits next**.

Subscription Foresight predicts, per client, the **next recurring merchant family** in the 90 days after cutoff: cloud, gym, insurance, mobile, music, software, streaming — or none.

We don’t guess from a single merchant name. Signature merchants are only ~35% pure. So we stack three complementary AI signals — history+periodicity, due-stream geometry, cross-fitted family identity — then blend with **unlabeled pretrain embeddings**. Official VALID Macro-F1: **0.457**, up from Ulmans V2 **0.392**.

The demo is agentic: load a client, explain their recurring timeline, predict the next family, show calibrated probabilities — tools a bank ops or wealth desk could call.

Impact: cashflow foresight, retention on at-risk subscriptions, fewer surprise overdrafts. Reproducible pipeline, modern ML stack, challenge objective nailed.

We’re Ulmans — Subscription Foresight.

## Timing cues

| 0:00–0:10 | Problem (bank / next bill) |
| 0:10–0:25 | What we predict (8 families + none) |
| 0:25–0:40 | Why hard + our stack + 0.457 |
| 0:40–0:55 | Demo agent + market impact |
| 0:55–1:00 | Close: name + ask for Q&A |

## Demo command (screenshot before pitch)

```bash
python scripts/demo_subscription_foresight.py metrics
python scripts/demo_subscription_foresight.py tour --client C000000
# if predict says no cache:
python scripts/demo_subscription_foresight.py warm   # once
python scripts/demo_subscription_foresight.py predict --client C000000
python scripts/demo_subscription_foresight.py explain --client C000000
```
