# Q&A cheat sheet → jury criteria

Source criteria: https://zh.ai-weeks.ch/jury-process-and-criteria

Expert round: **1 min pitch + 3 min Q&A**. Keep answers ≤30s.

---

### 1) Technical Functionality & AI
**Q: Does it work? Is it reproducible?**
A: Yes. Official VALID Macro-F1 **0.457** with `StreamV3PushEmbed`. Reproduce: `python scripts/run_ubs_v3_pretrain.py --push-weight 0.85`. Train/valid clients are disjoint at predict time; cutoff leakage checks on load.

**Q: Why not just keyword “Spotify → music”?**
A: Merchants like `urban gym` / `audio streaming` co-occur with the true family only ~30–38%. We use stacked models + pretrain embeds, not hard rules.

**Q: What’s 0.424 on the board?**
A: Ulmans V2 **accuracy**. Our metric is **Macro-F1**. V2 Macro-F1 is **0.392**; we are at **0.457**.

**Q: What’s the 0.77 number?**
A: Diagnostic **oracle ceiling** if you knew which candidate family is true — not a submission. Shows headroom; our deployable model doesn’t peek at labels.

---

### 2) User Experience
**Q: Who uses this?**
A: Bank ops / wealth / cashflow teams — one client → next recurring family + timeline of due streams. CLI tools: `metrics`, `explain`, `predict`.

**Q: Show me.**
A: Run `demo_subscription_foresight.py tour --client C000000` — JSON narrative + probabilities, not a spreadsheet dump.

---

### 3) Agentic Depth
**Q: Where are the agents?**
A: A tool-calling demo agent: `show_metrics`, `explain_streams`, `predict_client`, `list_clients`, `warm`. Same spine can sit behind a bank copilot later.

---

### 4) Originality & Fun
**Q: What’s new?**
A: “Subscription Foresight” — treating next recurring merchant as a **cashflow event**, blending due-stream geometry with **unlabeled pretrain** embeddings. Fun angle: we literally forecast the next subscription that hits your account.

---

### 5) Potential & Market Impact
**Q: Beyond the hackathon?**
A: Overdraft prevention, subscription retention, liquidity buffers for SMEs, alerts before a gym/insurance charge. Fits UBS challenge: next recurring merchant family in 90 days.

**Q: Can it grow?**
A: Same stack extends to amount/date estimates (demo-optional), multi-cutoff monitoring, and agent workflows for advisors.

---

### Trap doors (don’t say)
- Don’t claim 0.70 or oracle 0.77 as our score.
- Don’t say we hardcode VALID merchants.
- Don’t confuse accuracy with Macro-F1.
