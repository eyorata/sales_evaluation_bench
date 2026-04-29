# Cost Log — Tenacious-Bench v0.1

**Date:** 2026-04-29  
**Budget Envelope:** $10

---

## Budget Summary

| Bucket | Budget | Spent | Remaining |
|--------|--------|-------|-----------|
| Dataset authoring | $3-5 | $2.87 | $1.13 |
| Training | $0-5 | $0.00 | $5.00 |
| Held-out evaluation | $2-3 | $0.00 | $2.50 |
| Reserve | $1-2 | $0.00 | $1.50 |
| **Total** | **$10** | **$2.87** | **$7.13** |

---

## Dataset Authoring Costs

| Date | Purpose | Model | Cost | Notes |
|------|---------|-------|------|-------|
| 2026-04-29 | Multi-LLM synthesis (hard seeds) | DeepSeek-V3.2 | $0.45 | ~50 tasks |
| 2026-04-29 | Bulk programmatic variations | Qwen3-Next-80B-A3B | $0.62 | ~80 tasks |
| 2026-04-29 | Judge filtering | Qwen3-Next-80B-A3B | $0.38 | ~254 tasks |
| 2026-04-29 | Dev-tier iteration | DeepSeek-V3.2 | $0.42 | ~30 tasks |
| 2026-04-29 | Contamination check | all-MiniLM-L6-v2 | $0.00 | Free (local) |
| 2026-04-29 | Judge scoring (held_out) | Qwen3-Next-80B-A3B | $1.00 | ~66 tasks |
| **Subtotal** | | | **$2.87** | |

---

## Training Costs (Reserved)

| Date | Purpose | Model | Cost | Status |
|------|---------|-------|------|--------|
| TBD | SimPO/ORPO training run | Colab T4 | $0.00 | Not started |
| TBD | Ablation runs | Eval-tier | $0.00 | Not started |

---

## Held-Out Evaluation Costs (Reserved)

| Date | Purpose | Model | Cost | Status |
|------|---------|-------|------|--------|
| TBD | Delta A (trained vs baseline) | Claude Sonnet 4.6 | $0.00 | Not started |
| TBD | Delta B (trained vs prompt) | Claude Sonnet 4.6 | $0.00 | Not started |
| TBD | Delta C (vs τ²-Bench) | Claude Sonnet 4.6 | $0.00 | Not started |

---

## Reserve Usage

| Date | Purpose | Cost | Notes |
|------|---------|------|-------|
| - | - | $0.00 | Not used |

---

## Cost Discipline Notes

1. **No τ²-Bench retail re-runs:** Week 10 score reused as baseline
2. **No eval-tier on Days 2-3:** Dev-tier used exclusively for iteration
3. **Free compute:** Contamination checks run locally (no API cost)
4. **Colab T4:** Training will use free GPU (no compute cost)

---

## Compliance

- [x] No τ²-Bench retail validation runs
- [x] No eval-tier model on Days 2-3
- [x] All API charges recorded with timestamp and bucket
- [x] Cost log committed alongside dataset

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-04-29 | Initial cost log |