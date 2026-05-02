# Memo to the Tenacious CEO and CFO

**From:** Eyoel Nebiyu (10Academy TRP1) **Date:** 2026-05-02
**Subject:** Tenacious-Bench v0.1 + Path B Judge — Decision and Skeptic's Appendix

## Page 1 — The Decision

**Executive summary.** Tenacious-Bench v0.1 is a 266-task evaluation benchmark that grades B2B sales-outreach agents on ten Tenacious-specific failure dimensions no public benchmark covers. A SimPO-tuned 3 B-parameter LoRA judge on 128 LLM-rewritten preference pairs lifts held-out accuracy from **0.167 (untrained baseline) to 0.417 — Delta A = +25 pp, 95 % CI [0.00, 0.50], paired-bootstrap p = 0.0316** on n = 12. **However, the same Qwen2.5-3B backbone with no training and a Tenacious-rubric system prompt scores 0.833 on the same slice, beating the LoRA by 42 pp** — recommendation: **deploy with caveat**, deploying the prompt-engineered judge (not the LoRA) under the Page-2 trigger conditions.

### Headline numbers

| Metric | Value |
|---|---:|
| Trained LoRA judge held-out accuracy (n = 12) | **0.417** |
| Untrained backbone (Δ-A baseline) | 0.167 |
| Prompt-engineered same-backbone (Δ-B comparator) | **0.833** |
| **Delta A** (trained vs. untrained) | **+25 pp**, 95 % CI [0.00, 0.50], paired bootstrap **p = 0.0316** |
| **Delta B** (trained vs. prompt, same backbone, same intervention shape) | **−42 pp**, 95 % CI [−0.75, 0.00], p = 0.992 |
| Per-judgment latency (trained / prompt) | 417 / 372 ms (+45 ms) |
| Per-judgment cost on RunPod 4090 (trained / prompt) | $3.9e-5 / $3.5e-5 (+$0.04 / 1 000 judgments) |
| Total Week 11 build spend | $0.041 of $10 envelope |

### Recommendation: **deploy with caveat**

Deploy the **prompt-engineered judge** (not the LoRA) as a rejection-sampling layer running *after* `policy.py`. Cost: ~$0.018/week at 500 outreach/week, +45 ms latency. Gating conditions:

1. **Lift threshold:** weekly held-out accuracy ≥ **0.73** (10 pp tolerance below 0.833 launch baseline). Below this, disable; root-cause within 48 h.
2. **Cost ceiling:** per-judgment cost **< $0.0001**, p95 latency **< 800 ms** (currently 4× and 2× under).
3. **Data prerequisite:** collect 30+ Tenacious-graded outreach examples to expand held-out from 12 → 42 before the next training run. n = 12 is the binding constraint on Delta A's CI lower bound.
4. **Re-evaluate gate:** re-train on Qwen2.5-7B and re-run Delta B at n = 42; if Delta B flips positive, swap prompt judge for LoRA.

Recommendation rests on Delta B (−42 pp, p = 0.99) being a stronger signal than Delta A (+25 pp, p = 0.03): a negative Delta B with a tight CI dominates a positive Delta A whose CI grazes zero.

## Page 2 — Skeptic's Appendix

### Four behaviors v0.1 has zero tasks against — what v0.2 must add

1. **Multi-turn trajectory drift.** v0.1 grades single-turn drafts only. A 5-turn thread where each draft passes but cumulative drift toward over-commitment ("~7 Python" → "probably 10" → "yes, 12") is invisible. **v0.2:** `multi_turn_trajectory` source mode, PRM-style step-level annotations on 30 trace reconstructions.
2. **Warm-reply objection handling.** Only 1 warm-reply task in held-out (SG-07). Cannot grade common B2B objections (*"we have a vendor"*, *"send pricing"*, *"we don't outsource"*). **v0.2:** `warm_objection_handling` partition of 30+ tasks anchored in `tenacious_sales_data/seed/discovery_transcripts/`.
3. **Cross-account confidentiality.** v0.1 multi-thread probes cover same-prospect-two-contacts only — not cross-account leakage (agent must never name Account A's data in an email to Account B). **v0.2:** `cross_account_isolation` family with two-prospect scenarios + "no cross-naming" rubric.
4. **Follow-up cadence calibration.** 14 scheduling tasks, none grade re-engagement *cadence* (3 days too soon = needy; 30 days too late = cold). **v0.2:** `cadence_calibration` partition with time-progressed prior threads + checks on proposed next-touch timestamps.

### Public-signal lossiness in the ground truth

12 held-out pairs use Style Guide v2 as ground truth — public-signal-grounded, **not deal-outcome-grounded**. Two specific mechanisms on existing tasks:

- **Hiring-signal lag.** Crunchbase + BuiltIn lag actual hiring by 30–90 days. A draft grounding on "your $14M Series A in February" is rewarded even when the prospect closed Series B in March that public data hasn't picked up. **The 0.833 headline systematically over-rewards drafts anchored on stale signals.**
- **No reply-rate ground truth.** Rubric measures *correctness*, not *effectiveness*. A perfectly graded draft can score 1.0 and get zero replies. **The headline under-represents confidence-aware phrasing on weak-signal prospects (correlates with reply rate) and over-represents grounding-signal naming completeness.**

### One unresolved training failure (training-process artifact)

**SimPO at 128 preference pairs cannot encode what a system prompt encodes for free.** v1 (1.5B + templated pairs) hit 1.00 train acc in 5 steps and stayed flat at 0.25 held-out — overfit to template artifacts. v2 (3B + Llama-3.3-70B-rewritten pairs) lifted held-out to 0.42 but still loses Delta B by 42 pp. **Tried, did not resolve:** bigger backbone (1.5B→3B), real LLM-rewritten chosens, longer training (200→300 steps), higher LR (8e-6→1e-5). **Try next:** scale to 1 024+ pairs (~$0.40 incremental rewrite spend) and switch to ORPO with SFT regularization to constrain the model to the prompt-aligned manifold rather than shortcut-fitting the small training set.

### Kill-switch trigger conditions

Deployed behind `TENACIOUS_JUDGE_ENABLED=true`. Each trigger observable in production without re-running held-out.

| Trigger | Threshold | Justification | Action |
|---|---|---|---|
| False-approval rate (staff audit) | > **10 %** / rolling 30-draft sample | Launch disagreement ~17 %; healthy week ≤ 10 % | Disable judge; route to staff review |
| Latency p95 | > **800 ms** | 2× the 372 ms launch baseline | Disable judge; fall back to `policy.py` |
| Reply-rate drop | > **20 %** rolling 14-day vs prior 14-day | Judge must improve, not hurt, conversion | Disable judge; run 50/50 A/B 4 weeks |
| Disqualifying violation in production | **Any 1** bench-overcommit, signal fabrication, or post-opt-out re-engagement | Worst-case public-trust events | Auto-flip flag; audit prior 24 h; root-cause 48 h |

— Eyoel Nebiyu
