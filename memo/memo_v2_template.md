# Memo to Tenacious CEO and CFO

**From:** Eyoel Yorat (10Academy TRP1)
**Date:** 2026-05-02
**Subject:** Tenacious-Bench v0.1 + Path B Judge — Decision and Skeptic's Appendix
**Length:** 2 pages

> *Final-render note: PDF generated from this Markdown. Page break before §"Skeptic's Appendix" enforces the 2-page constraint. `[TBD]` placeholders fill from `ablation_results.json` once the v2 training run lands.*

---

## Page 1 — The Decision

**Three sentences.** Tenacious-Bench v0.1 is a 266-task evaluation benchmark that grades B2B sales-outreach agents on ten Tenacious-specific failure dimensions that no public benchmark covers. A SimPO-tuned 3B-parameter LoRA judge, trained on 128 LLM-rewritten preference pairs anchored in the v2 Style Guide, lifts held-out preference accuracy by `[TBD: Delta A pp]` (95% CI `[TBD: ci_low, ci_high]`, paired bootstrap p=`[TBD]`) over the untrained backbone — though the eval slice is small (n=12 ground-truth pairs) and the lift is best read as directional. Production recommendation: **`[TBD: deploy / deploy with caveat / do not deploy]`**.

### Headline numbers

| Metric | Value |
|---|---|
| Held-out preference accuracy, trained judge | `[TBD]` |
| Held-out preference accuracy, untrained backbone (Δ-A baseline) | `[TBD]` |
| **Delta A** (raw lift) | `[TBD pp]` (95% CI `[TBD]`, p=`[TBD]`) |
| **Delta B** (vs prompt-engineered same-backbone) | `[TBD pp]` (95% CI `[TBD]`, p=`[TBD]`) |
| Per-judgment latency (trained judge) | `[TBD ms]` |
| Per-judgment latency (prompt-engineered baseline) | `[TBD ms]` |
| Cost per 1,000 judgments at RunPod 4090 | `[TBD] cents` |
| Total Week 11 spend | $0.04 of $10 envelope |

### What changes in production

| Component | Today | Recommended |
|---|---|---|
| Outreach generation | Week 10 generator (Qwen 2.5 + policy.py) | unchanged |
| Pre-send check | `policy.py` regex + heuristic tone score | **add Path B judge as rejection-sampling layer**: regenerate iff judge prefers the rejected variant in a (chosen, rejected) draft pair |
| Held-out scoring of weekly outreach | none | weekly Tenacious-Bench held-out pass; alert when score drops > 5pp |

The Path B judge is not a replacement for `policy.py`. It runs *after* the policy regex in series. The policy module catches deterministic violations (banned phrases, capacity over-commit arithmetic). The judge catches the residual: tone drift, condescending phrasing, signal-confidence misalignment — categories where regex is necessary but not sufficient.

### Cost envelope

- **Week 11 build cost**: $0.041 (OpenRouter dev-tier for synthesis + judge + Llama-3.3-70B chosen-rewrite).
- **Training cost**: $0 (Colab T4 free tier; 55 min wall time on Qwen2.5-3B + LoRA).
- **Production marginal cost**: `[TBD ¢/judgment]`. At Tenacious's projected ~500 outreach/week, the judge layer adds `[TBD]/week`.

### Delta C (informational, not a claim)

Re-using the Week 10 τ²-Bench retail score (pass@1 = 0.7267, 95% CI [0.6504, 0.7917], 30 tasks × 5 trials) per the brief's no-re-run rule. The trained judge was not trained on retail-domain tasks; we do not claim transfer. Tenacious-Bench lift is Tenacious-specific by construction.

---

## Page 2 — Skeptic's Appendix

If Tenacious's CFO trusts a positive lift number from a single eval pass on n=12 held-out pairs without reading this section, that is the kind of trust that ends in a refund. Four things the v0.1 bench *does not* capture, and one thing I left unresolved.

### Four failure modes Tenacious-Bench v0.1 still does not grade

1. **Multi-turn trajectory failure.** v0.1 grades single-turn drafts. A 5-turn email thread where each individual draft is fine but the *sequence* drifts toward over-commitment (a "boiling frog" pattern) will score 5/5 on every turn and still lose the deal. v0.2 needs a process-reward-model component (Path C territory).

2. **Personalization-staleness drift.** v0.1 has a static `prior_thread` field. Real production drift comes from a 90-day-stale signal brief getting cited as fresh. The hand-authored adversarial slice (TB-0610 family) covers a few cases, but a longitudinal eval would need synthetic time progression.

3. **Reply-rate ground truth.** Tenacious's actual ROI metric is reply rate. v0.1 grades *whether the draft is correct*, which correlates with reply rate but is not equal to it. We have no production reply-rate data because Tenacious doesn't run a randomized A/B over outreach styles. Adding a small held-out slice of Tenacious's actual sent emails (with redaction) is the v0.3 ask if Tenacious is willing.

4. **Cross-cultural tone calibration.** The Style Guide v2 markers are calibrated to North American and European founders. The Eid-al-Fitr scheduling task (TB-0614) is the only nod to non-Western context. If Tenacious starts targeting LATAM or APAC at scale, the tone rubric needs a regional layer.

### Public-signal lossiness in our ground truth

The 12 held-out style-guide pairs are *our* labels of what Tenacious tone is. If the CEO would rewrite three of them differently, the ground truth shifts and so do all the held-out numbers. I labeled them twice 24 hours apart with 100% within-±1 agreement, but that's *intra-rater*, not *inter-rater with the CEO*. A 30-minute calibration session with the CEO on the 12 pairs is the lowest-cost ground-truth audit we can run.

### One honest unresolved failure

The v1 training run trained perfectly (1.00 train accuracy) and scored 25% on held-out. Diagnosis took half a day. The failure was templated synthetic chosen outputs that the model overfit to. v2 fixed the data via real LLM rewrites. **The failure mode is real and could recur** — any future re-training pipeline needs the chosen outputs gated by `scoring_evaluator >= 0.7` and the held-out preference accuracy plotted alongside training loss from step 1. If they diverge for more than 50 steps, kill the run.

### Kill-switch trigger

The trained judge ships behind a runtime flag (`TENACIOUS_JUDGE_ENABLED=true`). Trigger conditions for flipping the flag off:

1. Held-out preference accuracy drops below `[TBD: trained_acc - 0.10]` (10pp degradation from launch baseline) on the weekly eval pass.
2. The judge's average per-judgment latency exceeds 600ms on the production GPU (2× the launch baseline).
3. A sampled 30-task manual audit by Tenacious staff shows judge-rejected drafts that staff would have approved at >20% rate.
4. Any disqualifying violation (capacity fabrication, signal hallucination, opt-out re-engagement) ships to a real prospect — even one — auto-flips the flag and triggers human review of the prior 24 hours of outbound.

### One thing I'd ask the CEO and CFO

What's the bar that makes this judge worth deploying — a positive Delta A at p<0.05, a 5pp accuracy lift on a calibrated 30-pair held-out slice, or proof that *no* production draft would have been falsely rejected over 1,000 historical sends? Each bar implies a different next-week's work. I have a clear plan for each; the right question is which one you'd accept.

— Eyoel Yorat
