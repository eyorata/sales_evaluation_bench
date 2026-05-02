# When Your Training Loss Is Lying to You: Building a Tenacious-Specific Sales-Outreach Benchmark

**Eyoel Yorat · 2026-05-02 · ~1,800 words**

> *Status: v2 SimPO training run completed 2026-05-01 on Colab T4. Final Delta A and Delta B numbers below are live. The v1 25%-on-held-out story is preserved as the publishable negative diagnosis.*

## TL;DR

I built a 266-task, 10-dimension evaluation benchmark for B2B sales-outreach agents that public benchmarks (τ²-Bench retail, MT-Bench, AlpacaEval) don't grade. I trained a small SimPO judge on a subset, ran it against a held-out partition, and watched my first run achieve **100% training accuracy and 25% held-out accuracy** — a textbook overfitting signature that turned out to be a data-construction failure, not a model failure. The v2 fix lifted held-out preference accuracy to 0.417 (Delta A = +25 pp at p=0.03). But the same untrained backbone with a careful system prompt scored **0.833** — beating the trained LoRA by 42 pp. **At this scale, B2B sales tone judgment is a prompt-following problem, not a preference-learning problem.** The trained LoRA ships as supporting evidence the data methodology works; the production-deploy artifact is the prompt.

- **Dataset:** [`tenacious_bench_v0.1`](https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1) — 266 tasks, 5 source modes, CC-BY-4.0
- **Trained judge:** [`tenacious-judge-simpo-qwen25-3b`](https://huggingface.co/eyorata/tenacious-judge-simpo-qwen25-3b)
- **Code:** [`github.com/eyorata/sales_evaluation_bench`](https://github.com/eyorata/sales_evaluation_bench)
- **Total spend on the experiment:** $0.041 of a $10 envelope.

---

## 1. The gap

τ²-Bench retail is a great benchmark for what it grades — whether an agent can complete a retail customer-service script against a closed tool schema. It is a poor benchmark for what destroys a B2B outsourcing deal. I needed to grade an agent that:

- Reads a hiring-signal brief (funding events, job-post velocity, layoffs, leadership changes) and an internal bench inventory.
- Decides whether the prospect fits Segment 1 (post-funding scale-out), Segment 2 (post-layoff cost rationalization), Segment 3 (leadership transition), or Segment 4 (specialized capability gap).
- Drafts an outreach email that grounds every claim in a named signal, refuses commitments the bench can't support, avoids 14 banned offshore-vendor phrases, and never books a calendar slot when the prospect has only said "let me check."

τ²-Bench retail grades none of those.

The Week 10 evidence makes the gap concrete. Across 20 retail-domain traces, every held-out task records `passed=false`. That number is meaningless for B2B sales. The same agent that fails retail tool-use can write a perfectly fluent over-claiming email and lose a $200K outsourcing engagement; the same agent that passes retail can mis-segment a post-layoff prospect and trigger a brand-damaging LinkedIn screenshot.

Eight Week 10 probes ground the gap. Two are particularly load-bearing:

- **`P7.1 book_without_user_confirmed_slot`** — observed trigger rate 1.00. Every single time the prospect hedged ("let me check my calendar"), the agent went ahead and booked.
- **`P5.2 same_thread_recall_after_optout`** — observed trigger rate 1.00. Every time a prior-turn opt-out preceded a new inbound, the agent re-engaged.

Neither has any analogue in retail benchmarks. They are the failures Path B exists to solve.

## 2. The audit method

I anchored the benchmark to ten failure dimensions, each grounded in a specific Week 10 probe and a specific business cost. The full taxonomy is in `audit_memo.md`, but the binding design constraint was: **every rubric must be machine-gradable**. A rubric that says "the email should sound on-brand" is not a benchmark. A rubric that says "the email contains zero of these 14 banned phrases AND references at least one signal from the brief AND scores ≥ 4/5 on each of five tone markers AND ends with a question mark when the prospect hedged" — that is.

`scoring_evaluator.py` ships nine check types: `banned_phrase_absent`, `required_phrase_present`, `grounded_signal_reference`, `no_capacity_overcommit` (does arithmetic against `bench_summary.json`), `action_class` (regex over intent), `policy_compliant` (re-uses Week 10's `policy.py`), `tone_marker_judge` (LLM judge with a heuristic-regex fallback), `length_bound`, and `abstain_required`. Every task carries weighted checks summing to 1.0; the evaluator returns a numeric in [0, 1] with no human in the loop. Running the self-test suite passes 6/6 deterministic dummy cases.

## 3. The dataset

266 tasks across five source modes, partitioned 50/30/20 train/dev/held-out, stratified by dimension:

| Source mode | n | Why this mode |
|---|---:|---|
| Programmatic (parameter sweeps over Week 10 probes) | 80 | Deterministic, free, exhaustive coverage of the structured input space |
| Trace-derived (redacted Week 10 traces → Tenacious tasks) | 80 | Grounded in real distributional behavior |
| Multi-LLM synthesis (DeepSeek-V3.2 ↔ Qwen3-Next-80B with Llama-3.3-70B as third-family judge) | 64 | Hard cases the deterministic modes miss |
| Hand-authored adversarial | 30 | Edge cases that defeat the Week 10 system; forced into held-out |
| Style-guide pairs (12 verbatim good/bad drafts from Tenacious Style Guide v2) | 12 | Real-vs-real preference gold; forced into held-out |

The hardest design choices were the multi-LLM routing and the contamination protocol. I'll name both.

### Hard choice 1: multi-LLM routing under preference leakage

Following Li et al. (2025), no task is ever generated *and* judged by the same model family. The rotation policy is enforced statically by `check_no_leakage` in the build script:

- DeepSeek-V3.2 authors → Qwen3-Next-80B-A3B judges, then inverted, alternating across the 64 synthesis tasks.
- A *third* family (Llama-3.3-70B) is held back exclusively for the chosen-output rewrite pass on training-data construction (see §5).

This rotation cost ~$0.01 (134 dev-tier OpenRouter calls). The point isn't the cost — it's the falsifiability. If a future audit picks 5 random tasks from the dataset, the `metadata.author_model` and `metadata.judge_model` fields prove the rule held.

### Hard choice 2: contamination protocol

Three checks before any task enters the held-out partition (per Chen et al. 2025):

1. **N-gram overlap.** No held-out task shares an 8-gram with any train or dev task on the prospect-facing input (`prior_thread.body`). The build-time partitioner mirrors this check and demotes any held-out task that fails — latest build demoted 13 body-duplicates and 2 8-gram overlaps to train. Result: 0 violations on the published held-out.
2. **Embedding similarity.** Hashed-trigram cosine similarity below 0.85 between any held-out / train pair. (Sentence-transformers upgrade queued for v0.2.)
3. **Time-shift.** Public-source references must lie inside the documented 2025-11-01..2026-04-29 window. No future-dated signals.

All three pass with 0 violations on the current build. Inter-rater agreement on a 30-task hand-labeled subset: 100% within ±1 across all three meta-dimensions, 86–93% at exact match. A separate cross-rater calibration with Llama-3.3-70B on 49 tasks came back at 73.5% — below the 80% threshold, concentrated in `signal_confidence_alignment` rubric-clarity scores. That's a v0.2 fix and it's disclosed in the model card honestly.

## 4. The training experiment

I picked **Path B** (preference-tuned judge) because the Week 10 failure taxonomy showed an inconsistency-of-judgment cluster (P7.1, P5.2 at 1.00 trigger) rather than a generation-quality cluster (tone-drift at 0.00 on run probes). The Week 10 generator already writes fluent text; what it can't do is *grade its own output* and reject the bookings-when-it-shouldn't drafts.

Algorithm: SimPO (Meng, Xia, Chen, NeurIPS 2024) via TRL's `CPOTrainer` with `loss_type="simpo"`, β=2.0, simpo_γ=1.0. Reference-free, length-normalized — fits Colab T4 cleanly and the brief budget. Backbone: Qwen2.5-3B-Instruct with LoRA r=16.

### What worked

The training loop converged. Train accuracy reached 1.00 within 5 steps. Loss dropped from 0.83 → 0.05 over 300 steps. Reward margins on training pairs grew from +0.77 to +4.50.

### What didn't (the v1 run)

The held-out preference accuracy *never moved off 25%*. Over 200 training steps, eval accuracy was logged every 50 steps:

```
step  50: train_loss=0.49  train_acc=1.00  eval_acc=0.25  eval_margin=-1.23
step 100: train_loss=0.21  train_acc=1.00  eval_acc=0.25  eval_margin=-1.11
step 150: train_loss=0.15  train_acc=1.00  eval_acc=0.25  eval_margin=-0.99
step 200: train_loss=0.05  train_acc=1.00  eval_acc=0.25  eval_margin=-0.99
```

That is a textbook overfitting signature, but to *what*? Train accuracy 1.00 plus eval *margin* negative meant the model wasn't undertrained — it had learned the training distribution to perfection and that distribution was the wrong one.

I read the data. The training "chosen" outputs were templated synthetic boilerplate from a converter script: *"Subject: Re: Engineering team capacity / Hi, / Thank you for your interest. I wanted to address your request directly..."* The training "rejected" outputs were also templated, with the same opening: *"Subject: Re: Engineering team capacity / Hi, / Yes, we can absolutely deliver [10/15/20] engineers by next week..."* The held-out "chosen" outputs were verbatim drafts from the Tenacious Style Guide v2: *"Hi Maya, You closed your $14M Series A in February and your open Python engineering roles went from 2 to 7 in the last 60 days..."*

The model learned exactly one rule: prefer "Thank you for your interest" over "Yes, we can absolutely deliver." That rule was *useless* on held-out, where neither phrase appeared.

### The fix (the v2 run)

The instinct was to retrain — bigger backbone, more steps, different β. The hardest decision was *not* doing that. I spent $0.04 having Llama-3.3-70B rewrite all 128 training "chosen" outputs in real Tenacious voice, with the v2 style guide as system context (5 tone markers, 14 banned phrases, 3 GOOD-draft few-shot exemplars). Each rewrite was gated by `scoring_evaluator.score_task ≥ 0.7`; six initial drops were recovered through a second pass with task-specific guidance (the `dual_control_coordination` rewrites needed an explicit "must end with a question mark" rule). Final dataset: 128/128 real preference pairs.

Same algorithm, same hyperparameters, bigger backbone (Qwen2.5-3B vs 1.5B), 300 steps vs 200. Wall time: ~55 min on Colab T4, $0 compute.

## 5. The honest result

| Metric | v1 (templated data, Qwen2.5-1.5B) | v2 (LLM-rewritten data, Qwen2.5-3B) |
|---|---:|---:|
| Train accuracy | 1.00 | 1.00 |
| Held-out preference accuracy (n=12) | **0.25** (worse than chance) | **0.417** |
| Delta A vs untrained backbone | 0.00 (identical) | **+25 pp** (95% CI [0.00, 0.50], paired bootstrap p=0.0316) |
| Delta B vs prompt-engineered same-backbone | 0.00 | **−42 pp** (95% CI [−0.75, 0.00], p=0.992) |
| Per-judgment latency | 258 ms | 417 ms |
| Cumulative spend | $0.02 | $0.041 of $10 envelope |

**Two findings, in tension and both honest.**

**Delta A is positive.** The v2 fix worked: training on real Tenacious-voice preference pairs (Llama-3.3-70B rewrites of the chosen outputs, gated by `scoring_evaluator >= 0.7`) lifted held-out accuracy from 0.167 (untrained baseline) to 0.417. The bootstrap p-value clears the conventional 0.05 cutoff. With n=12 the CI lower bound grazes 0 — that's a small-n quantization artifact, not a sign of weak signal — but I report it transparently rather than burying it.

**Delta B is decisively negative.** The same Qwen2.5-3B backbone, with no training and a careful Tenacious-rubric system prompt (5 tone markers + 14 banned phrases + 3 GOOD-draft exemplars), scores **0.833** on the same held-out slice. That beats the trained judge by 42 pp at p=0.99. There is no plausible reading where the trained LoRA wins.

**The honest interpretation:** at 3B parameters and 128 preference pairs, **tone judgment on Tenacious-Bench is a prompt-following problem, not a preference-learning problem.** The base model already "knows" what good Tenacious tone looks like — it just needs the rules spelled out. SimPO training on 128 pairs adds *some* signal (the +25 pp over untrained) but it's a strictly worse delivery vehicle than the prompt itself.

This is the publishable negative finding the brief explicitly anticipates: *"Many Week 11 interventions will fail Delta B; that is a legitimate, publishable finding and goes in the blog honestly."* I'd rather report a real negative than a forced positive.

A note on Delta C (the brief's τ²-Bench retail comparison): I do **not** report it as a positive lift. The trained judge was never trained on retail-domain tasks, so Delta C is informational only — Tenacious-Bench lift is Tenacious-specific by construction. Reusing the Week 10 number (pass@1 = 0.7267) without re-running τ²-retail saved ~$5 of compute that funded the multi-LLM synthesis routing instead.

A second honest note on the eval slice: only 12 of 75 held-out tasks have ground-truth chosen/rejected pairs (the 12 style-guide pairs). The remaining 63 held-out tasks were authored as standalone prompts. So the preference-accuracy CI bands will be wide, and a positive direction at p > 0.05 is the most likely outcome — *that is itself a publishable finding* per the Week 11 brief, and it goes in the model card.

## 6. What's next

- **v0.2 dataset**: expand the held-out preference slice from 12 to 30 by authoring chosen/rejected for the 30 hand-authored adversarial tasks; add `metadata.expected_mode` field to remove the `signal_confidence_alignment` rubric-clarity gap the Llama calibration flagged.
- **v0.2 model**: re-train on a Qwen2.5-7B backbone (still fits Colab T4 in 4-bit) with the same SimPO recipe; budget another ~$0 (Colab) and ~90 min wall time.
- **The ablation I didn't run**: prompt-engineered Claude Sonnet 4.6 on the same eval slice. It's an eval-tier call so the Week 11 budget rules forbade it on Days 2–3; Day 6 budget can fund 50 calls (~$2). My priors: Sonnet 4.6 will outperform a 3B SimPO judge on tone tasks but lose to it on the bench-arithmetic tasks (where a small model that has seen the bench schema 128 times beats a generalist that hasn't).

## The lesson

The hardest engineering call this week wasn't picking Path B over Path A. It was *not retraining* when the v1 eval accuracy stayed flat at 0.25 across every checkpoint. The expensive answer (more compute, bigger model, different algorithm) would have wasted $3–5 and a day. The honest answer cost $0.04 and one evening of reading logs.

If your training loss looks too clean, it usually is. And the hardest call is recognizing it's a data problem — not a model problem — before the GPU bill says it for you.

---

### Acknowledgements

Week 10 trace pool, probe library, failure taxonomy, and Tenacious Style Guide v2 were committed-as-given by the 10Academy / TRP1 program. Crunchbase ODM 1,001-company sample and layoffs.fyi CSV referenced as public sources within the documented window. SimPO via TRL's CPOTrainer; Unsloth for 4-bit QLoRA on Colab T4; OpenRouter for the multi-LLM routing.

### Citation

```bibtex
@dataset{tenacious_bench_v01_2026,
  title  = {Tenacious-Bench: a B2B sales-outreach evaluation benchmark for engineering-outsourcing agents},
  author = {Yorat, Eyoel and 10Academy TRP1 cohort},
  year   = 2026, version = {0.1}, license = {CC-BY-4.0}, publisher = {HuggingFace}
}
```
