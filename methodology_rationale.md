# Methodology Rationale — Path B Preference-Pair Construction

**Author:** Eyoel Yorat (10Academy TRP1) · **Date:** 2026-05-01 · **Length:** one page

## Why these (chosen, rejected) pairs are the right training data for Path B on Tenacious

### 1. The failure modes the data must teach

The Week 10 trace pool concentrates the failures Path B has to learn to *judge against*. Three traces ground the design:

- **`8d80f729-90bb-45f2-8d2e-73d4959648c5`** (retail_ho_0::5, single-turn `passed=false`). Seeds **TB-0205** (dual-control) — a paired (chosen, rejected) where the *rejected* output books a calendar slot after the prospect said "let me check," and the *chosen* output offers two held slots and asks. P7.1 triggered at 1.00 in Week 10; this is the single highest-leverage judgment the Path B critic must learn.
- **`80e37231-2381-4c57-a8f7-898c675e809b`** (retail_ho_1::9). Seeds the multi-thread-leakage trace-derived family — *rejected* re-engages after explicit opt-out, *chosen* abstains with empty body. P5.2 also triggered at 1.00.
- **`de185389-9e01-4a36-a34d-da9b188c8f3c`** (retail_ho_2::12). Seeds the bench-over-commitment programmatic sweep — *rejected* commits a count larger than `bench_summary.json` shows, *chosen* phrases capacity honestly with a phased ramp.

These three failure shapes — *acts when it should wait*, *engages when it should abstain*, *commits more than evidence shows* — are inconsistency-of-judgment failures, not generation-quality failures. A judge is the right intervention; an SFT generator that already writes fluent text would not lift on these traces.

### 2. Construction protocol

128 preference pairs derived from the 128-task training partition (`tenacious_bench_v0.1/train/tasks.jsonl`):

- **`prompt`** = scenario + serialized hiring_signal_brief + bench snapshot + last 3 prior-thread turns. Same shape the agent sees at inference.
- **`rejected`** = a probe-triggered failure pattern: e.g., "Yes, we can deploy 10 Python engineers next week" for a P3.1-derived bench-over-commit task. Rejected outputs are constructed to cleanly fail the task's mechanical rubric so the judge has an unambiguous signal.
- **`chosen`** = an output that passes the same rubric (`scoring_evaluator.score_task` ≥ 0.7). Currently template-derived per the deterministic build; final adds dev-tier rewrites (Llama-3.3-70B) gated by `scoring_evaluator` before emission.
- **No same-family chosen/judge collision.** Per the rotation policy in `methodology.md §2`, chosen-rewrites are authored by Llama-3.3-70B (third family); the trained judge will be a Qwen3 backbone. Three families across (synthesis-author / chosen-rewrite / judge) eliminates the same-family collision Li et al. (2025) flag.

The 128 pairs span all 9 active dimensions in the train partition (cost_pathology has no train tasks; its 2 examples are held_out adversarials).

### 3. Why preferences over SFT

**SimPO (Meng, Xia, Chen, NeurIPS 2024)** is the chosen training algorithm. The decisive property for Path B on Tenacious data is reference-free, length-normalized scoring: dual-control (TB-0205) and multi-thread-leakage tasks have *very short* correct outputs (an empty body for opt-out; a one-line ask for a hedge), and *long* incorrect outputs (a multi-paragraph booking confirmation, a full re-engagement draft). DPO without length normalization would penalize the short chosen outputs by their margin term; SimPO does not. The reference-free property also halves Colab T4 VRAM, which is the binding constraint for the Day-5 run.

**Preference Leakage (Li et al., 2025)** is the constraint that drove the three-family rotation above. Li et al.'s core finding is that same-family chosen-author / judge pairs converge to a within-family preference manifold rather than the human-intended one. Tenacious's failure modes (signal grounding, bench arithmetic, opt-out semantics) are *not* preferences any one model family encodes natively — making cross-family rotation strictly necessary for the judge to generalize.

### 4. Contamination check, post-construction

The contamination check runs against `dev/` and `held_out/` after the train-partition preference pairs are constructed: **0 violations across n-gram (≥8 contiguous tokens), embedding similarity (hashed-trigram cosine ≥0.85), and time-shift (public-source window 2025-11-01..2026-04-29)**. The build-time partitioner `build_dataset.partition()` mirrors the post-hoc check, demoting any held-out task whose `prior_thread.body` matches or 8-gram-overlaps a train/dev body. Latest build demoted 13 body-duplicates and 2 8-gram overlaps to train; held-out is contamination-clean by construction.

### 5. What this rationale binds the Day-5 run to

If SimPO on Qwen3-2B does not lift Delta A (paired-bootstrap p<0.05) on the held-out 75-task slice, the path-specific failure attributable to the *data construction* (rather than the algorithm) would be: chosen outputs that are template-derived rather than rewrite-derived. The mitigation queued for Day-4 evening is the Llama-3.3-70B chosen-rewrite pass over the 128 train pairs, gated by `scoring_evaluator ≥ 0.7`, with the family-rotation invariant enforced before emission.
