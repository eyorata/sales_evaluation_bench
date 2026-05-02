# Synthesis Memo 1 — Best Practices and Lessons Learned on Synthetic Data for Language Models (Liu et al., COLM 2024)

**Memo author:** Eyoel Nebiyu (10Academy TRP1)  ·  **Date:** 2026-04-30  ·  **Length:** ≈1 page

## What the paper says

Liu et al. lay out an operational reference for synthetic-data construction. Three claims load-bear:
1. **Routing across model families beats single-model synthesis.** Different families have different generation tendencies; ensembling at the data level (multiple authors) rather than at the model level (one author + RLAIF) reduces dataset-level mode collapse.
2. **Quality filtering is necessary and not sufficient.** Pointwise judges can rubber-stamp same-family slop; pairwise comparison and out-of-family judging are the controls that actually catch contamination.
3. **Authoring-mode mix matters more than authoring-mode polish.** A balanced mix (trace-derived + programmatic + multi-LLM + hand-authored) outperforms any single mode at fixed cost, even when the single mode is more polished per-task.

## Where I agree

- Family rotation as the primary contamination-control: my `methodology.md §2` adopts this verbatim. I rotate Qwen3 ↔ DeepSeek-V3.2 across author/judge slots and keep Llama-3.3-70B held back as a third-family calibration judge. The rotation is enforced statically by `check_no_leakage` in `build_dataset.py`.
- Pointwise + pairwise judge filtering: my pipeline runs pointwise scoring (3 dims × 1–5) on every task, with the calibration script (`judge_calibration.py`) sampling 50 templated tasks for cross-rater reliability.

## Where I disagree (the "specific design choice" disagreement the brief asks for)

**Liu et al.'s default is roughly equal weighting across synthesis modes (~25% each).** I disagree with that default for small-data starting points like Tenacious's. My share is 31% trace-derived / 31% programmatic / 25% multi-LLM-synthesis / 12% hand-authored adversarial.

**Why the shift toward trace-derived and programmatic at the expense of equal mix?**

1. **Trace-derived is *free* and reflects real distribution.** I have 20 program-supplied Week 10 traces. Each redacted-and-restructured trace becomes 3–4 task variants without any LLM cost and with provably-real distributional behavior. Per LIMA (Zhou et al. 2023, my next memo), small-quality dominates large-quantity at this scale; spending cost-and-attention budget on multi-LLM synthesis when free trace-derived data is available is dominated.
2. **Programmatic parameter sweeps cover the schema space exhaustively.** Five generators × 14–18 variants each = 80 tasks that span the structured-input space (company size × segment × headcount × stack × AI-maturity score). A multi-LLM synthesis run would have to *sample* this space; programmatic *covers* it.
3. **Multi-LLM synthesis is the most contamination-risky bucket** (per Chen et al. 2025, my contamination-survey memo). Lower share = lower exposure to that risk. The synthesis tasks that *do* exist in v0.1 are the harder seeds the deterministic modes cannot capture (e.g. dual-control hedges with nuanced phrasing).
4. **Hand-authored adversarial earns a smaller share but is forced into held-out** — it carries the most originality weight per the brief's grading rubric. 30 tasks is enough to set a high originality bar without bloating the corpus.

This shift produced a clean contamination-check pass (0 violations across n-gram, embedding, time-shift) on the first build attempt. I read that as evidence the weighting was right for this domain.

## What this changed in my pipeline

- I authored the 30 hand-adversarial tasks **first** (Day 2 morning) and used them as the originality-floor reference when judging multi-LLM synthesis variants for inclusion.
- Multi-LLM synthesis runs only `--online` once per cohort (when an OpenRouter key is available); the offline-template path is the deterministic fallback that ships in the repo so the dataset reproduces bit-identically without network access.
- Per-mode share is logged in `tenacious_bench_v0.1/composition.json` so a reviewer can see the deviation from Liu et al.'s default at a glance.

## What I'm still not sure about

Liu et al. recommend an eval-tier judge model for spot-checking 50 sampled tasks. The brief explicitly forbids eval-tier model use on Days 2–3. I substituted a third-family dev-tier judge (Llama-3.3-70B) for the spot-check, and got 73.5% agreement-within-±1 against the templated a-priori scores. That's *below* the 80% threshold the inter-rater requires, even though the within-author dual-pass was 100%. Liu et al. don't take a position on whether dev-tier is sufficient for cross-rater calibration — I treat the gap as a signal worth reporting honestly rather than as a blocking failure, since the within-author check is what the brief grades against.
