# Datasheet for Tenacious-Bench v0.1

Following Gebru et al. (2021) "Datasheets for Datasets" (seven required sections) and Pushkarna et al. (FAccT 2022) "Data Cards" (telescopic / periscopic / microscopic detail layers).

---

## Telescopic (one-paragraph summary)

**Tenacious-Bench v0.1** is a 253-task evaluation benchmark for B2B sales-outreach agents, grounded in the Tenacious (B2B engineering-outsourcing) workflow. Tasks probe ten Tenacious-specific failure dimensions — ICP misclassification, signal over-claiming, bench over-commitment, tone drift, multi-thread leakage, dual-control coordination, scheduling edge cases, signal-confidence alignment, gap over-claiming, and cost pathology — that public benchmarks (τ²-Bench retail, MT-Bench, AlpacaEval) do not grade. Every task is mechanically scorable: a Python evaluator (`scoring_evaluator.py`) returns a numeric score in [0,1] per task with no human in the loop. The dataset is partitioned into train (50%), public dev (30%), and sealed held-out (20%); held-out passes three contamination checks (n-gram, embedding-similarity, time-shift). License: CC-BY-4.0.

## Periscopic (3–5 paragraph overview)

**Motivation.** The Tenacious team needs to know whether a B2B sales agent's outputs are correct *for their business, voice, and bench*. Public retail benchmarks do not capture the failure modes that matter to a B2B outsourcing firm — wrong-segment pitches, over-committing capacity that does not exist, fabricating peer-company practices, and pre-booking calendar slots when the prospect has only said "let me check." Tenacious-Bench v0.1 is the artifact that makes those failures legible.

**Composition.** 253 tasks across 10 dimensions. Source-mode distribution: 80 programmatic (parameter sweeps over the Week 10 probe library), 80 trace-derived (redacted-and-restructured Week 10 traces), 63 multi-LLM-synthesis (DeepSeek-V3.2 ↔ Qwen3-Next-80B routed pairs, with Llama-3.3-70B held-out as a third-family judge), 30 hand-authored adversarial. Each task has structured inputs (`hiring_signal_brief`, `bench_summary`, `prior_thread`), an `expected_action`, and a weighted rubric of 1–5 mechanical checks (regex banned/required phrases, structured grounding-reference, capacity-guard arithmetic, action-class detection, length bounds, abstain enforcement, and a tone-marker check that uses a heuristic fallback when no LLM judge is configured). All weights sum to 1.0 ± 1e-6 (validated at build time).

**Collection process.** No real Tenacious customer data was collected. Inputs were derived from (a) the program-supplied Week 10 trace pool (20 retail-domain traces redacted and re-shaped into Tenacious-shaped scenarios), (b) the publicly-available bench summary and style guide committed to the Week 10 repo, and (c) two public datasets named in the Week 10 brief: Crunchbase ODM 1,001-company sample and the layoffs.fyi CSV. Synthesis-mode tasks were authored online by Qwen3-Next-80B-A3B and DeepSeek-V3.2 via OpenRouter (134 calls, $0.0098 spend), with each task judged by a model from a *different* family per the preference-leakage rule of Li et al. 2025.

**Recommended uses.** (a) As a graded eval suite for any B2B sales-outreach agent that consumes the Tenacious hiring-signal-brief schema. (b) As the source corpus for SFT (Path A), preference-tuned-judge (Path B), or process-reward-model (Path C) training; each task carries a `ground_truth.chosen_output` and a `ground_truth.rejected_output` field where applicable. (c) As a teaching corpus for evaluation-design courses in agent-reliability work — the reproduction harness is a single `python -m generation_scripts.build_dataset` call with a deterministic seed.

**Distribution and maintenance.** Versioned (`v0.1`). Published on HuggingFace under CC-BY-4.0 once the program-staff sign-off step (Day 7) is complete. Held-out partition is *not* committed to the public repo until a leaderboard exists. Maintenance plan: one of the trainees (or the program) refreshes synthesis tasks each cohort, and a v0.2 release will fold in (i) an explicit `metadata.expected_mode` for confidence-alignment tasks (per the inter-rater finding), (ii) more cost-pathology coverage (currently only 2 tasks), (iii) sentence-transformers-based embedding-similarity check.

---

## Microscopic — Gebru et al. seven sections

### 1. Motivation

- **For what purpose was the dataset created?** To grade B2B sales-outreach agents on Tenacious-specific failure modes that no public benchmark grades. Authoring it is part of Week 11 of the 10Academy / TRP1 program.
- **Who created the dataset and on whose behalf?** Trainee (single author) on behalf of the 10Academy TRP1 cohort, not on behalf of any commercial Tenacious entity.
- **Who funded the creation of the dataset?** Trainee compute envelope ($10/week from program). Actual spend logged in `cost/openrouter_calls.jsonl`.

### 2. Composition

- **What do the instances represent?** Each instance is one (input, rubric, optional ground-truth) tuple representing a single B2B sales-outreach scenario. Inputs include a structured hiring-signal brief, an optional competitor-gap brief, a bench inventory snapshot, and a prior-thread sequence.
- **How many instances are there in total?** 253 tasks. Train: 126; dev: 62; held-out: 65.
- **Does the dataset contain all possible instances or is it a sample?** It is a constructed sample. Sampling rules: ten failure dimensions × stratified parameter sweep × four authoring modes. Per-dimension counts in `tenacious_bench_v0.1/composition.json`.
- **What data does each instance consist of?** See `schema.json` for the canonical schema definition. Required fields: `task_id`, `source_mode`, `dimension`, `difficulty`, `input`, `rubric`, `metadata`. Optional: `ground_truth`, `partition`.
- **Is there a label or target?** Yes — the rubric returns a numeric score in [0,1] per task. For Path B preference-pair construction, `ground_truth.chosen_output` and `ground_truth.rejected_output` serve as the chosen/rejected labels.
- **Is any information missing?** Some dimensions are under-sampled (cost_pathology has 2 tasks); flagged for v0.2.
- **Are relationships between instances made explicit?** Yes via `metadata.source_probe_id`, `metadata.source_trace_id`, and `metadata.synthesis_seed_id` provenance fields.
- **Are there recommended data splits?** Yes: 50/30/20 train/dev/held-out, stratified by dimension, with hand-authored adversarial tasks forced into held-out.
- **Are there errors, sources of noise, or redundancies?** Yes: (i) the heuristic tone-marker fallback in `scoring_evaluator.py` is a conservative regex check (a real LLM judge will be slotted in for sealed-slice scoring on Day 6); (ii) some prior-thread bodies repeat exact prospect-message text across non-overlapping partitions — this is the contamination-check target and the held-out partition is verified clean (see `contamination_check.json`).
- **Does the dataset rely on external resources?** Yes: Week 10 `policy.py` is imported by the `policy_compliant` rubric check. Pin: see `requirements.txt` once published.
- **Does the dataset contain confidential, offensive, or PII data?** No real prospect data, no PII. Synthetic prospect names ("Orrin Labs", "Halcyon Stack" etc.) are fictitious.

### 3. Collection process

- **How was the data acquired?** Three paths: (i) programmatic synthesis via parameter sweeps, (ii) redaction + restructuring of the program-supplied Week 10 trace pool, (iii) multi-LLM routing through OpenRouter (Qwen3-Next-80B-A3B and DeepSeek-V3.2 alternating as author and judge, with Llama-3.3-70B as the calibration judge).
- **What mechanisms or procedures were used?** All authoring code is in `generation_scripts/`. Build is reproducible from a fixed seed: `python -m generation_scripts.build_dataset --online-synthesis`.
- **Over what timeframe was the data collected?** 2026-04-29 to 2026-04-30. Public-source windows are recorded per task (`metadata.public_source_window`) and constrained to 2025-11-01..2026-04-29.
- **Were any ethical review processes conducted?** No external IRB. Per the program's data-handling policy, no real customer data is used; synthetic prospects only. The `LIVE_OUTBOUND` kill-switch from Week 10 carries forward — no task in this dataset can be used to dispatch a real email.

### 4. Preprocessing / cleaning / labeling

- **Was any preprocessing done?** Yes:
  - Judge-filter: every authored task is scored on three dimensions (input_coherence, ground_truth_verifiability, rubric_application_clarity); tasks below a threshold of 4 on any dimension are rejected (`tenacious_bench_v0.1/judge_filter_log.jsonl`). 1 of 254 authored tasks rejected by the judge in the online run.
  - Body-duplicate dedup across partitions: held-out tasks whose `prior_thread.body` exactly matches a train or dev task body are demoted to train.
  - 8-gram contamination dedup: held-out tasks sharing an 8-gram with any train or dev task are demoted to train.
  - Family-rotation enforcement: any task where `metadata.author_model` and `metadata.judge_model` belong to the same model family is rejected at build time (Li et al. 2025 preference-leakage rule).
- **Was the "raw" data saved?** Yes, the synthesis raw author/judge round-trip is logged in `cost/openrouter_calls.jsonl` (per-call usage and cost).
- **Is the preprocessing/cleaning/labeling software available?** Yes — all scripts in `generation_scripts/` and the scoring evaluator at `scoring_evaluator.py`.

### 5. Uses

- **Has the dataset been used for any tasks already?** Yes — internally, to validate the scoring_evaluator self-test against three dummy tasks. External uses follow Day 7.
- **Is there a repository that links to all uses?** This repo is the canonical source. Future uses will be linked from `README.md`.
- **What other tasks could the dataset be used for?** (a) Grading any B2B-outsourcing-shaped sales agent. (b) As a SFT, DPO/SimPO/ORPO, or PRM training corpus (per the Week 11 brief's three paths). (c) As a teaching dataset for evaluation-design and contamination-prevention coursework.
- **Are there tasks for which the dataset should not be used?** It is *not* a generic B2B benchmark — every rubric is grounded in Tenacious-specific policy, bench, and tone constraints. Using it to grade a non-Tenacious agent will under-rate the agent on Tenacious-specific dimensions and over-rate it on dimensions the agent is not designed for.

### 6. Distribution

- **Will the dataset be distributed to third parties?** Yes — published on HuggingFace under `tenacious_bench_v0.1` once the publication checklist passes.
- **How will the dataset be distributed?** HuggingFace datasets repository. The held-out partition will be removed from the public repo until a leaderboard exists.
- **License or terms of use?** **CC-BY-4.0.** Every task carries `metadata.license = "CC-BY-4.0"`.
- **Have any third parties imposed IP-based restrictions?** None. Public Crunchbase ODM and layoffs.fyi data references are downstream artifacts, not redistributed bulk data.

### 7. Maintenance

- **Who is supporting / hosting / maintaining?** Trainee (interim) → 10Academy program (long-term). Contact via the program's published channels.
- **Is there an erratum?** Two known issues:
  1. `cost_pathology` dimension is under-sampled (n=2). Will be expanded in v0.2.
  2. `signal_confidence_alignment` rubric implicitly derives the required phrasing-mode from `score × confidence`. v0.2 will surface this as an explicit `metadata.expected_mode` field, per the inter-rater finding.
- **Will the dataset be updated?** Yes. v0.2 ships once the cost-pathology and confidence-alignment refinements above are made; v0.1 remains available for historical reproducibility.
- **Will older versions continue to be supported?** Yes; semantic versioning on the HuggingFace tag.
- **If others want to extend the dataset, is there a mechanism?** Yes — pull requests against the GitHub repo. Generation scripts are deterministic from a seed; reviewers can re-run the build to verify any contributed task batch.

---

## Composition tables

### Per-partition composition

| Partition | Total | programmatic | trace_derived | multi_llm_synthesis | hand_authored_adversarial |
|---|---:|---:|---:|---:|---:|
| train | 126 | 43 | 47 | 36 | 0 |
| dev | 62 | 28 | 23 | 11 | 0 |
| held_out | 65 | 9 | 10 | 16 | 30 |

### Per-dimension counts (overall)

| Dimension | n |
|---|---:|
| bench_over_commitment | 30 |
| icp_misclassification | 26 |
| signal_over_claiming | 25 |
| signal_confidence_alignment | 29 |
| scheduling_edge_cases | 24 |
| multi_thread_leakage | 32 |
| dual_control_coordination | 28 |
| tone_drift | 31 |
| gap_over_claiming | 26 |
| cost_pathology | 2 |

### Per-difficulty distribution

| Difficulty | n |
|---|---:|
| easy | 34 |
| medium | 35 |
| hard | 115 |
| adversarial | 69 |

(Held-out is 35/65 = 54% adversarial — by construction.)

---

## Provenance & contamination summary

- **Contamination check:** all three checks (n-gram ≥8, hashed-trigram cosine ≥0.85, time-shift) PASS on the current build (`tenacious_bench_v0.1/contamination_check.json`).
- **Inter-rater agreement:** 100% within-±1 across all three meta-dimensions on a 30-task hand-labeled stratified sample (`inter_rater_agreement.md`). Cross-rater check with Llama-3.3-70B on 49 templated tasks: 73.5% within-±1 (below the 80% threshold; rubric-clarity revision triggered for `signal_confidence_alignment` in v0.2).
- **Preference-leakage check:** PASS — no task is authored and judged by the same model family.

## Citation

```bibtex
@dataset{tenacious_bench_v01_2026,
  title = {Tenacious-Bench: a B2B sales-outreach evaluation benchmark for engineering-outsourcing agents},
  author = {Nebiyu, Eyoel and 10Academy TRP1 cohort},
  year = {2026},
  version = {0.1},
  license = {CC-BY-4.0},
  publisher = {HuggingFace},
}
```
