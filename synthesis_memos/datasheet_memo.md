# Synthesis Memo 2 — Datasheets for Datasets (Gebru et al., 2021) and Data Cards (Pushkarna et al., FAccT 2022)

**Memo author:** Eyoel Yorat (10Academy TRP1)  ·  **Date:** 2026-04-30  ·  **Length:** ≈1 page

## What both papers ask for

- **Gebru et al.** define a seven-section datasheet (motivation, composition, collection, preprocessing, uses, distribution, maintenance) intended to give a downstream user enough to decide whether the dataset is safe to use for their task.
- **Pushkarna et al.** extend Gebru with a *layered-detail* doctrine: a *telescopic* one-paragraph summary, a *periscopic* 3–5 paragraph overview, and a *microscopic* per-row-or-per-section detail layer. The layering means an under-prepared reader can stop at the telescopic level and still be safer than they would be otherwise; a careful reviewer can drill into the microscopic layer.

## Where I agree (and apply directly)

- **Seven Gebru sections are non-negotiable as the minimum schema.** My `tenacious_bench_v0.1/datasheet.md` covers all seven; absence is graded.
- **Pushkarna's layered detail is the right form for evaluation benchmarks specifically.** Benchmark consumers split into "users who want to grade their agent" (telescopic + periscopic) and "users who want to extend the benchmark" (microscopic). Single-layer datasheets make the second group re-read the first group's content.
- **License declaration *per row* is non-negotiable.** Every Tenacious-Bench task carries `metadata.license = "CC-BY-4.0"`. A whole-dataset license with row-level exceptions is a known contamination vector at fork time.

## Where I disagree (the "specific design choice" disagreement)

**Gebru's implicit single-author voice.** The seven-section schema reads as if one team controls the dataset and one team takes the questions. For Tenacious-Bench v0.1 the *author* is **trainee + program staff sign-off**. The accountability model for this benchmark differs from a corporate dataset: my datasheet records both the trainee and the program-staff role (the staff member who signs off the publication checklist before the artifact goes public).

**Why this matters.** Gebru's section 7 (maintenance) asks "who is supporting / hosting / maintaining?" with a default assumption that the answer is one organization. For a cohort-authored benchmark, the honest answer is two-tier: the trainee maintains the v0.1 build through the cohort window; the program owns long-term hosting and triage. The datasheet records both, and the README points reviewers to the program's published triage channel rather than to me. This is a small textual change but a meaningful accountability difference, and I think Gebru's template under-specifies it for educational-cohort artifacts.

**Pushkarna's microscopic detail at the row level.** I disagree with Pushkarna's default that microscopic detail lives in a separate document. For an LLM-eval benchmark, *every row* should carry its own provenance triple (`author_model`, `judge_model`, `source_probe_id` / `source_trace_id` / `synthesis_seed_id`) inside the JSONL itself. A reviewer auditing one task should not have to cross-reference an external file. Tenacious-Bench v0.1 puts the microscopic-detail layer **inside `metadata` on every task**; the datasheet's microscopic section then summarizes the schema rather than duplicating per-row content.

## What this changed in my datasheet

- Telescopic summary in the *first paragraph* of `datasheet.md`, not buried after the section list.
- Per-row provenance fields (`author_model`, `judge_model`, `source_probe_id`, `source_trace_id`, `synthesis_seed_id`, `public_source_window`) are required by `schema.json` and validated at build time. A reviewer sampling 5 random tasks gets 5 complete provenance audits without reading the datasheet.
- Maintenance section names *both* the trainee and the program responsibility, with the contact channel pointing to the program rather than to a personal email.
- Erratum (section 7) was written *before* publication, not after — it lists two known limitations (cost_pathology under-sample, expected-mode field for confidence-alignment) that v0.2 addresses. Disclosing limitations pre-publication is a Gebru-spirit move; the original template is silent on whether errata can lead.

## What I'm still not sure about

Gebru et al.'s "have any third parties imposed IP-based restrictions" question is awkward for a benchmark whose inputs are derived from the program's Week 10 trace pool (the program owns those traces). I answer "none" because the Week 10 traces are committed-as-given for trainee use, but a stricter reading would require a release statement from the program. I'd rather over-disclose this in v0.2 than ride the awkwardness, and I'll add a "program-supplied source attestation" subsection if program staff confirms it's needed for HuggingFace publication.
