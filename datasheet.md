# Tenacious-Bench v0.1 Datasheet

> Based on Datasheets for Datasets (Gebru et al., 2021) and Data Cards (Pushkarna et al., 2022)

---

## 1. Motivation

### Why was this dataset created?

Tenacious-Bench v0.1 was created to address a critical gap in B2B sales evaluation: existing benchmarks (including τ²-Bench retail) grade whether an agent completes a retail customer-service script against a closed tool schema, but they fail to evaluate the failure modes that destroy a Tenacious deal — specifically:

1. **Evidence-discipline failures**: signal over-claiming, bench over-commitment, wrong-segment pitch
2. **Dual-control coordination failures**: booking without explicit user commitment

The dataset is designed to evaluate Tenacious-style B2B sales agents that must ground outreach in public signals, qualify leads against ICP segments, and book discovery calls — all against private signal briefs and bench inventory that no existing benchmark supplies or scores.

### Who funded the creation?

This dataset was created as part of the Tenacious Academy Week 11 project, funded by the Tenacious executive team. No external funding was received.

### What are the intended uses?

- **Primary:** Evaluate Tenacious Conversion Engine agents on Tenacious-specific failure modes
- **Secondary:** Train preference-tuned judges (Path B) or generation components (Path A) that lift agent performance on identified failure modes
- **Tertiary:** Serve as a template for other B2B sales teams building domain-specific evaluation benchmarks

### What are not the intended uses?

- This dataset is NOT intended for evaluating general-purpose conversational agents on retail tasks
- This dataset is NOT intended for training models that will be deployed without human oversight in sales contexts
- This dataset is NOT validated for use in regulated industries beyond B2B software sales

---

## 2. Composition

### What does the dataset contain?

The dataset contains 254 evaluation tasks across 10 failure dimensions:

| Dimension | Count | Description |
|-----------|-------|-------------|
| bench_over_commitment | 30 | Agent promises staffing capacity it may not have |
| icp_misclassification | 26 | Wrong segment classification leads to wrong pitch |
| signal_over_claiming | 25 | Agent overstates weak or missing public signals |
| signal_confidence_alignment | 29 | Language too assertive relative to evidence confidence |
| scheduling_edge_cases | 24 | Time-zone, holiday, culturally sensitive scheduling |
| multi_thread_leakage | 32 | State bleeds across conversation threads |
| dual_control_coordination | 28 | Agent books when it should wait for commitment |
| tone_drift | 31 | Style deviates from Tenacious voice |
| gap_over_claiming | 27 | Competitor-gap language outruns evidence |
| cost_pathology | 2 | Prompt/response size exceeds token/latency bounds |

### How was data labeled?

Each task carries metadata including:
- `author_model`: The model that generated the task (or "human" for hand-authored)
- `judge_model`: The model that scored the task for quality filtering
- `judge_scores`: Three dimensions (input_coherence, ground_truth_verifiability, rubric_application_clarity) each rated 1-5
- `source_probe_id`: Reference to the Week 10 probe that inspired the task

### What is the data format?

Each task is a JSONL record with:
- `task_id`: Unique identifier (TB-XXXX format)
- `input`: Scenario, hiring_signal_brief, bench_summary, prior_thread
- `rubric`: Weighted checks with type, patterns, and thresholds
- `metadata`: Provenance, quality scores, license

### Are there relationships between data points?

Yes. Tasks are linked to:
- **Week 10 probes** via `source_probe_id` (e.g., P1.1 → TB-0026)
- **Synthesis seeds** via `synthesis_seed_id` (e.g., S-1 → TB-0401)
- **Author/judge pairs** via metadata (rotation policy prevents same-family generation+judging)

---

## 3. Collection Process

### What processes influenced data collection?

1. **Audit memo** (Day 1) identified the gap between τ²-Bench retail and Tenacious-specific behavior
2. **Failure taxonomy** from Week 10 informed dimension prioritization
3. **Multi-LLM synthesis pipeline** routed across model families with judge filtering
4. **Preference leakage prevention** (Li et al., 2025) enforced model-family rotation

### Who was involved in data collection?

- **Primary annotator:** Program staff (single annotator for consistency)
- **Generation:** Multiple LLM families (DeepSeek-V3.2, Qwen3-Next-80B-A3B) via OpenRouter
- **Judging:** Dev-tier models (Qwen3-Next-80B-A3B, DeepSeek-V3.2) for iteration; eval-tier (Claude Sonnet 4.6) for sealed held-out

### Over what time period was data collected?

- **Start:** 2026-04-29
- **End:** 2026-04-29 (single-day authoring with generation scripts)
- **Total wall time:** ~4 hours for full dataset generation

### What are the data collection modalities?

| Mode | Count | % |
|------|-------|---|
| Trace-derived (Week 10 traces) | 80 | 31.5% |
| Programmatic (parameter sweeps) | 80 | 31.5% |
| Multi-LLM synthesis | 64 | 25.2% |
| Hand-authored adversarial | 30 | 11.8% |

---

## 4. Preprocessing

### Was any preprocessing applied?

Yes. The following preprocessing was applied:

1. **Redaction:** All trace-derived tasks redact company names, contact names, and specific deal values
2. **Signal windowing:** Public signals (Crunchbase, layoffs.fyi) are tagged with `public_source_window` to enable time-shift verification
3. **Banned phrase extraction:** Failure patterns from probes are converted to regex for deterministic scoring
4. **Partition assignment:** Tasks are stratified by dimension and difficulty, with adversarial tasks forced into held_out

### Was there quality filtering?

Yes. Every generated task passed a judge filter:
- Pointwise scoring: input coherence, ground-truth verifiability, rubric-application clarity (1-5 each)
- Threshold: ≥4 on each dimension for inclusion
- Tasks below threshold were regenerated or discarded

### What post-processing was applied?

- N-gram overlap check (≥8-gram overlap rejected)
- Embedding similarity check (cosine ≥0.85 rejected)
- Time-shift verification for public-signal tasks

---

## 5. Uses

### What is the dataset used for?

1. **Evaluation:** Score Tenacious Conversion Engine agents on Tenacious-specific failure modes
2. **Training:** Construct preference pairs for Path B (judge training) or input/output pairs for Path A (SFT)
3. **Benchmarking:** Compare against τ²-Bench retail to demonstrate Tenacious-specific gap

### Has the dataset been used for any publications?

Not yet. The dataset is being prepared for:
- HuggingFace dataset publication
- Technical blog post
- Community engagement (τ²-Bench GitHub issue)

### Are there any tasks for which the dataset should NOT be used?

- Do not use for evaluating agents without access to signal briefs and bench inventory
- Do not use for training models that will operate without human oversight in sales contexts
- Do not use for benchmarking general-purpose conversational agents

---

## 6. Distribution

### How will the dataset be distributed?

- **Primary:** HuggingFace Hub (dataset repository)
- **License:** CC-BY-4.0
- **URL format:** `https://huggingface.co/datasets/{username}/tenacious_bench_v0.1`

### Will the dataset be distributed with supporting artifacts?

Yes:
- `datasheet.md` (this document)
- `scoring_evaluator.py` (machine-verifiable scorer)
- `schema.json` (task schema with examples)
- `composition.json` (dataset statistics)
- `contamination_check.json` (contamination verification)

### Are there any restrictions on the dataset?

- **License:** CC-BY-4.0 (attribution required)
- **Partitioning:** Held_out partition is sealed until leaderboard publication
- **Commercial use:** Allowed with attribution

---

## 7. Maintenance

### Who will maintain the dataset?

- **Primary:** Dataset author (Tenacious Academy trainee)
- **Secondary:** Program staff for critical updates

### How will the dataset be updated?

- **v0.2 (planned):** Additional adversarial tasks based on training run findings
- **v1.0 (planned):** Full public release with leaderboard

### Will external contributions be accepted?

Yes, via standard HuggingFace PR workflow. Contributors must:
1. Pass the contamination check script
2. Maintain ≥80% inter-rater agreement on new tasks
3. Document author/judge model rotation

---

## Pushkarna Layered Detail (Data Cards)

### Telescopic View (High-Level Summary)

- **Purpose:** Evaluate Tenacious-specific B2B sales agent failure modes
- **Scope:** 254 tasks across 10 dimensions, 3 partitions (train/dev/held_out)
- **Key insight:** τ²-Bench retail fails to capture evidence-discipline and dual-control failures

### Periscopic View (Structural Summary)

| Partition | Tasks | % | Intended Use |
|-----------|-------|---|--------------|
| train | 125 | 49% | Training data for Path A/B/C |
| dev | 63 | 25% | Public iteration and development |
| held_out | 66 | 26% | Sealed evaluation (Delta A/B/C) |

### Microscopic View (Per-Task Detail)

Each task includes:
- Full input context (scenario, signal brief, bench summary, prior thread)
- Complete rubric with weighted checks
- Metadata: author_model, judge_model, judge_scores, source_probe_id, creation timestamp
- License: CC-BY-4.0

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.1 | 2026-04-29 | Initial release with 254 tasks |

---

## Citation

```bibtex
@misc{tenacious_bench_v0.1,
  title={Tenacious-Bench v0.1: A Domain-Specific Evaluation Benchmark for B2B Sales Agents},
  author={Tenacious Academy},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/.../tenacious_bench_v0.1}},
  license={CC-BY-4.0}
}
```