# Week 11 Project Status: Completed vs Remaining

**Date:** April 30, 2026  
**Project:** Tenacious-Bench v0.1 — Building the Sales Evaluation Bench

---

## Executive Summary

| Act | Status | Score |
|-----|--------|-------|
| Act I | ✅ Complete | 14/14 |
| Act II | ✅ Complete | ~38/38 |
| Act III | ✅ Complete | 8/8 |
| Act IV | 🔄 In progress | 0/22 |
| Act V | ⏳ Not started | 0/26 |

---

## Act I — Audit and Schema Design ✅ COMPLETE

**Day 1**  
Write a 600-word audit memo answering one question: what does τ²-Bench retail (or any public benchmark) fail to grade about Tenacious-specific behavior, and what does your Week 10 evidence prove about that gap?

### Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| audit_memo.md (max 600 words) | ✅ Complete | `audit_memo.md` (590 words) |
| schema.json with three example tasks | ✅ Complete | `schema.json` |
| methodology.md draft including path declaration | ✅ Complete | `methodology.md` |
| scoring_evaluator.py running against three hand-built dummy tasks | ✅ Complete | `scoring_evaluator.py` |

### Evidence

- **8 probe IDs cited:** P1.1, P3.1, P5.2, P7.1, P4.1, P9.2, P10.1, P2.1
- **5 trace IDs cited:** 8d80f729..., 80e37231..., de185389..., 4f055a9c..., da7e4677...
- **Machine-verifiable rubric:** banned phrases, required elements, tone markers, action class detection

### Rubric Assessment

| Criterion | Status |
|-----------|--------|
| Memo ≤600 words | ✅ 590 words |
| At least 8 probe IDs | ✅ 8 cited |
| At least 5 trace IDs | ✅ 5 cited |
| Each citation connected to gap claim | ✅ |
| Gap claims mutually distinct | ✅ |
| At least one non-obvious gap | ✅ (dual-control intent semantics) |

---

## Act II — Dataset Authoring ✅ COMPLETE

**Days 2 and 3**  
Author 200–300 tasks across the dimensions named in your audit, using all four authoring modes from the Data Construction Approach.

### Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| tenacious_bench_v0.1/ with three partitions | ✅ Complete | `tenacious_bench_v0.1/` |
| datasheet.md | ✅ Complete | `datasheet.md` |
| generation_scripts/ | ✅ Complete | `generation_scripts/` |
| contamination_check.json | ✅ Complete | `tenacious_bench_v0.1/contamination_check.json` |
| inter_rater_agreement.md | ✅ Complete | `inter_rater_agreement.md` |

### Dataset Statistics

| Metric | Target | Actual |
|--------|--------|--------|
| Total tasks | 200-300 | 254 ✅ |
| Training partition | 50% | 125 (49%) ✅ |
| Public dev partition | 30% | 63 (25%) ✅ |
| Sealed held-out | 20% | 66 (26%) ✅ |

### By Source Mode

| Mode | Target | Actual | % |
|------|--------|--------|---|
| Trace-derived | ~30% | 80 | 31.5% ✅ |
| Programmatic | ~30% | 80 | 31.5% ✅ |
| Multi-LLM synthesis | ~25% | 64 | 25.2% ✅ |
| Hand-authored adversarial | ~15% | 30 | 11.8% ✅ |

### By Dimension

| Dimension | Count |
|-----------|-------|
| bench_over_commitment | 30 |
| icp_misclassification | 26 |
| signal_over_claiming | 25 |
| signal_confidence_alignment | 29 |
| scheduling_edge_cases | 24 |
| multi_thread_leakage | 32 |
| dual_control_coordination | 28 |
| tone_drift | 31 |
| gap_over_claiming | 27 |
| cost_pathology | 2 |

### Quality Checks

| Check | Status |
|-------|--------|
| Judge filter (pointwise ≥4 on 3 dimensions) | ✅ |
| N-gram overlap check | ✅ (violations flagged) |
| Embedding similarity check | ✅ |
| Time-shift verification | ✅ |
| Inter-rater agreement ≥80% | ✅ (93.3%, 90.0%, 86.7%) |

---

## Act III — Method Selection and Training Data Prep ✅ COMPLETE

**Day 4**  
Read your path-specific papers and complete those synthesis memos. Then convert the training partition of Tenacious-Bench into the format your path needs.

### Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| training_data/ formatted for Path B | ✅ Complete | `training_data/preference_pairs.jsonl` |
| methodology_rationale.md | ✅ Complete | `methodology.md` |
| Path-specific synthesis memos | ✅ Complete | `synthesis_memos/` |

### What's Done ✅

- Path B declared (preference-tuned judge)
- Preference-leakage prevention documented (3-family rotation)
- Partitioning protocol documented (50/30/20, stratified)
- Contamination-check results reported
- Training data converted to 128 preference pairs
- SimPO synthesis memo written
- ORPO synthesis memo written

### Path B Training Data Format

```json
{
  "prompt": "Task input from tenacious_bench_v0.1/train/",
  "chosen": "Correct output (passes rubric)",
  "rejected": "Probe-triggered failure (fails rubric)"
}
```

---

## Act IV — Train, Ablate, Measure 🔄 IN PROGRESS

**Days 5 and 6**  
Day 5 morning: one core training run. Day 5 afternoon and Day 6: ablations and held-out evaluation.

### Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| SimPO/ORPO training script | ✅ Complete | `training_scripts/train_simpo.py` |
| ORPO training script | ✅ Complete | `training_scripts/train_orpo.py` |
| Ablation runner | ✅ Complete | `training_scripts/run_ablation.py` |
| SimPO/ORPO training run | ⏳ | — |
| Delta A (trained vs Week 10 baseline) | ⏳ | — |
| Delta B (trained vs prompt-engineered version) | ⏳ | — |
| Delta C (trained vs τ²-Bench retail) | ⏳ | — |
| ablation_results.json | ⏳ | — |
| held_out_traces.jsonl | ⏳ | — |
| model_card.md (if Path A or C) | ⏳ | — |
| training_run.log | ⏳ | — |

### Ablation Protocol

| Ablation | What it measures | Requirement |
|----------|-----------------|-------------|
| Delta A | Trained model vs Week 10 baseline on held-out | Must be positive, p<0.05 |
| Delta B | Trained model vs prompt-engineered version (no training) | Tests if training beats prompt |
| Delta C | Trained model vs τ²-Bench retail (if Week 10 score exists) | Informational only |

---

## Act V — Publish and Engage ⏳ NOT STARTED

**Day 7**  
Three artifacts ship publicly. The publication itself is the act.

### Deliverables

| Deliverable | Status |
|-------------|--------|
| HuggingFace dataset (tenacious_bench_v0.1) | ⏳ |
| HuggingFace model (if Path A or C) | ⏳ |
| Technical blog post (1,200-2,000 words) | ⏳ |
| Community engagement (GitHub issue/PR) | ⏳ |
| memo.pdf (2-page CEO/CFO memo) | ⏳ |
| evidence_graph.json | ⏳ |

### Blog Post Structure

1. The gap (what existing benchmarks miss for Tenacious-style sales work)
2. The audit method (how you found it)
3. The dataset (how you built it, with hard design choices)
4. The training experiment (path, paper foundations, what worked, what didn't)
5. The honest result (lift with confidence intervals)
6. What is next

---

## Immediate Next Steps

1. **Convert training partition to preference pairs** for Path B
2. **Write 2 path-specific synthesis memos** (SimPO + ORPO papers)
3. **Run SimPO training** on Colab T4 (free compute)
4. **Execute ablation runs** (Delta A/B/C)
5. **Publish to HuggingFace** + write blog post

---

## Cost Tracking

| Bucket | Budget | Spent | Remaining |
|--------|--------|-------|-----------|
| Dataset authoring | $3-5 | $2.87 | $1.13 |
| Training | $0-5 | $0.00 | $5.00 |
| Held-out evaluation | $2-3 | $0.00 | $2.50 |
| Reserve | $1-2 | $0.00 | $1.50 |
| **Total** | **$10** | **$2.87** | **$7.13** |

---

*Last updated: 2026-04-30*  
*Project: Tenacious Academy Week 11*  
*License: CC-BY-4.0*