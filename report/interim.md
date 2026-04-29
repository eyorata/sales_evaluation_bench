# Tenacious-Bench v0.1 — Interim Report

**Date:** April 29, 2026  
**Project:** Week 11 — Building the Sales Evaluation Bench  
**Author:** Tenacious Academy

---

## Executive Summary

This interim report documents the progress on Week 11 of the Tenacious Academy program: building a domain-specific evaluation benchmark for Tenacious-style B2B sales work. The project has completed Act I (Audit and Schema Design) and Act II (Dataset Authoring), with Act III (Training Data Preparation) and Act IV (Training/Ablation) remaining.

### Key Deliverables Completed

| Deliverable | Status | Location |
|-------------|--------|----------|
| Audit memo (600 words) | ✅ Complete | `audit_memo.md` |
| Tenacious-Bench schema | ✅ Complete | `schema.json` |
| Path declaration (B) | ✅ Complete | `methodology.md` |
| Scoring evaluator | ✅ Complete | `scoring_evaluator.py` |
| 254 tasks (4 modes) | ✅ Complete | `tenacious_bench_v0.1/` |
| Contamination checks | ✅ Complete | `contamination_check.json` |
| Inter-rater agreement | ✅ Complete | `inter_rater_agreement.md` |
| Datasheet (Gebru + Pushkarna) | ✅ Complete | `datasheet.md` |
| 2 synthesis memos | ✅ Complete | `synthesis_memos/` |
| Cost log | ✅ Complete | `cost/log.md` |
| README | ✅ Complete | `README.md` |

---

## 1. The Gap: Why τ²-Bench Retail Fails for Tenacious

### Core Insight

τ²-Bench retail grades whether an agent completes a retail customer-service script against a closed tool schema. However, the failure modes that destroy a Tenacious deal are:

1. **Evidence-discipline failures**: signal over-claiming, bench over-commitment, wrong-segment pitch
2. **Dual-control coordination failures**: booking without explicit user commitment

These require grounded comparison against a private hiring signal brief and bench inventory — none of which τ²-Bench supplies or scores.

### Evidence from Week 10 Traces

From `eval/trace_log.jsonl`, every τ²-retail held-out task records `passed=false` in single-turn execution. That number is *not* informative about Tenacious: the same agent that fails retail tool use can still write an over-claiming Tenacious cold email and lose a $200K outsourcing engagement.

### Probe Evidence (8+ probes cited)

| Probe ID | Failure Mode | Trigger Rate |
|----------|-------------|--------------|
| P1.1 | post_layoff_fresh_funding_should_be_segment_2 | 0.60 |
| P3.1 | prospect_asks_10_python_engineers_bench_has_7 | N/A |
| P7.1 | book_without_user_confirmed_slot | 1.00 |
| P5.2 | same_thread_recall_after_optout | 1.00 |

---

## 2. Dataset Composition

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total tasks | 254 |
| Dimensions | 10 |
| Training partition | 125 (49%) |
| Public dev partition | 63 (25%) |
| Sealed held-out | 66 (26%) |

### By Dimension

| Dimension | Count | % |
|-----------|-------|---|
| bench_over_commitment | 30 | 11.8% |
| icp_misclassification | 26 | 10.2% |
| signal_over_claiming | 25 | 9.8% |
| signal_confidence_alignment | 29 | 11.4% |
| scheduling_edge_cases | 24 | 9.4% |
| multi_thread_leakage | 32 | 12.6% |
| dual_control_coordination | 28 | 11.0% |
| tone_drift | 31 | 12.2% |
| gap_over_claiming | 27 | 10.6% |
| cost_pathology | 2 | 0.8% |

### By Source Mode

| Mode | Count | % |
|------|-------|---|
| Trace-derived | 80 | 31.5% |
| Programmatic | 80 | 31.5% |
| Multi-LLM synthesis | 64 | 25.2% |
| Hand-authored adversarial | 30 | 11.8% |

---

## 3. Contamination Verification

### Checks Performed

| Check | Threshold | Result |
|-------|------------|--------|
| N-gram overlap | <8-gram | Some violations found (TB-0015) |
| Embedding similarity | <0.85 cosine | Passed |
| Time-shift verification | Documented window | Passed |

### Violations Addressed

TB-0015 showed 8-gram overlap with multiple training/dev tasks. This task has been flagged in `contamination_check.json` and will be excluded from the sealed leaderboard if it affects evaluation validity.

---

## 4. Inter-Rater Agreement

### Results (30-task sample)

| Dimension | Exact Match | ±1 Tolerance | Target |
|-----------|-------------|---------------|--------|
| Input coherence | 93.3% | 100% | ≥80% |
| Ground truth verifiability | 90.0% | 100% | ≥80% |
| Rubric application clarity | 86.7% | 96.7% | ≥80% |

**All three dimensions exceed the 80% threshold.**

### Rubric Clarifications

Two minor clarifications were made:
1. TB-0618: Seniority-specific requests check senior subset, not total count
2. TB-0615: Hedge phrases always trigger ask_question action

---

## 5. Path Declaration: Path B (Preference-Tuned Judge)

### Justification

The Week 10 failure taxonomy shows:

| Failure Cluster | Trigger Rate | Path Implication |
|-----------------|-------------|------------------|
| Inconsistency-of-judgment (dual_control, multi_thread_leakage) | 1.00 | Agent gets it right but cannot tell when wrong → Path B |
| Generation-quality (tone_drift, signal_over_claiming) | ≈0.00 | Current prompt + policy holds → not Path A |
| Trajectory failures | N/A | Trace pool too thin → not Path C |

### Algorithm Choice

- **Primary:** SimPO (reference-free, length-normalized, fits Colab T4)
- **Backup:** ORPO (monolithic alternative if SimPO underperforms)

### Papers Informing Choice

- DPO (Rafailov et al., 2023) — foundational, but KL term adds overhead
- SimPO (Meng et al., 2024) — reference-free, fits budget
- ORPO (Hong et al., 2024) — monolithic backup
- Prometheus 2 (Kim et al., 2024) — output format template
- Preference Leakage (Li et al., 2025) — rotation policy constraint

---

## 6. Cost Tracking

### Budget Summary

| Bucket | Budget | Spent | Remaining |
|--------|--------|-------|-----------|
| Dataset authoring | $3-5 | $2.87 | $1.13 |
| Training | $0-5 | $0.00 | $5.00 |
| Held-out evaluation | $2-3 | $0.00 | $2.50 |
| Reserve | $1-2 | $0.00 | $1.50 |
| **Total** | **$10** | **$2.87** | **$7.13** |

### Cost Discipline Compliance

- [x] No τ²-Bench retail re-runs (Week 10 score reused)
- [x] No eval-tier model on Days 2-3 (dev-tier used for iteration)
- [x] All API charges recorded with timestamp and bucket

---

## 7. Remaining Work

### Act III: Method Selection and Training Data Prep (Day 4)

- [ ] Convert training partition to preference pairs format
- [ ] Apply preference-leakage prevention (3-family rotation)
- [ ] Finalize methodology_rationale.md

### Act IV: Train, Ablate, Measure (Days 5-6)

- [ ] Run SimPO training on Colab T4
- [ ] Delta A: Trained vs Week 10 baseline (must be positive, p<0.05)
- [ ] Delta B: Trained vs prompt-engineered version
- [ ] Delta C: Trained vs τ²-Bench retail (if applicable)
- [ ] Cost-Pareto analysis

### Act V: Publish and Engage (Day 7)

- [ ] HuggingFace dataset publication
- [ ] Model card (if Path A or C)
- [ ] Technical blog post
- [ ] Community engagement (GitHub issue)
- [ ] 2-page memo to Tenacious CEO/CFO

---

## 8. Artifacts Summary

### Files Created

```
sales_evaluation_bench/
├── README.md                    # Project overview
├── audit_memo.md                # Gap analysis (590 words)
├── methodology.md               # Path declaration, leakage prevention
├── datasheet.md                 # Full documentation (Gebru + Pushkarna)
├── inter_rater_agreement.md     # 30-task label/re-label
├── schema.json                  # Task schema with examples
├── scoring_evaluator.py         # Machine-verifiable scorer
├── cost/
│   └── log.md                   # Cost tracking
├── synthesis_memos/
│   ├── synthetic_data_memo.md   # Liu et al. synthesis
│   └── datasheet_memo.md        # Gebru/Pushkarna synthesis
├── tenacious_bench_v0.1/
│   ├── composition.json         # Dataset statistics
│   ├── contamination_check.json  # Contamination verification
│   ├── train/tasks.jsonl        # 125 training tasks
│   ├── dev/tasks.jsonl         # 63 public dev tasks
│   └── held_out/tasks.jsonl     # 66 sealed tasks
└── generation_scripts/          # Authoring pipeline
```

---

## 9. Next Steps

1. **Day 4 morning:** Complete training data preparation for Path B
2. **Day 4 afternoon:** Run contamination check against training partition
3. **Day 5 morning:** SimPO training run on Colab T4
4. **Day 5 afternoon:** Begin ablation runs
5. **Day 6:** Complete Delta A/B/C evaluation
6. **Day 7:** Publish artifacts and memo

---

## Appendix: Key References

- Liu et al., "Best Practices and Lessons Learned on Synthetic Data for Language Models" (COLM 2024)
- Gebru et al., "Datasheets for Datasets" (FAccT 2021)
- Pushkarna et al., "Data Cards" (FAccT 2022)
- Chen et al., "Recent Advances in LLM Benchmarks against Data Contamination" (EMNLP 2025)
- Gu et al., "A Survey on LLM-as-a-Judge" (2024-2025)
- SimPO (Meng et al., 2024)
- Li et al., "Preference Leakage" (2025)

---

*Report generated: 2026-04-29*  
*Project: Tenacious Academy Week 11*  
*License: CC-BY-4.0*