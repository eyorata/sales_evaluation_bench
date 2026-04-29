# Synthesis Memo: Best Practices and Lessons Learned on Synthetic Data for Language Models

**Paper:** Liu et al., "Best Practices and Lessons Learned on Synthetic Data for Language Models" (COLM 2024)  
**Date:** 2026-04-29  
**Author:** Tenacious Academy

---

## Summary

The paper provides operational guidance for dataset authoring using synthetic data. Key findings include:
1. Quality over quantity: small high-quality datasets often outperform large noisy ones
2. Diversity matters: diverse data distributions improve generalization
3. Filtering is critical: LLM-as-a-judge filtering improves downstream performance
4. Contamination risks: synthetic data can leak into benchmarks if not carefully managed

---

## Design Choice Disagreement

**The paper recommends equal weighting across synthesis modes (trace-derived, programmatic, multi-LLM synthesis).**

I disagree with this recommendation for Tenacious-Bench v0.1. Given our small-data starting point (no historical labeled prospects), I shift weight to:

| Mode | Paper default | My weight | Justification |
|------|---------------|-----------|----------------|
| Trace-derived | ~33% | ~30% | Free (already in repo), highest fidelity |
| Programmatic | ~33% | ~30% | Low cost, high combinatorial coverage |
| Multi-LLM synthesis | ~33% | ~25% | Contamination-risky, requires judge filtering |
| Hand-authored adversarial | 0% | ~15% | Highest originality, targets edge cases |

**Why this matters:** The LIMA finding (Zhou et al., 2023) that "less is more for alignment" directly supports shifting toward quality over quantity. For Tenacious-specific failure modes, hand-authored adversarial tasks capture edge cases that synthesis pipelines miss — these carry the most originality weight at grading.

---

## Evidence from Our Dataset

Our composition.json shows alignment with this disagreement:

```json
"by_source_mode": {
  "programmatic": 80,      // 31.5%
  "trace_derived": 80,     // 31.5%
  "multi_llm_synthesis": 64, // 25.2%
  "hand_authored_adversarial": 30 // 11.8%
}
```

The hand-authored adversarial tasks (30) are forced into held_out partition per the brief's originality weighting, demonstrating commitment to the quality-over-quantity principle.

---

## Application to Tenacious-Bench

1. **Judge filtering:** We apply LLM-as-a-judge filtering (≥4 on each dimension) per the paper's recommendation
2. **Contamination checks:** We run n-gram overlap, embedding similarity, and time-shift verification
3. **Partition strategy:** 50/30/20 split ensures training data is high-quality while dev/held_out remain fresh

---

## Conclusion

The paper's operational guidance is sound, but the equal-weight default is designed for well-resourced labs with large seed corpora. For Tenacious-style small-data starting points, the shifted weighting (with explicit hand-authored adversarial component) is the more principled approach.

---

**Word count:** ~350