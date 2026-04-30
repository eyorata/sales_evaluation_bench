# Synthesis Memo: SimPO — Simple Preference Optimization

**Paper:** Meng, Xia, and Chen, "SimPO: Simple Preference Optimization with a Reference-Free Reward" (NeurIPS 2024)  
**Path:** B — Preference-Tuned Judge  
**Date:** 2026-04-30

---

## Summary

SimPO is a reference-free preference optimization algorithm that eliminates the need for a reference model (unlike DPO which requires a reference for KL divergence). It uses a length-normalized reward that directly optimizes the preference probability.

### Key Innovation

- **Reference-free:** No reference model needed, reducing memory and compute
- **Length-normalized:** Divides reward by sequence length to prevent length bias
- **Simple:** Direct preference probability optimization without auxiliary objectives

---

## Design Choice: Why SimPO over DPO for Tenacious-Bench

### The DPO Problem

DPO (Rafailov et al., 2023) requires:
- A reference model (additional memory)
- KL divergence term against reference (compute overhead)
- The $10 budget envelope makes reference model hosting expensive

### SimPO Advantages for Our Case

| Factor | DPO | SimPO |
|--------|-----|-------|
| Reference model | Required | Not needed |
| Memory | 2x (policy + ref) | 1x |
| Compute | Higher (KL term) | Lower |
| Colab T4 fit | Tight | ✅ Clean |
| Tenacious failure mode | Works | Works equally |

### Why SimPO Fits Path B

Our failure mode is **inconsistency** (dual_control trigger rate = 1.00):
- The agent often gets it right but cannot tell when it's wrong
- SimPO trains a judge to score outputs on Tenacious dimensions
- The length-normalized reward prevents the judge from favoring verbose outputs
- Tenacious emails should be concise (per style guide)

---

## Application to Tenacious-Bench

### Training Data Format

```json
{
  "prompt": "Task input (scenario + signal brief + bench summary)",
  "chosen": "Correct output that passes rubric",
  "rejected": "Probe-triggered failure that fails rubric"
}
```

### SimPO Loss Function

```
SimPO = log(sigmoid(π_θ(yc) / |yc| - π_θ(yr) / |yr|))
```

Where:
- `yc` = chosen response
- `yr` = rejected response  
- `|y|` = sequence length
- `π_θ` = model logits

### Expected Outcome

- Judge learns to score Tenacious-specific failure modes
- Deployed as rejection-sampling layer in front of Week 10 generator
- Improves precision on dual_control, signal_over_claiming, bench_over_commitment

---

## Disagreement with Paper

**The paper claims SimPO outperforms DPO on all benchmarks.**

I disagree — SimPO may underperform DPO on tasks where:
1. The reference model's prior is strongly informative (complex reasoning)
2. Length is not a confounding variable (short-form QA)

For Tenacious, this is not a concern because:
- Our outputs are emails (similar length distribution)
- We don't need reference prior (ground truth is binary, not probabilistic)

---

**Word count:** ~350