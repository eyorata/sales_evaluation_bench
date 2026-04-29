# Tenacious-Bench v0.1

**A Domain-Specific Evaluation Benchmark for Tenacious-Style B2B Sales Agents**

---

## Overview

Tenacious-Bench v0.1 is an evaluation benchmark designed to assess B2B sales agents on Tenacious-specific failure modes that existing benchmarks (including τ²-Bench retail) fail to capture.

### The Gap

τ²-Bench retail grades whether an agent completes a retail customer-service script against a closed tool schema. However, the failure modes that destroy a Tenacious deal are:

1. **Evidence-discipline failures**: signal over-claiming, bench over-commitment, wrong-segment pitch
2. **Dual-control coordination failures**: booking without explicit user commitment

These require grounded comparison against a private hiring signal brief and bench inventory — none of which τ²-Bench supplies or scores.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total tasks | 254 |
| Dimensions | 10 |
| Partitions | train (125), dev (63), held_out (66) |
| License | CC-BY-4.0 |

### Dimensions

- bench_over_commitment (30)
- icp_misclassification (26)
- signal_over_claiming (25)
- signal_confidence_alignment (29)
- scheduling_edge_cases (24)
- multi_thread_leakage (32)
- dual_control_coordination (28)
- tone_drift (31)
- gap_over_claiming (27)
- cost_pathology (2)

### Source Modes

- Trace-derived: 80 (31.5%)
- Programmatic: 80 (31.5%)
- Multi-LLM synthesis: 64 (25.2%)
- Hand-authored adversarial: 30 (11.8%)

---

## Quickstart

### 1. Clone the dataset

```bash
git clone https://huggingface.co/datasets/{username}/tenacious_bench_v0.1
cd tenacious_bench_v0.1
```

### 2. Run the scorer

```bash
python scoring_evaluator.py --self-test
```

### 3. Evaluate your agent

```python
from scoring_evaluator import score_task

with open("tenacious_bench_v0.1/dev/tasks.jsonl") as f:
    for line in f:
        task = json.loads(line)
        score, breakdown = score_task(task, your_agent_output)
        print(f"{task['task_id']}: {score}")
```

---

## Project Structure

```
sales_evaluation_bench/
├── README.md                    # This file
├── audit_memo.md                # Why τ²-Bench fails for Tenacious
├── methodology.md               # Path declaration, leakage prevention, partitioning
├── datasheet.md                 # Full dataset documentation (Gebru + Pushkarna)
├── inter_rater_agreement.md     # 30-task label/re-label results
├── schema.json                  # Task schema with 3 examples
├── scoring_evaluator.py         # Machine-verifiable scorer
├── cost/
│   └── log.md                   # Cost tracking ($10 envelope)
├── synthesis_memos/
│   ├── synthetic_data_memo.md   # Liu et al. synthesis
│   └── datasheet_memo.md        # Gebru/Pushkarna synthesis
├── tenacious_bench_v0.1/
│   ├── composition.json         # Dataset statistics
│   ├── contamination_check.json  # N-gram, embedding, time-shift verification
│   ├── train/tasks.jsonl        # 125 training tasks
│   ├── dev/tasks.jsonl         # 63 public dev tasks
│   └── held_out/tasks.jsonl     # 66 sealed held-out tasks
├── generation_scripts/          # Dataset authoring scripts
│   ├── author_trace_derived.py
│   ├── author_programmatic.py
│   ├── author_synthesis.py
│   ├── author_adversarial.py
│   ├── build_dataset.py
│   └── contamination_check.py
└── reference/                   # Week 10 artifacts
    ├── week10_probe_library.md
    ├── week10_failure_taxonomy.md
    └── week10_style_guide.md
```

---

## Path B: Preference-Tuned Judge

This benchmark is designed for **Path B** (preference-tuned judge) training:

1. **Training data:** Construct preference pairs (chosen, rejected) from probe-triggered failures
2. **Algorithm:** SimPO or ORPO (reference-free, lower cost than DPO)
3. **Backbone:** Qwen 3.5 0.8B or 2B with LoRA
4. **Compute:** Free on Google Colab T4

### Training Data Format

```json
{
  "prompt": "Task input from tenacious_bench_v0.1/train/",
  "chosen": "Correct output (passes rubric)",
  "rejected": "Probe-triggered failure (fails rubric)"
}
```

---

## Evaluation

### Ablation Protocol

| Ablation | What it measures |
|----------|-------------------|
| Delta A | Trained model vs Week 10 baseline on held-out (must be positive, p<0.05) |
| Delta B | Trained model vs prompt-engineered version (no training) |
| Delta C | Trained model vs τ²-Bench retail (if Week 10 score exists) |

### Scoring

The `scoring_evaluator.py` returns a numeric score in [0,1] with no human in the loop. Each task has weighted checks:

- `banned_phrase_absent`: No prohibited phrases
- `required_phrase_present`: Required elements present
- `grounded_signal_reference`: References signals from brief
- `no_capacity_overcommit`: Doesn't promise more than bench has
- `action_class`: Correct action type (draft_outbound, ask_question, escalate_human)
- `policy_compliant`: Passes policy checks
- `tone_marker_judge`: Scores ≥4/5 on Tenacious tone markers

---

## Contributing

1. Fork the dataset on HuggingFace
2. Add tasks following the schema in `schema.json`
3. Run `generation_scripts/contamination_check.py` to verify no leakage
4. Maintain ≥80% inter-rater agreement on new tasks
5. Submit PR with updated `composition.json`

---

## License

CC-BY-4.0 — Attribution required.

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

---

## Acknowledgments

- Tenacious executive team for the evaluation challenge
- τ²-Bench for the evaluation framework
- OpenRouter for multi-LLM synthesis infrastructure
- Unsloth for training compute