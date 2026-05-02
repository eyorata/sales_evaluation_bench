# Tenacious-Bench v0.1

**A domain-specific evaluation benchmark for B2B sales-outreach agents.** 266 mechanically-gradable tasks across 10 Tenacious-specific failure dimensions, plus a SimPO-trained Path B judge with honest Delta A / Delta B reporting.

| Public artifact | URL |
|---|---|
| 🤗 HuggingFace dataset | https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1 (held-out sealed) |
| 🤗 HuggingFace model (LoRA adapter) | *Not required — the brief makes the HuggingFace model upload optional for Path B (judge/critic). Adapter weights remain locally in `outputs/tenacious_judge_simpo/` and the methodology is fully reproducible from `training_scripts/train_simpo.py`.* |
| 📝 Technical blog post | https://dev.to/eyorata/when-your-training-loss-is-lying-to-you-building-a-tenacious-specific-sales-outreach-benchmark-2jgd |
| 💬 Community engagement (τ²-Bench GitHub issue) | https://github.com/sierra-research/tau2-bench/issues/295 |
| 📦 Source code | https://github.com/eyorata/sales_evaluation_bench |

---

## Status

- **Acts I + II + III + IV complete.** v2 SimPO LoRA training run on Qwen2.5-3B with 128 LLM-rewritten preference pairs lands the held-out preference accuracy at **0.417** vs 0.167 untrained baseline (Delta A = +25 pp, p=0.0316).
- **Act V complete.** All required public artifacts live: HuggingFace dataset, blog post on dev.to, τ²-Bench community-engagement issue. HuggingFace model upload is optional for Path B per the brief and is not posted.

---

## The gap (one paragraph)

τ²-Bench retail grades whether an agent completes a retail customer-service script against a closed tool schema. The failure modes that destroy a B2B outsourcing deal — bench over-commitment, signal over-claim, dual-control coordination, condescending gap framing — require grounded comparison against a private hiring-signal brief and bench inventory. Tenacious-Bench grades exactly that, mechanically, no human in the loop. See [`audit_memo.md`](audit_memo.md) (598 words, 9 probes, 5 traces) for the full gap argument.

---

## Dataset summary

| Metric | Value |
|--------|-------|
| Total tasks | **263** (266 authored − 3 dropped by pairwise near-dup gate) |
| Dimensions | 10 |
| Partitions | train (128), dev (63), held_out (75) (post-dedup; counts may shift ±1 with seed) |
| Source modes | 5 (programmatic, trace-derived, multi-LLM synthesis, hand-authored adversarial, style-guide pair) |
| License | CC-BY-4.0 |

| Source mode | n | Share |
|---|---:|---:|
| Programmatic (parameter sweeps) | 80 | 30.4% |
| Trace-derived (Week 10 traces) | 80 | 30.4% |
| Multi-LLM synthesis (DeepSeek↔Qwen3, Llama-3.3-70B as third-family judge) | 61 | 23.2% |
| Hand-authored adversarial | 30 | 11.4% |
| Style-guide pair (verbatim from Tenacious Style Guide v2) | 12 | 4.6% |

The 3 dropped tasks are logged in [`tenacious_bench_v0.1/judge_filter_log.jsonl`](tenacious_bench_v0.1/judge_filter_log.jsonl) with full pairwise rationale (cosine, tie-breaker used, both sides' mean judge scores).

Per-dimension counts are in [`tenacious_bench_v0.1/composition.json`](tenacious_bench_v0.1/composition.json). Datasheet (Gebru + Pushkarna) is at [`datasheet.md`](datasheet.md).

---

## Headline result (v2 training run)

| Metric | Value | Notes |
|---|---:|---|
| Held-out preference accuracy (trained Path B judge, n=12) | **0.417** | 5/12 |
| Held-out preference accuracy (untrained Qwen2.5-3B baseline) | 0.167 | 2/12 |
| **Delta A** (trained vs untrained) | **+25 pp** | 95% CI [0.00, 0.50], paired bootstrap p=**0.0316** |
| **Delta B** (trained vs prompt-engineered same-backbone) | **−42 pp** | 95% CI [−0.75, 0.00], p=0.992 |
| Cost-Pareto (trained latency / prompt latency) | 417 ms / 372 ms | per judgment |
| Total Week 11 spend | $0.041 of $10 envelope | |

**Honest interpretation:** Delta A is directionally positive — training did something — but Delta B is decisively negative; a prompt-engineered Qwen2.5-3B with the v2 style guide as system context beats the trained LoRA judge. This is the *publishable negative finding* the brief explicitly anticipates. Production recommendation: deploy the prompt-engineered judge; ship the LoRA adapter as supporting evidence that the data methodology works. See [`memo/memo_v2_template.md`](memo/memo_v2_template.md) for the 2-page CEO/CFO memo and [`blog/tenacious_bench_v0.1_blog.md`](blog/tenacious_bench_v0.1_blog.md) for the full story.

---

## Quickstart — reproduce the headline number in under 10 minutes

### 1. Setup

```bash
git clone https://github.com/eyorata/sales_evaluation_bench
cd sales_evaluation_bench
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Verify the scoring evaluator

```bash
python scoring_evaluator.py --self-test
# Expected: 6/6 PASS across 3 dummy tasks (chosen + rejected pairs)
```

### 3. Reproduce the dataset deterministically (offline; no API key needed)

```bash
python -m generation_scripts.build_dataset
# Expected: "authored: prog=80 trace=80 syn=64 adv=30 sg=12 total=266"
# Expected: contamination check PASS (n-gram 0, embedding 0, time-shift 0)
```

### 4. Reproduce the headline Delta A from the saved adapter

```python
import json
from scoring_evaluator import score_task

# Read the v2 ablation results computed by docs/act4_colab_path_b.md (Cells 11–15)
results = json.load(open("ablations/ablation_results.json", encoding="utf-8"))
print(f"Delta A trained_acc: {results['delta_a']['trained_acc']}")
print(f"Delta A baseline_acc: {results['delta_a']['baseline_acc']}")
print(f"Delta A raw_lift_pp: {results['delta_a']['raw_lift_pp']}")
print(f"Delta A p_value: {results['delta_a']['bootstrap_p_value']}")
```

To re-train end-to-end:

- **Locally / on RunPod:** `python training_scripts/train_simpo.py`. The script pins the HF model revision (`MODEL_REVISION`), seeds every RNG (`random`/`numpy`/`torch`/`torch.cuda`/`transformers`) from a single `SEED=3407`, and resolves the actual revision SHA at runtime so the run is byte-traceable in `training/training_run.log`.
- **On Colab T4 (free):** open `docs/act4_colab_path_b.md` and run cells 1–17 in order. ~50–60 min.

Both paths produce the same LoRA adapter at the HuggingFace URL above.

### 5. (Optional) Score your own agent against Tenacious-Bench

```python
import json
from scoring_evaluator import score_task

with open("tenacious_bench_v0.1/dev/tasks.jsonl", encoding="utf-8") as f:
    for line in f:
        task = json.loads(line)
        breakdown = score_task(task, your_agent_output(task))
        print(f"{task['task_id']}: {breakdown.total:.3f}  per_check={breakdown.per_check}")
```

---

## Repository structure

```
sales_evaluation_bench/
├── README.md                    # this file
├── LICENSE                      # CC-BY-4.0 (full text)
├── requirements.txt             # pinned dependencies
├── audit_memo.md                # Act I — gap memo (598 words, 9 probes, 5 traces)
├── schema.json                  # task schema + 3 example tasks
├── scoring_evaluator.py         # 9 check types, machine-gradable, --self-test
├── methodology.md               # Path B declaration + 3-family rotation policy
├── methodology_rationale.md     # Day 4 — Path B preference-pair construction (≥3 traces, ≥2 papers)
├── inter_rater_agreement.md     # 30-task hand-label dual-pass; per-dimension agreement matrix
├── inter_rater_agreement.json   # machine output
├── datasheet.md                 # Gebru 7 sections + Pushkarna layered detail
├── evidence_graph.json          # 24 nodes; every numeric claim resolves to a source
│
├── tenacious_bench_v0.1/        # the dataset
│   ├── train/tasks.jsonl        # 128 train tasks
│   ├── dev/tasks.jsonl          # 63 dev tasks
│   ├── held_out/tasks.jsonl     # 75 held-out tasks (sealed before HuggingFace publish)
│   ├── datasheet.md             # canonical datasheet (mirror of root datasheet.md)
│   ├── composition.json         # per-partition × per-dimension counts
│   ├── contamination_check.json # n-gram + embedding + time-shift report (all PASS)
│   ├── judge_filter_log.jsonl   # tasks rejected by the judge filter
│   └── judge_calibration.json   # 50-task Llama-3.3-70B cross-rater spot-check
│
├── training_data/
│   ├── preference_pairs.jsonl       # v1 (templated chosens; kept for ablation history)
│   ├── preference_pairs_v2.jsonl    # v2 (Llama-3.3-70B rewrites; canonical training set)
│   └── prompts_only.jsonl
│
├── training_scripts/
│   └── train_simpo.py           # canonical .py training script — runnable outside Colab,
│                                  # pins HF revision, propagates SEED=3407 to all RNGs,
│                                  # resolves actual revision SHA at runtime, writes
│                                  # training/training_run.log + loss_curve.png
│
├── training/
│   ├── training_run.log         # backbone, hyperparameters, per-step loss, eval rows,
│   │                              # backbone_revision_pinned + backbone_revision_actual
│   └── loss_curve.png           # SimPO training loss + eval-margin overlay
│
├── ablations/
│   ├── ablation_results.json    # Delta A / B / C + Cost-Pareto + 95% CI + p-values
│   └── held_out_traces.jsonl    # per-pair logprobs across all 3 arms
│
├── generation_scripts/          # all authoring code, deterministic from a seed
│   ├── common.py                # shared types + 3-family rotation policy
│   ├── author_programmatic.py   # parameter sweeps over Week 10 probes
│   ├── author_trace_derived.py  # Week 10 traces → Tenacious tasks
│   ├── author_synthesis.py      # multi-LLM author + judge (offline + online modes)
│   ├── author_adversarial.py    # 30 hand-authored adversarial tasks
│   ├── style_guide_seed.py      # 12 verbatim good/bad pairs from Style Guide v2
│   ├── build_dataset.py         # coordinator: author → filter → partition → dedup
│   ├── contamination_check.py   # n-gram + embedding + time-shift
│   ├── judge_calibration.py     # Llama-3.3-70B cross-rater spot-check
│   ├── inter_rater.py           # within-author dual-pass agreement
│   ├── rewrite_chosen_outputs.py # v2 Llama-3.3-70B rewrite for chosen outputs
│   ├── fill_missing_pairs.py    # recover dropped pairs to reach full 128
│   ├── publish_hf_dataset.py    # publish dataset to HuggingFace Hub (auto-strips held-out)
│   ├── openrouter_client.py     # thin OpenRouter API client (cost-logged)
│   └── smoke_test.py            # 1-call sanity check
│
├── synthesis_memos/             # required reading memos
│   ├── synthetic_data_memo.md   # Liu et al. COLM 2024
│   ├── datasheet_memo.md        # Gebru 2021 + Pushkarna FAccT 2022
│   ├── simpo_memo.md            # SimPO (Path B-specific)
│   └── orpo_memo.md             # ORPO (Path B-specific)
│
├── docs/                        # Colab training notebook (markdown)
│   └── act4_colab_path_b.md     # 17 cells; SimPO via TRL CPOTrainer; Qwen2.5-3B
│
├── blog/
│   └── tenacious_bench_v0.1_blog.md   # 2,244-word technical blog post
│
├── memo/
│   └── memo_v2_template.md      # 2-page CEO/CFO memo (decision + skeptic appendix)
│
├── community_engagement/
│   └── tau2_bench_issue.md      # GitHub issue draft for τ²-Bench repo
│
├── cost/
│   ├── log.md                   # per-bucket spend record
│   └── openrouter_calls.jsonl   # per-call usage + cost
│
└── reference/                   # Week 10 seed corpus (read-only)
```

---

## Path B: SimPO-tuned preference judge

**Algorithm:** SimPO via TRL's `CPOTrainer` with `loss_type="simpo"`, β=2.0, simpo_γ=1.0, learning_rate=1e-5, 300 steps on Qwen2.5-3B + LoRA r=16.

**Training data format** (in `training_data/preference_pairs_v2.jsonl`):

```json
{
  "prompt": "Scenario + serialized hiring signal brief + last 3 prior-thread turns",
  "chosen": "LLM-rewritten Tenacious-voice draft (Llama-3.3-70B + v2 style guide)",
  "rejected": "Probe-triggered failure pattern from Week 10 evidence",
  "task_id": "TB-XXXX",
  "dimension": "bench_over_commitment",
  "chosen_provenance": "meta-llama/llama-3.3-70b-instruct:rewrite_v2"
}
```

**Reproduction notebook:** [`docs/act4_colab_path_b.md`](docs/act4_colab_path_b.md). Open in Colab, set `HF_TOKEN` in Colab Secrets, run cells 1 → 17 in order. Wall time ~55 min.

---

## Ablation protocol

| Ablation | What it measures | Implementation |
|---|---|---|
| **Delta A** | Trained model vs Week-10-equivalent (untrained backbone) on held-out preference accuracy. Must be positive at p<0.05. | `docs/act4_colab_path_b.md` Cell 11 |
| **Delta B** | Trained model vs prompt-engineered same-backbone on the same held-out slice. *Honestly reported* even when negative. | `docs/act4_colab_path_b.md` Cell 12 |
| **Delta C** | Trained model vs Week 10 τ²-retail (informational only; no re-run per brief budget rules) | `ablations/ablation_results.json` `delta_c_informational` block |
| **Cost-Pareto** | Per-judgment latency and projected $/judgment | `docs/act4_colab_path_b.md` Cell 13 |

Statistical test: paired bootstrap, 10,000 resamples, two-sided 95% CI, one-sided p-value (`P(trained ≤ baseline)`) per `docs/act4_colab_path_b.md` Cell 14.

---

## Methodology disclosures

These are stated explicitly so reviewers don't have to dig:

1. **Held-out used as during-training validation.** The n=12 ground-truth-bearing held-out slice (the 12 style-guide pairs) was passed as `eval_dataset` to `CPOTrainer`. No weight updates occurred from these pairs (CPOTrainer only updates from `train_dataset`), and no hyperparameter tuning happened between checkpoints — but a strict reading of "sealed held-out" would forbid this. v0.2 will build the validation slice from the dev partition. See [`memo/memo_v2_template.md`](memo/memo_v2_template.md) Page 2 for the full disclosure.

2. **Cross-rater calibration came in below 80%.** Llama-3.3-70B as a third-family calibrator on 49 templated tasks scored 73.5% within ±1 of my own labels. Concentrated in `signal_confidence_alignment` rubric-clarity scores. v0.2 will surface the implicit `expected_mode` field on those tasks.

3. **n=12 held-out preference slice is small.** Bootstrap CI bands on Delta A and Delta B are wide. The directional signal is honest; the precision is constrained by sample size. v0.2 will author chosen/rejected for the 30 hand-authored adversarial held-out tasks to expand the slice.

---

## Cost discipline

- **Total Week 11 spend:** $0.041 of $10 envelope (per `cost/openrouter_calls.jsonl`).
- **No τ²-Bench retail re-runs** (brief forbids; Week 10 number reused informationally).
- **No eval-tier model on Days 2–3** (per brief; Llama-3.3-70B used as third-family calibrator only).
- **Per-call usage logged** with timestamp, model, prompt/completion tokens, cost.

---

## What's next (v0.2 roadmap)

- Expand held-out preference slice from 12 to ~30 by authoring chosen/rejected for the 30 hand-authored adversarial held-out tasks.
- Add `metadata.expected_mode` field to `signal_confidence_alignment` tasks (closes the 73.5% Llama calibration gap).
- Re-train on Qwen2.5-7B-Instruct backbone with the same SimPO recipe; test whether Delta B flips.
- Build a separate dev preference slice so the held-out 12 stays untouched until final scoring.
- Publish a leaderboard at https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1.

---

## License

[CC-BY-4.0](LICENSE) — attribution required, commercial use permitted, modification permitted. License rationale and citation in [`LICENSE`](LICENSE).

## Citation

```bibtex
@dataset{tenacious_bench_v01_2026,
  title  = {Tenacious-Bench: a B2B sales-outreach evaluation benchmark for engineering-outsourcing agents},
  author = {Nebiyu, Eyoel and 10Academy TRP1 cohort},
  year   = 2026,
  version = {0.1},
  license = {CC-BY-4.0},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1}
}
```

## Attribution and credits

- **Author:** Eyoel Nebiyu (10Academy TRP1 cohort, 2026).
- **Source corpus:** Week 10 trace pool, probe library, failure taxonomy, and Tenacious Style Guide v2 were committed-as-given by the 10Academy program.
- **Public sources referenced:** Crunchbase ODM 1,001-company sample, layoffs.fyi CSV, all within the documented 2025-11-01..2026-04-29 window.
- **Models used:** DeepSeek-V3.2 + Qwen3-Next-80B-A3B + Llama-3.3-70B (all via OpenRouter dev-tier) for synthesis and rewriting; Qwen2.5-3B-Instruct (via Unsloth + TRL `CPOTrainer`) as the trained Path B backbone.
- **Algorithmic foundations:** SimPO (Meng, Xia, Chen, NeurIPS 2024); preference-leakage protocol (Li et al., 2025); contamination-check protocol (Chen et al., EMNLP 2025); datasheet template (Gebru et al., 2021 + Pushkarna et al., FAccT 2022); synthesis-mode weighting (Liu et al., COLM 2024).
- **Prior work referenced:** τ²-Bench retail (Sierra Research) as the baseline this benchmark complements rather than replaces.

The Tenacious entity, ICP segments, bench inventory, and Style Guide v2 are part of the 10Academy TRP1 program corpus. No real prospect data is included; all synthetic prospect names are fictitious.
