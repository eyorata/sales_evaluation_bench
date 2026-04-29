# Methodology — Tenacious-Bench v0.1

This document is the methodology record required by the Week 11 brief. Sections answer the five required questions plus the multi-LLM rotation policy that prevents preference leakage.

---

## 1. Path declaration: Path B (preference-tuned judge)

**Declared path: B — preference-tuned critic.** I will train a small classifier/preference scorer (DPO or SimPO/ORPO) that grades agent outputs on Tenacious dimensions and is deployed as a rejection-sampling layer in front of the Week 10 generator.

### Justification (from Week 10 evidence)

The Week 10 failure taxonomy (`reference/week10_failure_taxonomy.md`) shows two clusters:

| Failure cluster | Trigger rate | Path implication |
|---|---|---|
| Inconsistency-of-judgment failures (`dual_control_coordination` 1.00 on `P7.1`, `multi_thread_leakage` 1.00 on `P5.2`) | 1.00 | The agent often gets it right, but cannot tell when its own draft is wrong. → Path B (judge). |
| Generation-quality failures (`tone_drift` 0.00, `signal_over_claiming` 0.00 on run probes) | ≈0.00 | The current prompt + policy stack already holds tone and signal-grounding when probes are run; an SFT generation rewrite would have nothing to lift. → not Path A. |
| Trajectory failures (multi-turn compounding) | not measurable from current 20-trace single-turn snapshot | Path C is appealing on theory but data prep is the bottleneck; the trace pool is too thin for stepwise process labels. → not Path C. |

The decision is forced by the data, not by curiosity. The decisive observable is `P7.1 book_without_user_confirmed_slot` triggering at 1.00: every time the prospect says "let me check," the Week 10 agent goes ahead and books. A judge that can score "this output books when it shouldn't" against the same brief is a **deterministic rejection layer** the generator can be sampled into until it produces an acceptable draft. That is exactly the Path B production-relevance pattern named in the brief.

### Trace Evidence of Path B-Relevant Failures (≥3 required)

The following Week 10 traces explicitly illustrate Path B-relevant failures where the agent struggled with judgment and boundaries rather than basic fluency:

- `8d80f729-90bb-45f2-8d2e-73d4959648c5` (retail_ho_0::5, single-turn `passed=false`) — Illustrates a Path B-relevant **dual-control failure**. The agent prematurely booked a slot without final confirmation. Used as the seed for **TB-0002**.
- `80e37231-2381-4c57-a8f7-898c675e809b` (retail_ho_1::9) — Illustrates a Path B-relevant **multi-thread leakage failure**. The agent failed to keep state strictly separated. Used as the seed for the parametric expansion (`P5.2` family).
- `de185389-9e01-4a36-a34d-da9b188c8f3c` (retail_ho_2::12) — Illustrates a Path B-relevant **bench-over-commitment failure**. Basis for the programmatic sweep (`P3.1`-derived TB-0001).
- (Supporting) `4f055a9c-d1b8-4bd3-97d0-4a8a5d3ccd78` and `da7e4677-dc1a-4825-acfd-5b9cd20c475b` illustrate ICP misclassification and are reused for the dev-partition.

### Papers informing the Path B choice (≥2 required)

- **SimPO (Meng, Xia, Chen, NeurIPS 2024).** Reference-free, length-normalized formulation. Our Week 10 traces showed generation-quality was already high, with failures concentrated strictly in judgement boundaries (e.g., booking without final confirmation). This means we don't need a heavy reference model to maintain fluency (as in DPO). SimPO is computationally efficient enough to fit our Colab T4 envelope while perfectly addressing these discrete, binary boundary violations.
- **Prometheus 2 (Kim et al., 2024).** This work demonstrated that smaller LLMs can act as high-quality judges if trained on specific critique formats. Connected directly to my Week 10 evidence, our B2B agent needs to execute discrete judgements (e.g., checking if the bench has capability X before committing). I designed my judge utilizing Prometheus 2's pattern: outputting a criteria-aware critique followed by a discrete score, which empirically scales down effectively to catching specific B2B failures like `dual_control_coordination`.
- **Preference Leakage (Li et al., 2025).** Direct constraint on the rotation policy below.

The full path-specific synthesis memos (one per paper, ≤1 page) will land in `synthesis_memos/` on Day 4. The two common-reading memos (Liu synthetic data, Gebru/Pushkarna datasheets) ship with this interim submission.

---

## 2. Preference-leakage prevention (per Li et al., 2025)

**Hard rule: no task is generated and judged by the same model family.** Same family = same provider's same generation lineage (e.g. all DeepSeek-V3.x variants count as one family; Qwen3 variants are one family; Llama-3.x variants are one family).

### Rotation policy used in `generation_scripts/`

| Stage | Model A (allowed roles) | Model B (allowed roles) | Why |
|---|---|---|---|
| Multi-LLM synthesis hard seeds (~30–50) | DeepSeek-V3.2 (author) | Qwen3-Next-80B-A3B (judge) | Different families, ~equal cost. |
| Bulk programmatic variations (~60–90) | Qwen3-Next-80B-A3B (author) | DeepSeek-V3.2 (judge) | Inverted from above so each model authors and judges roughly half. |
| Chosen-output rewrites for preference pairs (Path B training data, Day 4) | DeepSeek-V3.2 OR human | Llama-3.3-70B (judge) — third family, never used to author | Three-family rotation eliminates the same-family chosen/judge collision Li et al. flag. |
| Sealed held-out scoring (Day 6) | n/a (agent under test produces output) | Claude Sonnet 4.6 (eval-tier judge) | Eval-tier model never sees training-partition tasks during authoring. |

The rotation is logged per task in `metadata.author_model` and `metadata.judge_model`. A static check (`generation_scripts/check_no_leakage.py`) refuses any task where the family of `author_model` matches the family of `judge_model`.

---

## 3. Partitioning protocol

- **train: 50%** — exposed to the Path B preference-pair construction pipeline.
- **dev: 30%** — public, for trainees and external users to iterate on.
- **held_out: 20%** — sealed; not committed to the public repo until the leaderboard is published; used for Delta A/B/C only at eval-tier.

Stratification is by `dimension` so each partition contains the same proportional mix of all 10 failure dimensions. Within `dimension`, hand-authored adversarial tasks are **forced into held_out** (per the brief's originality weighting), trace-derived tasks are split 60/30/10 across partitions, and programmatic + multi-LLM-synthesis tasks are split 50/30/20.

---

## 4. Contamination checks (script: `generation_scripts/contamination_check.py`)

All three checks run before any task enters `held_out/`:

1. **N-gram overlap.** For every (held_out_task, train_or_dev_task) pair, compute the longest contiguous n-gram overlap on the concatenated input fields (`scenario` + `prior_thread.body`). **Reject any pair with ≥8-gram overlap.**
2. **Embedding similarity.** Encode the same concatenated input field with `sentence-transformers/all-MiniLM-L6-v2` (cheap, deterministic, free). Reject any pair with cosine similarity ≥0.85.
3. **Time-shift verification.** Any task that references a public source (Crunchbase funding, layoffs.fyi event, BuiltIn job post) must carry a `metadata.public_source_window` field naming the window the data is drawn from. Tasks referencing windows that postdate the dataset publication date are rejected (no leakage from future state into a v0.1 frozen at 2026-04-29).

Results are written to `tenacious_bench_v0.1/contamination_check.json` (counts per check, list of rejected/repaired pairs).

### Summary of Contamination Outcomes

Based on the latest evaluation run:
1. **N-gram overlap:**
   - **Flags:** 17 violations found (held_out task `TB-0015` matched ≥8-grams with 17 other tasks).
   - **Actions Taken:** Rejected `TB-0015` from the `held_out` partition for regeneration/removal.
   - **Final Pass Status:** FAIL (resolution applied post-run).
2. **Embedding similarity:**
   - **Flags:** 0 violations (no task pairs exceeded 0.85 cosine similarity).
   - **Actions Taken:** None required.
   - **Final Pass Status:** PASS.
3. **Time-shift verification:**
   - **Flags:** 0 violations (no tasks leaked future knowledge past 2026-04-29).
   - **Actions Taken:** None required.
   - **Final Pass Status:** PASS.

---

## 5. Inter-rater agreement protocol

- 30-task subset, sampled stratified across dimensions (3 per dimension) and across difficulty (proportional).
- I label all 30 against the rubric on Day 2 evening (T0).
- I re-label the same 30 on Day 3 evening (T0 + 24h) without looking at T0 labels.
- Agreement target: **≥80% on each of the three rubric meta-dimensions** (input coherence, ground-truth verifiability, rubric-application clarity), measured as percent of tasks with identical 1–5 score *or* score within ±1.
- If any dimension is below 80%: revise the rubric, re-label, log the revision in `inter_rater_agreement.md`.

---

## 6. Synthesis-memo disagreements (anchored to ≥2 papers)

To meet the methodology grading criterion ("disagree with the paper on a specific design choice"):

- **Liu et al. on synthetic data:** I disagree with their default split-equal weighting across synthesis modes. Tenacious's small-data starting point makes hand-authored adversarial the highest-yield mode per task-hour; I shift weight to ~15% adversarial-first, ~30% trace-derived (free), and only ~25% multi-LLM synthesis (the most contamination-risky bucket). Memo justifies this against the LIMA finding that small-quality dominates.
- **Gebru et al. on datasheets:** I disagree with the implicit single-author voice. For Tenacious-Bench v0.1 the *author* is "trainee + program staff sign-off"; the datasheet records both, and the Pushkarna-style microscopic detail layer documents per-task provenance (author_model + judge_model + source_probe_id) so a reader can audit any single row without re-reading the whole sheet.

Full memos in `synthesis_memos/`.
