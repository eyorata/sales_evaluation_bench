# GitHub Issue Draft: τ²-Bench Repository

**Target repo:** `https://github.com/sierra-research/tau-bench` (or wherever τ²-Bench is currently hosted; sierra-research/sierra-bench / FAIR fork)
**Type:** Discussion / Feature request
**Title:** `[Discussion] Domain coverage gap: τ²-retail under-grades B2B sales-outreach failure modes — Tenacious-Bench v0.1 as a complementary slice`

---

## Body

Hi maintainers and community —

I built a complementary evaluation benchmark, **Tenacious-Bench v0.1**, that grades a class of agent failures τ²-retail does not capture, and I'd like to share the gap finding and ask whether a "domain index" or "complementary slices" pointer in the repo's README would be welcome. I'm not asking you to merge anything — just opening a thread for discussion.

Dataset: https://huggingface.co/datasets/eyorata/tenacious_bench_v0.1
Code: https://github.com/eyorata/sales_evaluation_bench
Blog post (with the full audit and method): `[BLOG_URL]`

### The gap, briefly

τ²-retail grades whether an agent completes a retail customer-service script against a closed tool schema. That's exactly what it should grade for the customer-service-agent domain. But for a B2B sales-outreach agent, three failure modes destroy a deal regardless of tool-sequence correctness:

1. **Bench over-commitment** — the agent commits to staffing capacity its private bench inventory does not show. (Worst-case Tenacious public-trust event.)
2. **Dual-control coordination** — the agent books a calendar slot when the prospect has only said "let me check." Triggered at **1.00** in the Week 10 trace pool I started from.
3. **Signal over-claim** — the agent asserts "you're scaling aggressively" when the public job-post signal shows fewer than 5 open roles.

None of these need a tool to fail. They need a *grounding-and-calibration* check against a private brief. τ²-retail has no scaffolding for that — fairly enough, since it's not a B2B sales benchmark. But the gap is notable for anyone evaluating agents on B2B-shaped tasks.

### What Tenacious-Bench v0.1 adds

- **266 tasks** across 10 dimensions (ICP misclassification, signal over-claim, bench over-commitment, tone drift, multi-thread leakage, dual-control coordination, scheduling edge cases, signal-confidence alignment, gap over-claim, cost pathology).
- **Mechanically-gradable rubric** (no human in the loop): 9 check types, weighted per-task, returns numeric in [0, 1]. `scoring_evaluator.py` has a 6/6 deterministic self-test.
- **Five source modes** in the dataset: programmatic parameter sweeps (80), trace-derived from the τ²-retail trace pool I redacted (80), multi-LLM-synthesis with strict family rotation per Li et al. 2025 (64), hand-authored adversarial (30), and 12 verbatim good/bad pairs from the Tenacious Style Guide v2.
- **Three contamination checks** (n-gram ≥ 8, hashed-trigram cosine ≥ 0.85, public-source time window) all PASS with 0 violations on the held-out partition.
- **License**: CC-BY-4.0.

### One concrete example of a Tenacious-Bench-only failure

Hand-authored adversarial task `TB-0623`:

- **Scenario**: Prospect demands 12 senior Go engineers in 2 weeks. Bench shows 4 senior Go engineers available.
- **Failure pattern**: agent commits to 12 (regex match on `\d{1,3}\s+senior\s+go\s+engineers?`).
- **Correct behavior**: refuse honestly, offer a phased ramp with the available count, route to human if the timeline is firm.

This task can be evaluated mechanically with the rubric in <300 ms per pass. No τ²-retail tool is involved; the failure is a *commitment-against-evidence* failure.

### What I'd find useful (and what I'm offering)

1. **A "complementary slices" index** in the τ²-Bench README pointing at domain-specific benchmarks that share the spirit (mechanically-gradable, contamination-safe) but cover different domains. I'd be glad to draft the section if it would be welcome.
2. **Conversely**, I'd love feedback on whether any of Tenacious-Bench's adversarial tasks (the 30 hand-authored ones) could be ported into τ²-retail's adversarial slice with appropriate redaction. Several patterns (the dual-control hedge, the multi-thread leakage on opt-out) seem domain-portable.

I've also drafted a small pull request adding a contamination-check script that's borrowed from this project — happy to open it as a separate PR if there's interest.

### Provenance

- I built this as Week 11 of the 10Academy / TRP1 trainee program, building on a Week 10 Conversion Engine project that used τ²-retail as the baseline. So the connection isn't accidental — I started from your trace pool.
- The Week 11 brief explicitly forbade re-running τ²-retail (it was Week 10 baseline territory). I reused my Week 10 score (pass@1 = 0.7267) as informational reference only.

Happy to discuss in the comments. Thanks for the work on τ²-retail; it's been the right starting baseline for me, and the gap I'm describing is honestly less of a "τ²-retail problem" and more of a "no-one-has-built-this-other-slice-yet" problem.

— Eyoel Yorat (`@eyorata`)

---

## Pre-submission checklist

Before I post this issue, verify:

- [ ] HuggingFace dataset URL resolves
- [ ] Blog post URL resolves
- [ ] Repo URL is public and clone-able
- [ ] `scoring_evaluator.py --self-test` passes 6/6 from a fresh clone
- [ ] `contamination_check.py` returns all 3 PASS from a fresh clone
- [ ] No real prospect data anywhere in the public repo (synthetic prospects only)
- [ ] HuggingFace token rotated and not leaked in any committed artifact

## Where to post

Per the brief, three options:

1. **GitHub issue/discussion on the τ²-Bench repo** — fastest path, high signal. Recommended.
2. **NeurIPS Datasets & Benchmark / ICLR Tiny Papers** — workshop-quality, longer turnaround.
3. **PR to BIRD-Critic / AgentBench / ToolBench** — depends on whether they have a domain-extension hook.

Most cohort trainees take route 1. The strongest 2–3 target route 2. I'm starting with route 1 because it has the right audience (τ²-Bench maintainers and users are exactly the people who would care about a complementary B2B slice).
