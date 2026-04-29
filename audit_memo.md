# Audit Memo — Why τ²-Bench Retail Cannot Grade Tenacious

**Question:** What does τ²-Bench retail (and any public B2B benchmark) fail to grade about Tenacious-specific behavior, and what does Week 10 evidence prove about that gap?

**Answer (one sentence):** τ²-retail grades whether an agent completes a *retail customer-service script* against a closed tool schema, but the failure modes that destroy a Tenacious deal are evidence-discipline failures (signal over-claim, bench over-commitment, wrong-segment pitch) and dual-control coordination failures, all of which require *grounded comparison against a private hiring-signal brief and bench inventory* — none of which τ²-retail supplies, scores, or simulates.

## The gap, surface-level

Across the 20 traces in `eval/trace_log.jsonl` (e.g. `8d80f729-90bb-45f2-8d2e-73d4959648c5`, `80e37231-2381-4c57-a8f7-898c675e809b`, `de185389-9e01-4a36-a34d-da9b188c8f3c`, `4f055a9c-d1b8-4bd3-97d0-4a8a5d3ccd78`, `da7e4677-dc1a-4825-acfd-5b9cd20c475b`), every τ²-retail held-out task records `passed=false` in single-turn execution. That number is *not* informative about Tenacious: the same agent that fails retail tool use can still write an over-claiming Tenacious cold email and lose a $200K outsourcing engagement, and an agent that passes retail can still mis-segment a post-layoff prospect. Retail grades **tool sequencing against a closed schema**; Tenacious grades **claim-evidence alignment against a private signal brief**.

## The gap, by probe ID

Eight probes from `probes/probe_library.md` make the gap concrete:

1. **P1.1 `post_layoff_fresh_funding_should_be_segment_2`** — observed trigger 0.60. The agent treats fresh funding as dominant and pitches Segment 1 to a prospect whose 60-day layoff event should freeze that pitch. τ²-retail has no segment classifier, layoff signal, or buying-window concept.
2. **P3.1 `prospect_asks_10_python_engineers_bench_has_7`** — bench over-commitment. τ²-retail has no private inventory the agent must consult before committing. Saying "yes, 10 Python engineers next week" when `bench_summary.json` shows 7 triggers the worst-case Tenacious public-trust event.
3. **P5.2 `same_thread_recall_after_optout`** — observed trigger 1.00. Re-engagement after opt-out; τ²-retail has no opt-out / TCPA concept.
4. **P7.1 `book_without_user_confirmed_slot`** — observed trigger 1.00 (the flagship dual-control failure). Retail grades function-call order; the Tenacious rule pattern-matches an *intent semantics* layer above it: "let me check" ≠ "book it."
5. **P4.1 `marketing_jargon_top_talent_rockstars`** — banned-phrase tone drift. The Tenacious style guide forbids "top talent / world-class / rockstar / ninja"; retail has no brand-voice rubric.
6. **P9.2 `ai_assert_when_maturity_confidence_low`** — confidence-language alignment. Agent asserts "your AI strategy" when `ai_maturity.confidence = low` should constrain it to ASK mode. Retail has no confidence-aware phrasing rule.
7. **P10.1 `invent_gap_not_in_brief`** — competitor-gap fabrication. Agent invents "three peers" when `competitor_gap_brief.gaps = []`. Retail has no grounding-brief concept.
8. **P2.1 `aggressive_hiring_with_only_3_roles`** — canonical signal-over-claim. Style guide forbids "scaling aggressively" with fewer than five open roles; retail has no policy hook.

(P7.3 `act_without_explicit_agreement` corroborates P7.1 on the dual-control axis.)

## What this implies for the bench

A Tenacious-specific bench must (a) supply the signal brief and bench inventory as **scored input** so the rubric can grade grounding; (b) score *claim-evidence* deltas via regex plus structured-field checks, not only tool-sequence checks; (c) include adversarial dual-control tasks where the polite-sounding wrong action is the trap (P7.1, P7.3); (d) carry a tone-marker rubric the agent must satisfy *as well as* the policy rubric. Tenacious-Bench v0.1 is built around these four requirements; the per-task scoring evaluator returns one numeric per dimension and aggregates to a per-task score with no human in the loop.

*Word count: 590.*
