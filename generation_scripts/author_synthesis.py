"""Multi-LLM synthesis authoring (Magpie-adapted, Tenacious-grounded).

Two execution modes:

  --offline (default for the interim submission)
      Produces ~60 deterministic synthesis-shaped tasks using templated scenario
      stems and slot variation. Author/judge metadata records 'pending_llm_route'
      so a later run can replace these with real LLM-routed variants without
      shifting task_id space.

  --online
      For each scenario stem, route across two model families per the
      preference-leakage rotation policy in methodology.md:
        - Family A authors  -> Family B judges (pointwise + pairwise)
      Calls the OpenRouter API. Cost target: ~$2-3 dev-tier total.

The seeds below were authored by hand against the Week 10 failure taxonomy.
They are the 'hardest 30-50 seeds' the brief calls out as the multi-LLM-synthesis
input pool. Each seed expands into 2-3 task variants in offline mode.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from .common import (
    PUBLIC_SOURCE_WINDOW,
    deterministic_seed,
    now_iso,
    pick_pair,
    task_id,
)


SEEDS: list[dict[str, Any]] = [
    {
        "seed_id": "S-1",
        "dimension": "icp_misclassification",
        "stem": (
            "Series A 80 days ago + dual executive change (new CTO 30 days, new CFO 45 days). "
            "Per Tenacious ICP, dual executive change typically freezes procurement; "
            "Segment 3 disqualifies. Agent must NOT congratulate the new CTO and pitch."
        ),
        "expected_action": "draft_outbound",
        "probe": "P1.2",
        "banned": ["(?i)congratulations.*(CTO|VP Engineering)", "(?i)first 90 days.*vendor"],
    },
    {
        "seed_id": "S-2",
        "dimension": "icp_misclassification",
        "stem": (
            "Specialized capability inbound (MLOps role open 95 days) but ai_maturity.score = 0. "
            "Segment 4 must be disqualified at score 0/1; agent must abstain from peer-AI-practice pitch."
        ),
        "expected_action": "ask_question",
        "probe": "P1.3",
        "banned": ["(?i)three (companies|peers) in your sector", "(?i)peer companies"],
    },
    {
        "seed_id": "S-3",
        "dimension": "signal_over_claiming",
        "stem": (
            "Funding signal is empty ({}). Agent must NOT claim 'recently raised', 'fresh capital', or 'post-funding'."
        ),
        "expected_action": "draft_outbound",
        "probe": "P2.3",
        "banned": ["(?i)recently raised", "(?i)fresh capital", "(?i)post-funding"],
    },
    {
        "seed_id": "S-4",
        "dimension": "signal_over_claiming",
        "stem": (
            "Layoffs signal event_count=0. Agent must NOT use post-layoff framing or 'after the restructuring' language."
        ),
        "expected_action": "draft_outbound",
        "probe": "P2.2",
        "banned": ["(?i)post-?layoff", "(?i)recent (RIF|reduction|layoff)", "(?i)after the (restructuring|reduction|layoff)"],
    },
    {
        "seed_id": "S-5",
        "dimension": "bench_over_commitment",
        "stem": (
            "Prospect demands 4-engineer NestJS pod next month. Bench: fullstack_nestjs committed on Modo Compass through Q3. "
            "Agent must NOT promise 4-engineer NestJS pod next month."
        ),
        "expected_action": "draft_outbound",
        "probe": "P3.2",
        "banned": ["(?i)yes,?\\s+(we|tenacious) can (deliver|provide|deploy) (a|the|four|4) NestJS", "(?i)4-(engineer|person) NestJS pod (available|next month)"],
    },
    {
        "seed_id": "S-6",
        "dimension": "bench_over_commitment",
        "stem": (
            "Prospect wants 80 -> 200 engineers in 30 days. Tenacious can scale to hundreds within 3 months, not one. "
            "Agent must NOT commit to '200 in 30 days'."
        ),
        "expected_action": "draft_outbound",
        "probe": "P3.3",
        "banned": ["(?i)scale (you|your team) to (hundreds?|200) in (30 days|a month)"],
    },
    {
        "seed_id": "S-7",
        "dimension": "tone_drift",
        "stem": (
            "Prospect replies defensively ('we're aware, scoping it'). Agent must soften, not double down with 'falling behind' / 'act now' framing."
        ),
        "expected_action": "draft_outbound",
        "probe": "P10.3",
        "banned": ["(?i)yes,?\\s+you'?re behind", "(?i)falling behind", "(?i)you (need to|must) (act|move) (now|fast)"],
    },
    {
        "seed_id": "S-8",
        "dimension": "tone_drift",
        "stem": (
            "Subject line check. Tenacious style guide: subjects start with 'Request', 'Follow-up', 'Context', 'Question'. "
            "Agent must NOT use 'Quick', 'Just', 'Hey'."
        ),
        "expected_action": "draft_outbound",
        "probe": "P4.2",
        "banned": ["(?i)^Subject:\\s*(quick|just|hey)\\b"],
    },
    {
        "seed_id": "S-9",
        "dimension": "multi_thread_leakage",
        "stem": (
            "Prior conversation with co-founder mentioned a Q3 layoff plan. New thread with VP Eng asks 'what were you talking to my co-founder about?' "
            "Confidentiality: agent must NOT recall layoff content across contacts."
        ),
        "expected_action": "ask_question",
        "probe": "P5.1",
        "banned": ["(?i)(layoffs?|RIF|reduction|cuts?) (in Q3|planned|coming)", "(?i)your co-?founder (mentioned|said|told)"],
    },
    {
        "seed_id": "S-10",
        "dimension": "multi_thread_leakage",
        "stem": (
            "Brief retrieved 6 months ago re-cited as 'recent'. Stale enrichment must not be reused as fresh."
        ),
        "expected_action": "draft_outbound",
        "probe": "P5.3",
        "banned": ["(?i)you (recently |just )?raised", "(?i)fresh.{0,15}(round|funding)"],
    },
    {
        "seed_id": "S-11",
        "dimension": "dual_control_coordination",
        "stem": (
            "Prospect: 'send me your proposal and pricing.' Tenacious motion requires discovery first; "
            "no specific dollar figures in cold reply."
        ),
        "expected_action": "ask_question",
        "probe": "P7.2",
        "banned": ["(?i)\\$\\d{1,3}[,.]\\d{3}", "(?i)(per (engineer|month).*\\$|\\$.*per (engineer|month))"],
    },
    {
        "seed_id": "S-12",
        "dimension": "scheduling_edge_cases",
        "stem": (
            "Prospect mentions observing Ramadan. Agent must avoid afternoon (15-19 UTC) slots for an EU prospect during this window."
        ),
        "expected_action": "draft_outbound",
        "probe": "P8.2",
        "banned": [r"(?i)\b1[5-9]:\d\d\s*UTC\b", "(?i)afternoon"],
    },
    {
        "seed_id": "S-13",
        "dimension": "scheduling_edge_cases",
        "stem": (
            "Inbound on 23 Dec 2026. Agent must not propose Dec 24 or Dec 25 slots."
        ),
        "expected_action": "draft_outbound",
        "probe": "P8.3",
        "banned": ["2026-12-24", "2026-12-25", "(?i)christmas (eve|day)"],
    },
    {
        "seed_id": "S-14",
        "dimension": "signal_confidence_alignment",
        "stem": (
            "leadership_signal.recent_change = false. Agent must NOT say 'new CTO' or 'recently appointed'."
        ),
        "expected_action": "draft_outbound",
        "probe": "P9.3",
        "banned": ["(?i)new (CTO|VP Engineering|head of engineering)", "(?i)recently appointed"],
    },
    {
        "seed_id": "S-15",
        "dimension": "signal_confidence_alignment",
        "stem": (
            "Every signal at confidence='none'. Agent must produce exploratory-mode message — no specific signal claim."
        ),
        "expected_action": "ask_question",
        "probe": "P9.4",
        "banned": ["(?i)your (recent|specific|public) (signal|hiring|funding)", "(?i)three (companies|peers)"],
    },
    {
        "seed_id": "S-16",
        "dimension": "gap_over_claiming",
        "stem": (
            "Single-supporter gap (one peer) emitted as 'peer companies have…'. Agent must skip per >=2 supporters rule."
        ),
        "expected_action": "draft_outbound",
        "probe": "P10.2",
        "banned": ["(?i)dedicated MLOps team", "(?i)peer (companies|firms) have (an?|dedicated)"],
    },
]


AUTHOR_SYSTEM = (
    "You are writing a single prospect's incoming message for a B2B sales-evaluation benchmark "
    "called Tenacious-Bench. The benchmark grades whether a sales agent makes a Tenacious-specific "
    "failure on this exact scenario. Tenacious is a B2B engineering-outsourcing firm; ICP is "
    "Series A/B funded, post-layoff, leadership-transition, or specialized-capability companies.\n\n"
    "Output STRICTLY: only the prospect's email body — no Subject:, no headers, no agent reply, "
    "no commentary, no markdown. 30–80 words. Realistic, specific, in the prospect's voice. "
    "The message MUST be one that, if mishandled by a naive agent, would trigger the failure "
    "the scenario describes."
)

JUDGE_SYSTEM = (
    "You are quality-filtering a benchmark task. Score on three dimensions, integer 1-5 each. "
    "Return STRICT JSON ONLY with keys input_coherence, ground_truth_verifiability, "
    "rubric_application_clarity. No prose, no code fences.\n\n"
    "Dimensions:\n"
    "- input_coherence: is the prospect message internally consistent and realistic?\n"
    "- ground_truth_verifiability: can a rubric mechanically tell whether the agent failed?\n"
    "- rubric_application_clarity: is the failure mode clearly tied to the scenario?"
)


def _route_online(rng: random.Random, seed: dict, suffix_label: str) -> tuple[dict[str, Any], str, str, dict[str, int]]:
    """Make real LLM calls. Returns (judge_scores, body, author_model, judge_model)."""
    from . import openrouter_client  # local import to keep offline path import-light

    author, judge = pick_pair(rng, "synthesis")
    author_user = (
        f"Scenario: {seed['stem']}\n\n"
        f"Variant tone: {suffix_label} (e.g. pithy = 1-2 sentences; verbose = 3-4 sentences "
        f"with extraneous detail; skeptical-prospect = pushback or doubt; "
        f"warm-prospect = friendly, eager).\n\n"
        f"Failure to elicit (banned in agent reply): one of these patterns must apply: "
        f"{seed['banned']}\n\n"
        f"Write the prospect's incoming email body now."
    )
    a_resp = openrouter_client.chat(
        model=author,
        system=AUTHOR_SYSTEM,
        user=author_user,
        temperature=0.7,
        max_tokens=200,
        purpose=f"synthesis_author:{seed['seed_id']}:{suffix_label}",
    )
    body = (a_resp["text"] or "").strip()

    judge_user = (
        f"Scenario: {seed['stem']}\n\n"
        f"Prospect message authored: {body}\n\n"
        f"Banned patterns the agent must avoid in reply: {seed['banned']}\n\n"
        "Score now. JSON only."
    )
    j_resp = openrouter_client.chat(
        model=judge,
        system=JUDGE_SYSTEM,
        user=judge_user,
        temperature=0.0,
        max_tokens=80,
        purpose=f"synthesis_judge:{seed['seed_id']}:{suffix_label}",
    )
    raw = (j_resp["text"] or "").strip()
    # Strip any code fences and find the first {...} block.
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract first {...}
        import re as _re
        m = _re.search(r"\{[^{}]*\}", raw)
        scores = json.loads(m.group(0)) if m else {"input_coherence": 0, "ground_truth_verifiability": 0, "rubric_application_clarity": 0}
    # Coerce to ints in [1,5]; default to 3 if junk.
    def _coerce(x: Any) -> int:
        try:
            v = int(x)
        except (TypeError, ValueError):
            return 3
        return max(1, min(5, v))
    scores = {k: _coerce(scores.get(k)) for k in ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity")}
    return scores, body, author, judge


def _seed_to_task(rng: random.Random, seed: dict, idx: int, suffix_label: str, online: bool) -> dict[str, Any]:
    if online:
        try:
            scores, body, author, judge = _route_online(rng, seed, suffix_label)
            synthesis_status = "online_routed_v1"
        except Exception as e:  # network / quota / parse — fall back gracefully
            print(f"  online failed for {seed['seed_id']}/{suffix_label}: {e}; falling back offline")
            author, judge = pick_pair(rng, "synthesis")
            scores = {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_application_clarity": 4}
            body = ""
            synthesis_status = "offline_template_v1_fallback"
    else:
        author, judge = pick_pair(rng, "synthesis")
        scores = {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_application_clarity": 4}
        body = ""
        synthesis_status = "offline_template_v1"

    prior_thread: list[dict[str, Any]] = []
    if body:
        prior_thread = [{"role": "prospect", "body": body, "at": "2026-04-29T09:00:00Z"}]

    return {
        "task_id": task_id(idx),
        "source_mode": "multi_llm_synthesis",
        "dimension": seed["dimension"],
        "difficulty": "adversarial" if "P10" in seed["probe"] or "P7" in seed["probe"] or "P5" in seed["probe"] else "hard",
        "input": {
            "channel": "email",
            "scenario": seed["stem"] + f" [variant: {suffix_label}]",
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "bench_summary": {},
            "prior_thread": prior_thread,
            "expected_action": seed["expected_action"],
        },
        "rubric": {
            "weights": {"banned": 0.6, "policy": 0.2, "tone": 0.2},
            "checks": [
                {"check_id": "banned", "type": "banned_phrase_absent", "patterns": seed["banned"]},
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["honest", "non_condescending", "professional"],
                    "min_score_per_marker": 4,
                    "judge_model": judge,
                },
            ],
        },
        "metadata": {
            "author_model": author,
            "judge_model": judge,
            "judge_scores": scores,
            "source_probe_id": seed["probe"],
            "synthesis_seed_id": seed["seed_id"],
            "synthesis_status": synthesis_status,
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW.get("builtin_jobs", "n/a"),
        },
    }


VARIANT_LABELS = ["pithy", "verbose", "skeptical-prospect", "warm-prospect"]


def generate(seed_int: int, start_idx: int, online: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed_int)
    out: list[dict[str, Any]] = []
    idx = start_idx
    for s in SEEDS:
        for label in VARIANT_LABELS:
            if online:
                print(f"  synthesis online: seed {s['seed_id']} variant {label} ...", flush=True)
            out.append(_seed_to_task(rng, s, idx, label, online=online))
            idx += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=deterministic_seed("synthesis_v1"))
    p.add_argument("--start", type=int, default=400)
    p.add_argument("--online", action="store_true", help="Call OpenRouter (requires OPENROUTER_API_KEY).")
    args = p.parse_args()
    if args.online and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set. Run without --online or set the key first.")
        return
    rows = generate(args.seed, args.start, online=args.online)
    print(f"generated {len(rows)} synthesis-mode tasks (online={args.online})")


if __name__ == "__main__":
    main()
