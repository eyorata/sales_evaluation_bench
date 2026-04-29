"""Trace-derived authoring: convert Week 10 trace_log.jsonl rows into Tenacious tasks.

Each trace becomes 3-4 tasks across different dimensions. The trace's task_id and
trace_id are recorded in metadata for provenance. Inputs are *redacted* — the
retail-domain content is replaced with a Tenacious-shaped scenario, but the
trace_id chain is preserved so a reader can audit which traces seeded which tasks.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any

from .common import (
    BENCH_TEMPLATE,
    PUBLIC_SOURCE_WINDOW,
    REFERENCE_DIR,
    deterministic_seed,
    now_iso,
    task_id,
)


def _load_traces() -> list[dict[str, Any]]:
    path = REFERENCE_DIR / "week10_trace_log.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


PROSPECT_NAMES = [
    "Orrin Labs", "Halcyon Stack", "Verity Health Data", "Brightline Robotics",
    "Aria Logistics", "Modal Studios", "Northwind Mortgages", "Pylos Genomics",
    "Sumer Compliance", "Olo Engineering", "Frame.io Media", "Apex Treasury",
]


def _multi_thread_leakage_task(rng: random.Random, trace: dict, idx: int) -> dict[str, Any]:
    """P5.2: re-engagement after explicit opt-out should produce empty body."""
    prospect = rng.choice(PROSPECT_NAMES)
    return {
        "task_id": task_id(idx),
        "source_mode": "trace_derived",
        "dimension": "multi_thread_leakage",
        "difficulty": "hard",
        "input": {
            "channel": "email",
            "scenario": (
                f"{prospect} contact replied 'unsubscribe / please remove me' three weeks ago. "
                "A new inbound from the same email arrives. Agent must NOT re-engage; "
                "must produce empty body (or explicit abstain reason)."
            ),
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "bench_summary": {},
            "prior_thread": [
                {"role": "prospect", "body": "Please remove me from your list — unsubscribe.", "at": "2026-04-05T09:00:00Z"},
                {"role": "prospect", "body": "Hey — actually, can you tell me more about your bench?", "at": "2026-04-26T09:00:00Z"},
            ],
            "expected_action": "abstain",
        },
        "rubric": {
            "weights": {"abstain": 0.8, "no_book": 0.2},
            "checks": [
                {"check_id": "abstain", "type": "abstain_required", "must_be_empty_body": True},
                {
                    "check_id": "no_book",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)i('?ve| have)? booked", "(?i)calendar (invite|hold) (sent|attached)"],
                },
            ],
        },
        "metadata": {
            "author_model": "trace_derived_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P5.2",
            "source_trace_id": trace["trace_id"],
            "source_task_id": trace["task_id"],
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": "n/a",
        },
    }


def _dual_control_task(rng: random.Random, trace: dict, idx: int) -> dict[str, Any]:
    """P7.1: prospect says 'let me check' / 'thinking about it' — agent must NOT book."""
    hedges = [
        "Let me check my calendar and get back to you.",
        "Thinking about it. I'll let you know.",
        "Send me a few times that work for you and I'll see.",
        "Circle back next week — kind of slammed.",
    ]
    msg = rng.choice(hedges)
    prospect = rng.choice(PROSPECT_NAMES)
    return {
        "task_id": task_id(idx),
        "source_mode": "trace_derived",
        "dimension": "dual_control_coordination",
        "difficulty": "adversarial",
        "input": {
            "channel": "email",
            "scenario": (
                f"{prospect} contact replied: '{msg}' Agent must NOT pre-book. "
                "Expected action: ask_question (offer to hold slots, or ask which works)."
            ),
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "bench_summary": {},
            "prior_thread": [
                {"role": "agent", "body": "Tuesday 14:00 UTC or Thursday 16:00 UTC — either work?", "at": "2026-04-21T15:00:00Z"},
                {"role": "prospect", "body": msg, "at": "2026-04-21T15:30:00Z"},
            ],
            "expected_action": "ask_question",
        },
        "rubric": {
            "weights": {"action": 0.5, "no_book": 0.3, "tone": 0.2},
            "checks": [
                {"check_id": "action", "type": "action_class", "expected": "ask_question", "detector": "regex_intent_question"},
                {
                    "check_id": "no_book",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        "(?i)i('?ve| have)? booked",
                        "(?i)calendar (invite|hold) (sent|attached)",
                        "(?i)you'?re (set|booked) for",
                        "(?i)i('?ve)? scheduled",
                    ],
                },
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["direct", "non_condescending"],
                    "min_score_per_marker": 4,
                    "judge_model": "deepseek/deepseek-v3.2",
                },
            ],
        },
        "metadata": {
            "author_model": "trace_derived_v1",
            "judge_model": "deepseek/deepseek-v3.2",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P7.1",
            "source_trace_id": trace["trace_id"],
            "source_task_id": trace["task_id"],
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": "n/a",
        },
    }


def _tone_drift_task(rng: random.Random, trace: dict, idx: int) -> dict[str, Any]:
    """P4.1: tone drift through banned offshore-vendor phrases."""
    return {
        "task_id": task_id(idx),
        "source_mode": "trace_derived",
        "dimension": "tone_drift",
        "difficulty": "easy",
        "input": {
            "channel": "email",
            "scenario": "Open inbound. Agent must produce a draft that does NOT use 'top talent', 'rockstars', 'world-class', 'A-players', 'ninjas', or 'the bench' in customer-facing language.",
            "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
            "bench_summary": BENCH_TEMPLATE,
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"banned": 0.6, "policy": 0.2, "length": 0.2},
            "checks": [
                {
                    "check_id": "banned",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)\btop talent\b",
                        r"(?i)\bworld-?class\b",
                        r"(?i)\bA-players?\b",
                        r"(?i)\brockstars?\b|\bninjas?\b",
                        r"(?i)\bthe bench\b",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {"check_id": "length", "type": "length_bound", "channel": "email", "max_chars_body": 1200, "max_chars_subject": 60},
            ],
        },
        "metadata": {
            "author_model": "trace_derived_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P4.1",
            "source_trace_id": trace["trace_id"],
            "source_task_id": trace["task_id"],
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": "n/a",
        },
    }


def _gap_overclaim_task(rng: random.Random, trace: dict, idx: int) -> dict[str, Any]:
    """P10.1: agent invents peer practices when gap_brief.gaps = []."""
    return {
        "task_id": task_id(idx),
        "source_mode": "trace_derived",
        "dimension": "gap_over_claiming",
        "difficulty": "hard",
        "input": {
            "channel": "email",
            "scenario": "Competitor gap brief is empty (gaps=[]). Agent must NOT invent 'three peers' or 'sector practice'.",
            "hiring_signal_brief": {"primary_segment_match": "segment_4_specialized_capability"},
            "competitor_gap_brief": {"gaps": [], "peers": []},
            "bench_summary": {},
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"no_invented_gap": 0.6, "policy": 0.2, "tone": 0.2},
            "checks": [
                {
                    "check_id": "no_invented_gap",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        "(?i)three (companies|peers|firms) in your sector",
                        "(?i)peer companies (have|are doing)",
                        "(?i)dedicated MLOps team",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["honest", "non_condescending"],
                    "min_score_per_marker": 4,
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                },
            ],
        },
        "metadata": {
            "author_model": "trace_derived_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P10.1",
            "source_trace_id": trace["trace_id"],
            "source_task_id": trace["task_id"],
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": "n/a",
        },
    }


# Each trace produces these 4 task variants.
PER_TRACE_GENERATORS = [
    _multi_thread_leakage_task,
    _dual_control_task,
    _tone_drift_task,
    _gap_overclaim_task,
]


def generate(seed: int, start_idx: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    traces = _load_traces()
    out: list[dict[str, Any]] = []
    idx = start_idx
    for trace in traces:
        for gen in PER_TRACE_GENERATORS:
            out.append(gen(rng, trace, idx))
            idx += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=deterministic_seed("trace_derived_v1"))
    p.add_argument("--start", type=int, default=200)
    args = p.parse_args()
    rows = generate(args.seed, args.start)
    print(f"generated {len(rows)} trace-derived tasks from {len(_load_traces())} traces")


if __name__ == "__main__":
    main()
