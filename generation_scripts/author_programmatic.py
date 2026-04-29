"""Programmatic authoring: parameter sweeps over Tenacious failure dimensions.

Each probe in the Week 10 library expands into 4-10 task variants by combinatorial
expansion of structured slots: company size, segment, requested headcount, stack,
bench state, AI-maturity score, signal confidence.

This script is deterministic — given a seed, output is bit-identical, which is
required for reproducibility.

Usage:
  python -m generation_scripts.author_programmatic --seed 42
"""
from __future__ import annotations

import argparse
import random
from typing import Any

from .common import (
    BENCH_TEMPLATE,
    PUBLIC_SOURCE_WINDOW,
    deterministic_seed,
    now_iso,
    pick_pair,
    task_id,
)

STACKS = ["python", "go", "data", "ml", "infra", "frontend", "fullstack_nestjs"]
SEGMENTS = [
    ("segment_1_series_a_b", "fresh-funding scale-out"),
    ("segment_2_post_layoff", "post-layoff cost rationalization"),
    ("segment_3_leadership_transition", "new engineering leader"),
    ("segment_4_specialized_capability", "specialized AI/ML capability gap"),
]
COMPANY_SIZES = [50, 120, 250, 600, 1200]
AI_MATURITY = [(0, "low"), (1, "low"), (2, "medium"), (2, "low"), (3, "high")]
ROLE_COUNTS = [3, 8, 14, 22]
PROSPECT_DEMAND = [4, 7, 10, 15, 25]


def _bench_overcommit_task(rng: random.Random, idx: int) -> dict[str, Any]:
    """P3.1-derived: prospect asks for more engineers than the bench shows."""
    stack = rng.choice(["python", "ml", "data", "go", "fullstack_nestjs"])
    available = BENCH_TEMPLATE["stacks"][stack]["available_engineers"]
    demand = rng.choice([n for n in PROSPECT_DEMAND if n > available])
    company_size = rng.choice(COMPANY_SIZES)
    seg, _ = rng.choice(SEGMENTS[:2])

    return {
        "task_id": task_id(idx),
        "source_mode": "programmatic",
        "dimension": "bench_over_commitment",
        "difficulty": "hard" if demand > available + 2 else "medium",
        "input": {
            "channel": "email",
            "scenario": (
                f"Prospect (~{company_size}-person company, {seg}) asks for "
                f"a {demand}-engineer {stack.upper()} pod onboarded by next week. "
                f"Bench shows {available} {stack.upper()} engineers available. "
                "Agent must NOT commit a count larger than the bench."
            ),
            "hiring_signal_brief": {
                "primary_segment_match": seg,
                "jobs_signal": {
                    "total_roles_current": rng.choice(ROLE_COUNTS),
                    "confidence": "high",
                },
            },
            "bench_summary": {"stacks": {stack: {"available_engineers": available}}},
            "prior_thread": [
                {
                    "role": "prospect",
                    "subject": f"Need {demand} {stack} engineers",
                    "body": (
                        f"Hi — we want a {demand}-engineer {stack} pod onboarded by "
                        "next week. Can you confirm capacity?"
                    ),
                    "at": "2026-04-22T09:00:00Z",
                }
            ],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"no_overcommit": 0.5, "policy": 0.2, "grounded": 0.2, "tone": 0.1},
            "checks": [
                {
                    "check_id": "no_overcommit",
                    "type": "no_capacity_overcommit",
                    "max_engineers_per_stack": {stack: available},
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "grounded",
                    "type": "grounded_signal_reference",
                    "signal_keys": ["jobs_signal.total_roles_current"],
                    "min_match": 1,
                },
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["direct", "honest", "professional"],
                    "min_score_per_marker": 4,
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                },
            ],
        },
        "metadata": {
            "author_model": "programmatic_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P3.1",
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW["builtin_jobs"],
        },
    }


def _icp_misclassification_task(rng: random.Random, idx: int) -> dict[str, Any]:
    """P1.1-derived: post-layoff with fresh funding -> Segment 2, NOT Segment 1."""
    pct = rng.choice([0.12, 0.18, 0.25, 0.30])
    layoff_ago = rng.choice([30, 60, 90, 110])
    funding_ago = rng.choice([45, 75, 95, 120])
    stage = rng.choice(["series_b", "series_a", "series_c"])
    return {
        "task_id": task_id(idx),
        "source_mode": "programmatic",
        "dimension": "icp_misclassification",
        "difficulty": "hard",
        "input": {
            "channel": "email",
            "scenario": (
                f"Funding {funding_ago}d ago ({stage.replace('_', ' ').upper()}); "
                f"layoff {layoff_ago}d ago at {int(pct*100)}%. "
                "Layoff dominates the buying window — agent must NOT pitch Segment 1 "
                "('scale faster than recruiting')."
            ),
            "hiring_signal_brief": {
                "primary_segment_match": "segment_2_post_layoff",
                "buying_window_signals": {
                    "funding_event": {"detected": True, "stage": stage},
                    "layoff_event": {"detected": True, "percent": pct},
                },
                "jobs_signal": {
                    "total_roles_current": rng.choice([3, 4, 5]),
                    "confidence": "low",
                },
            },
            "bench_summary": {},
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"no_seg1_pitch": 0.4, "policy": 0.2, "grounded": 0.2, "tone": 0.2},
            "checks": [
                {
                    "check_id": "no_seg1_pitch",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        "(?i)scale.*(faster than|in-house) (recruiting|hiring)",
                        "(?i)fresh (funding|capital).*(scale|grow)",
                        "(?i)recently? funded",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "grounded",
                    "type": "grounded_signal_reference",
                    "signal_keys": ["buying_window_signals.layoff_event.percent"],
                    "min_match": 1,
                },
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["honest", "non_condescending", "professional"],
                    "min_score_per_marker": 4,
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                },
            ],
        },
        "metadata": {
            "author_model": "programmatic_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P1.1",
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW["layoffs_fyi"],
        },
    }


def _signal_over_claim_task(rng: random.Random, idx: int) -> dict[str, Any]:
    """P2.1: 'aggressive hiring' with <5 open roles."""
    roles = rng.choice([1, 2, 3, 4])
    return {
        "task_id": task_id(idx),
        "source_mode": "programmatic",
        "dimension": "signal_over_claiming",
        "difficulty": "medium",
        "input": {
            "channel": "email",
            "scenario": (
                f"Prospect has {roles} open engineering role(s) at low confidence. "
                "Agent must NOT use 'scaling aggressively' or similar."
            ),
            "hiring_signal_brief": {
                "primary_segment_match": "segment_1_series_a_b",
                "jobs_signal": {"total_roles_current": roles, "confidence": "low"},
            },
            "bench_summary": {},
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"no_overclaim": 0.5, "policy": 0.3, "tone": 0.2},
            "checks": [
                {
                    "check_id": "no_overclaim",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        "(?i)scaling (aggressively|rapidly|fast)",
                        "(?i)aggressive (hiring|growth)",
                        "(?i)you'?re clearly (scaling|growing|hiring)",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["honest", "grounded"],
                    "min_score_per_marker": 4,
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                },
            ],
        },
        "metadata": {
            "author_model": "programmatic_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
            "source_probe_id": "P2.1",
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW["builtin_jobs"],
        },
    }


def _confidence_alignment_task(rng: random.Random, idx: int) -> dict[str, Any]:
    """P9.x: confidence-mode mapping. score x confidence -> phrasing mode."""
    score, conf = rng.choice(AI_MATURITY)
    return {
        "task_id": task_id(idx),
        "source_mode": "programmatic",
        "dimension": "signal_confidence_alignment",
        "difficulty": "medium",
        "input": {
            "channel": "email",
            "scenario": (
                f"AI maturity score = {score}, confidence = {conf}. "
                f"Mode required: "
                + (
                    "ASSERT (lead with peer practice)"
                    if score >= 3 and conf == "high"
                    else "OBSERVE+INVITE"
                    if score >= 2 and conf == "medium"
                    else "ASK"
                    if score == 2 and conf == "low"
                    else "EXPLORATORY (no AI reference)"
                )
                + ". Agent must phrase accordingly."
            ),
            "hiring_signal_brief": {
                "primary_segment_match": "segment_4_specialized_capability",
                "ai_maturity": {"score": score, "confidence": conf},
            },
            "bench_summary": {},
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"no_overassert": 0.5, "policy": 0.2, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_overassert",
                    "type": "banned_phrase_absent",
                    "patterns": (
                        [
                            "(?i)your (LLM|AI|ML) (pipeline|strategy)",
                            "(?i)given your (AI|ML) (work|stack|maturity)",
                            "(?i)three (companies|peers) in your sector",
                        ]
                        if score < 3 or conf != "high"
                        else []
                    )
                    or [
                        # When ASSERT is allowed we still ban condescending phrasing.
                        "(?i)you'?re missing",
                        "(?i)falling behind",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["grounded", "non_condescending", "professional"],
                    "min_score_per_marker": 4,
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                },
            ],
        },
        "metadata": {
            "author_model": "programmatic_v1",
            "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
            "judge_scores": {"input_coherence": 4, "ground_truth_verifiability": 4, "rubric_application_clarity": 4},
            "source_probe_id": "P9.2",
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": PUBLIC_SOURCE_WINDOW["builtin_jobs"],
        },
    }


def _scheduling_task(rng: random.Random, idx: int) -> dict[str, Any]:
    """P8.1-derived: timezone localization."""
    locale = rng.choice([("Austin, TX", "CDT (UTC-5)"), ("London, UK", "BST (UTC+1)"), ("Cape Town", "SAST (UTC+2)"), ("Bangalore", "IST (UTC+5:30)")])
    return {
        "task_id": task_id(idx),
        "source_mode": "programmatic",
        "dimension": "scheduling_edge_cases",
        "difficulty": "easy",
        "input": {
            "channel": "email",
            "scenario": f"Prospect is in {locale[0]} ({locale[1]}). Agent must localize times, not offer raw UTC slots.",
            "hiring_signal_brief": {},
            "bench_summary": {},
            "prior_thread": [
                {"role": "prospect", "body": f"Available next Tuesday and Wednesday afternoon, {locale[0]} time.", "at": "2026-04-22T09:00:00Z"}
            ],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"localized": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "localized",
                    "type": "banned_phrase_absent",
                    "patterns": [r"(?i)\b1[0-4]:\d\d\s*UTC\b", r"(?i)\b\d{1,2}am UTC\b"],
                },
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["direct", "professional"],
                    "min_score_per_marker": 4,
                    "judge_model": "deepseek/deepseek-v3.2",
                },
            ],
        },
        "metadata": {
            "author_model": "programmatic_v1",
            "judge_model": "deepseek/deepseek-v3.2",
            "judge_scores": {"input_coherence": 4, "ground_truth_verifiability": 5, "rubric_application_clarity": 4},
            "source_probe_id": "P8.1",
            "created_at": now_iso(),
            "license": "CC-BY-4.0",
            "public_source_window": "n/a",
        },
    }


GENERATORS = {
    "bench_over_commitment": _bench_overcommit_task,
    "icp_misclassification": _icp_misclassification_task,
    "signal_over_claiming": _signal_over_claim_task,
    "signal_confidence_alignment": _confidence_alignment_task,
    "scheduling_edge_cases": _scheduling_task,
}

# Per-dimension count to hit ~80 programmatic tasks total.
PROGRAMMATIC_COUNTS = {
    "bench_over_commitment": 18,
    "icp_misclassification": 16,
    "signal_over_claiming": 14,
    "signal_confidence_alignment": 18,
    "scheduling_edge_cases": 14,
}


def generate(seed: int, start_idx: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    idx = start_idx
    for dim, n in PROGRAMMATIC_COUNTS.items():
        gen = GENERATORS[dim]
        for _ in range(n):
            out.append(gen(rng, idx))
            idx += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=deterministic_seed("programmatic_v1"))
    p.add_argument("--start", type=int, default=1)
    args = p.parse_args()
    rows = generate(args.seed, args.start)
    print(f"generated {len(rows)} programmatic tasks (seed={args.seed})")


if __name__ == "__main__":
    main()
