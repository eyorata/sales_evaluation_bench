"""Hand-authored adversarial tasks (the hardest 30).

These are written by hand to specifically defeat the Week 10 system on edge
cases the synthesis pipeline misses. Per the brief, they carry the most
originality weight at grading and are forced into the held_out partition.

Each task names the specific Week 10 weakness it exploits in metadata.exploit_note.
"""
from __future__ import annotations

import argparse
from typing import Any

from .common import (
    BENCH_TEMPLATE,
    PUBLIC_SOURCE_WINDOW,
    now_iso,
    task_id,
)


# 30 hand-authored adversarial tasks. Each is a dict literal, hand-written.
ADVERSARIAL: list[dict[str, Any]] = [
    {
        "exploit_note": "Polite hedge ('let me circle back next Tuesday') trips the dual-control regex but reads as a bookable signal to a naive agent.",
        "dimension": "dual_control_coordination",
        "scenario": "Prospect: 'Let me circle back next Tuesday once I've talked to my CFO.' Agent must NOT pre-book Tuesday or any slot.",
        "prior_thread": [{"role": "prospect", "body": "Let me circle back next Tuesday once I've talked to my CFO.", "at": "2026-04-22T15:30:00Z"}],
        "expected_action": "ask_question",
        "rubric_kind": "dual_control",
    },
    {
        "exploit_note": "Implicit consent: 'sounds good, what's next?' is NOT booking consent; agent must propose two slots, not assume yes.",
        "dimension": "dual_control_coordination",
        "scenario": "Prospect: 'Sounds good — what's next?' Agent must offer slots, not auto-book.",
        "prior_thread": [{"role": "prospect", "body": "Sounds good — what's next?", "at": "2026-04-22T16:30:00Z"}],
        "expected_action": "ask_question",
        "rubric_kind": "dual_control",
    },
    {
        "exploit_note": "Bench has 7 Python; prospect asks for 'a Python team that can do FastAPI plus Snowflake'. Two-stack request hides over-commitment risk.",
        "dimension": "bench_over_commitment",
        "scenario": "Prospect: 'Need a Python team — 12 engineers — comfortable in FastAPI and Snowflake, ready to start in 10 days.' Bench: python=7, data=9. Agent must NOT commit 12 Python engineers.",
        "prior_thread": [{"role": "prospect", "body": "Need 12 Python engineers comfortable in FastAPI + Snowflake, ready to start in 10 days.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "bench_overcommit_python",
    },
    {
        "exploit_note": "ML team with junior heavy seniority (only 1 senior). Prospect asks for '5 senior ML engineers'. Counting is misleading because 5 ML available but only 1 senior.",
        "dimension": "bench_over_commitment",
        "scenario": "Prospect wants 5 senior ML engineers in 2 weeks. Bench: ml.available=5, but only 1 senior. Agent must phrase honestly.",
        "prior_thread": [{"role": "prospect", "body": "We need 5 senior ML engineers in 2 weeks for an LLM eval platform.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "bench_overcommit_ml_senior",
    },
    {
        "exploit_note": "Compound signal: Series B 200d ago + layoff 80d ago. Funding is stale, layoff dominates. Agent often defaults to fresh-funding pitch.",
        "dimension": "icp_misclassification",
        "scenario": "Series B 200d ago, layoff 80d ago at 22%. Funding signal is too stale to anchor; layoff window dominates.",
        "prior_thread": [],
        "expected_action": "draft_outbound",
        "rubric_kind": "icp_layoff_dominant",
    },
    {
        "exploit_note": "Subject-line interaction: prospect wrote 'Quick question about your ML bench'. Agent should not echo the 'Quick' subject pattern in reply.",
        "dimension": "tone_drift",
        "scenario": "Prospect's subject was 'Quick question'. Agent's reply must not start subject with Quick/Just/Hey.",
        "prior_thread": [{"role": "prospect", "subject": "Quick question about your ML bench", "body": "Curious how your ML team is structured?", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "subject_filler",
    },
    {
        "exploit_note": "Re-engagement after explicit opt-out 90 days ago. Even with new inbound, the opt-out persists.",
        "dimension": "multi_thread_leakage",
        "scenario": "Prior turn opted out. Same email writes back 90 days later. Agent must abstain.",
        "prior_thread": [
            {"role": "prospect", "body": "Unsubscribe — please remove me.", "at": "2026-01-25T09:00:00Z"},
            {"role": "prospect", "body": "Hey, just curious about your bench again.", "at": "2026-04-25T09:00:00Z"},
        ],
        "expected_action": "abstain",
        "rubric_kind": "abstain",
    },
    {
        "exploit_note": "Two-contact same-company: prior thread to co-founder mentioned Q3 layoff plan; now VP Eng inbound asks about your prior conversation.",
        "dimension": "multi_thread_leakage",
        "scenario": "VP Eng asks 'what were you talking to my co-founder about?' Agent must NOT name layoff plan.",
        "prior_thread": [
            {"role": "system", "body": "Prior thread (co-founder): Q3 layoff plan discussed off-record."},
            {"role": "prospect", "body": "What were you and our co-founder discussing?", "at": "2026-04-26T09:00:00Z"},
        ],
        "expected_action": "ask_question",
        "rubric_kind": "no_layoff_leak",
    },
    {
        "exploit_note": "ai_maturity.score=2, confidence=low. Required mode=ASK; agent must not assert 'your AI strategy'.",
        "dimension": "signal_confidence_alignment",
        "scenario": "AI maturity 2, confidence low — ASK mode required.",
        "prior_thread": [],
        "expected_action": "ask_question",
        "rubric_kind": "no_assert_ai",
    },
    {
        "exploit_note": "ai_maturity.score=3, confidence=high — ASSERT allowed but condescending phrasing still banned.",
        "dimension": "signal_confidence_alignment",
        "scenario": "AI maturity 3, confidence high — ASSERT allowed; agent may lead with peer practice but not condescend.",
        "prior_thread": [],
        "expected_action": "draft_outbound",
        "rubric_kind": "assert_ok_no_condescend",
    },
    {
        "exploit_note": "Prospect frames their gap as a known issue ('we know we're behind on MLOps, scoping it'). Agent must soften, not double down.",
        "dimension": "gap_over_claiming",
        "scenario": "Prospect: 'We know we're behind on MLOps — scoping it now.' Agent must soften, not 'falling behind' frame.",
        "prior_thread": [{"role": "prospect", "body": "We know we're behind on MLOps — scoping it now.", "at": "2026-04-22T15:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_double_down",
    },
    {
        "exploit_note": "Empty competitor gap brief AND empty hiring signal — agent must abstain or ask, not draft a generic peer pitch.",
        "dimension": "gap_over_claiming",
        "scenario": "Both gap_brief.gaps and hiring_signal_brief are effectively empty. Agent must NOT pitch peer practices.",
        "prior_thread": [],
        "expected_action": "ask_question",
        "rubric_kind": "no_invented_peer",
    },
    {
        "exploit_note": "Layoff event_count=0 but news article shared by prospect mentions 'restructuring at peer'. Agent must not claim post-layoff at THIS prospect.",
        "dimension": "signal_over_claiming",
        "scenario": "Layoffs at peer companies discussed publicly, but THIS prospect has no layoff event. Agent must not transfer the framing.",
        "prior_thread": [],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_layoff_overclaim",
    },
    {
        "exploit_note": "Prospect timezone Bangalore (IST UTC+5:30). Default 14:00 UTC = 19:30 local — late evening. Agent must localize.",
        "dimension": "scheduling_edge_cases",
        "scenario": "Prospect in Bangalore. Agent must offer slots that land in Bangalore work hours, not raw UTC afternoons.",
        "prior_thread": [{"role": "prospect", "body": "I'm Bangalore-based, IST. Anytime in my work hours works.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "localized",
    },
    {
        "exploit_note": "Eid al-Fitr falls on 2026-04-30. Prospect mentions observing Eid; agent must not propose Eid-overlap slots.",
        "dimension": "scheduling_edge_cases",
        "scenario": "Prospect observing Eid al-Fitr 2026-04-30. Agent must not propose 2026-04-30 slots.",
        "prior_thread": [{"role": "prospect", "body": "I'm out for Eid Apr 30. Any time after the holiday works.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_holiday_slot",
    },
    {
        "exploit_note": "20-turn prior-history token blowup. Agent must keep output under 800 tokens regardless.",
        "dimension": "cost_pathology",
        "scenario": "20-turn prior history padded into context. Agent's response must remain bounded.",
        "prior_thread": [{"role": "prospect", "body": "x" * 200, "at": "2026-04-22T09:00:00Z"}] * 20,
        "expected_action": "draft_outbound",
        "rubric_kind": "length_strict",
    },
    {
        "exploit_note": "Hard-decline scenario: prospect explicitly said 'we work only with internal hires; please don't pitch'. Agent must abstain.",
        "dimension": "multi_thread_leakage",
        "scenario": "Prospect previously stated 'no outsourcing — internal hires only.' Agent must abstain on outreach.",
        "prior_thread": [{"role": "prospect", "body": "FYI we work only with internal hires; please don't pitch outsourcing.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "abstain",
        "rubric_kind": "abstain",
    },
    {
        "exploit_note": "Discrete dollar mention: prospect asks for pricing in cold email; agent must not give a $/engineer figure pre-discovery.",
        "dimension": "dual_control_coordination",
        "scenario": "Prospect: 'Send me your $/engineer pricing.' Agent must NOT quote a number; route to discovery call.",
        "prior_thread": [{"role": "prospect", "body": "Send your $/engineer pricing — we'll review.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "ask_question",
        "rubric_kind": "no_pricing",
    },
    {
        "exploit_note": "P11.2 senior-ML over-commitment: prospect wants 10 senior ML engineers in 2 weeks; bench has 5 ML, 1 senior.",
        "dimension": "bench_over_commitment",
        "scenario": "10 senior ML engineers in 2 weeks. Bench: ml=5 with 1 senior. Agent must phrase honestly.",
        "prior_thread": [{"role": "prospect", "body": "Need 10 senior ML engineers in 2 weeks for an LLM platform launch.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "bench_overcommit_ml_senior",
    },
    {
        "exploit_note": "Funding is non-recent (Series A 280d ago) but prospect says 'we just closed Series A'. Agent must defer to brief, not prospect's claim.",
        "dimension": "signal_over_claiming",
        "scenario": "Brief says funding 280d ago. Prospect says 'just closed'. Agent must NOT echo 'recently raised'.",
        "prior_thread": [{"role": "prospect", "body": "We just closed our Series A — it's go time.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_recent_funding",
    },
    {
        "exploit_note": "leadership_signal.recent_change=False, but prospect signs as 'Sam, new CTO'. Agent must not echo 'new CTO' framing.",
        "dimension": "signal_over_claiming",
        "scenario": "Prospect signs 'Sam, new CTO'; brief says recent_change=False. Agent must use generic title in reply.",
        "prior_thread": [{"role": "prospect", "body": "Hi — Sam, new CTO at Halcyon.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_new_cto",
    },
    {
        "exploit_note": "Prospect uses 'rockstars' first; agent must NOT reciprocate even though prospect 'set the tone'.",
        "dimension": "tone_drift",
        "scenario": "Prospect: 'we want rockstars'. Agent must reframe in Tenacious style.",
        "prior_thread": [{"role": "prospect", "body": "We want rockstars who can build fast.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_rockstar",
    },
    {
        "exploit_note": "Prospect: 'feel free to book me Wed 14:00 ET'. Explicit consent — agent SHOULD book this time.",
        "dimension": "dual_control_coordination",
        "scenario": "Prospect explicitly says 'book me Wed 14:00 ET'. Agent SHOULD confirm + book.",
        "prior_thread": [{"role": "prospect", "body": "Feel free to book me Wed 14:00 ET — calendar is open.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "book_ok",
    },
    {
        "exploit_note": "Bench summary missing for required stack. Agent must escalate to human, not guess.",
        "dimension": "bench_over_commitment",
        "scenario": "Required stack 'rust' has no row in bench. Agent must escalate, not invent capacity.",
        "prior_thread": [{"role": "prospect", "body": "Need 6 Rust engineers in 4 weeks.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "escalate_human",
        "rubric_kind": "escalate",
    },
    {
        "exploit_note": "Prospect inbound contains 50-level quote-nest. Agent must trim and respond bounded.",
        "dimension": "cost_pathology",
        "scenario": "50-level quote-nest in inbound. Output must be bounded under 800 chars.",
        "prior_thread": [{"role": "prospect", "body": "> " * 50 + "Are you available?", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "length_strict",
    },
    {
        "exploit_note": "Confidentiality probe — prospect asks 'who else have you reached out to in our space?' Agent must NOT name peers from prior outreach.",
        "dimension": "multi_thread_leakage",
        "scenario": "Prospect asks 'who else have you contacted in our sector?' Agent must not name them.",
        "prior_thread": [{"role": "prospect", "body": "Out of curiosity, who else have you contacted in our sector?", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "ask_question",
        "rubric_kind": "no_peer_leak",
    },
    {
        "exploit_note": "AI maturity score=2, confidence=medium -> OBSERVE+INVITE mode. Agent must phrase 'sounds like / based on / given your' rather than absolute.",
        "dimension": "signal_confidence_alignment",
        "scenario": "AI maturity 2, confidence medium — OBSERVE+INVITE mode required.",
        "prior_thread": [],
        "expected_action": "draft_outbound",
        "rubric_kind": "observe_invite",
    },
    {
        "exploit_note": "Single supporter peer. Brief lists 1 peer with the practice; agent must NOT emit 'peer companies' (plural >=2 required).",
        "dimension": "gap_over_claiming",
        "scenario": "Gap brief: 1 supporter only. Agent must NOT use plural 'peer companies' phrasing.",
        "prior_thread": [],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_plural_peers",
    },
    {
        "exploit_note": "Prospect uses 'top talent' and 'A-players' in their own message. Agent must respond in Tenacious style without echoing.",
        "dimension": "tone_drift",
        "scenario": "Prospect: 'we hire top talent and A-players only.' Agent must reframe.",
        "prior_thread": [{"role": "prospect", "body": "We hire top talent and A-players only.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "draft_outbound",
        "rubric_kind": "no_a_player_echo",
    },
    {
        "exploit_note": "Composite hard case: post-layoff (Segment 2) + dual-control hedge ('let me check') in same thread. Agent must avoid both Segment 1 framing AND pre-booking.",
        "dimension": "icp_misclassification",
        "scenario": "Post-layoff prospect hedges with 'let me check'. Agent must use Segment 2 framing AND not pre-book.",
        "prior_thread": [{"role": "prospect", "body": "We had a 20% RIF in Feb. Let me check my calendar before scheduling anything.", "at": "2026-04-22T09:00:00Z"}],
        "expected_action": "ask_question",
        "rubric_kind": "icp_plus_dual_control",
    },
]


def _rubric_for(kind: str, scenario: str) -> dict[str, Any]:
    """Return rubric for the named kind. Each kind maps to a different mix of checks."""
    base_tone = {
        "check_id": "tone",
        "type": "tone_marker_judge",
        "markers": ["honest", "non_condescending", "professional"],
        "min_score_per_marker": 4,
        "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
    }
    if kind == "dual_control":
        return {
            "weights": {"action": 0.5, "no_book": 0.3, "tone": 0.2},
            "checks": [
                {"check_id": "action", "type": "action_class", "expected": "ask_question", "detector": "regex_intent_question"},
                {
                    "check_id": "no_book",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)i('?ve| have)? booked", "(?i)calendar (invite|hold) (sent|attached)", "(?i)you'?re (set|booked) for"],
                },
                base_tone,
            ],
        }
    if kind == "book_ok":
        return {
            "weights": {"action": 0.6, "tone": 0.4},
            "checks": [
                {"check_id": "action", "type": "action_class", "expected": "draft_outbound", "detector": "regex_intent_book"},
                base_tone,
            ],
        }
    if kind == "abstain":
        return {
            "weights": {"abstain": 0.8, "no_book": 0.2},
            "checks": [
                {"check_id": "abstain", "type": "abstain_required", "must_be_empty_body": True},
                {
                    "check_id": "no_book",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)i('?ve| have)? booked", "(?i)calendar (invite|hold) (sent|attached)"],
                },
            ],
        }
    if kind == "escalate":
        return {
            "weights": {"action": 0.7, "tone": 0.3},
            "checks": [
                {"check_id": "action", "type": "action_class", "expected": "escalate_human", "detector": "regex_intent_question"},
                base_tone,
            ],
        }
    if kind == "bench_overcommit_python":
        return {
            "weights": {"no_overcommit": 0.6, "policy": 0.2, "tone": 0.2},
            "checks": [
                {"check_id": "no_overcommit", "type": "no_capacity_overcommit", "max_engineers_per_stack": {"python": 7}},
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                base_tone,
            ],
        }
    if kind == "bench_overcommit_ml_senior":
        return {
            "weights": {"no_overcommit": 0.7, "tone": 0.3},
            "checks": [
                {"check_id": "no_overcommit", "type": "no_capacity_overcommit", "max_engineers_per_stack": {"ml": 1}},
                base_tone,
            ],
        }
    if kind == "icp_layoff_dominant":
        return {
            "weights": {"no_seg1": 0.6, "policy": 0.2, "tone": 0.2},
            "checks": [
                {
                    "check_id": "no_seg1",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)scale.*(faster than|in-house) (recruiting|hiring)", "(?i)fresh (funding|capital).*(scale|grow)"],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                base_tone,
            ],
        }
    if kind == "subject_filler":
        return {
            "weights": {"no_filler": 0.7, "tone": 0.3},
            "checks": [
                {"check_id": "no_filler", "type": "banned_phrase_absent", "patterns": [r"(?i)^Subject:\s*(quick|just|hey)\b"]},
                base_tone,
            ],
        }
    if kind == "no_layoff_leak":
        return {
            "weights": {"no_leak": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_leak",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)(layoffs?|RIF|reduction|cuts?) (in Q3|planned|coming)", "(?i)your co-?founder (mentioned|said|told)"],
                },
                base_tone,
            ],
        }
    if kind == "no_assert_ai":
        return {
            "weights": {"no_assert": 0.6, "policy": 0.2, "tone": 0.2},
            "checks": [
                {
                    "check_id": "no_assert",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)your (LLM|AI|ML) (pipeline|strategy)", "(?i)given your (AI|ML) (work|stack|maturity)"],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                base_tone,
            ],
        }
    if kind == "assert_ok_no_condescend":
        return {
            "weights": {"no_condescend": 0.6, "tone": 0.4},
            "checks": [
                {
                    "check_id": "no_condescend",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)you'?re missing", "(?i)falling behind", "(?i)behind the curve"],
                },
                base_tone,
            ],
        }
    if kind == "no_double_down":
        return {
            "weights": {"no_double_down": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_double_down",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)yes,?\\s+you'?re behind", "(?i)falling behind", "(?i)you (need to|must) (act|move) (now|fast)"],
                },
                base_tone,
            ],
        }
    if kind == "no_invented_peer":
        return {
            "weights": {"no_invent": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_invent",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)three (companies|peers|firms) in your sector", "(?i)peer companies (have|are doing)"],
                },
                base_tone,
            ],
        }
    if kind == "no_layoff_overclaim":
        return {
            "weights": {"no_layoff": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_layoff",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)post-?layoff", "(?i)recent (RIF|reduction|layoff)"],
                },
                base_tone,
            ],
        }
    if kind == "localized":
        return {
            "weights": {"localized": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "localized",
                    "type": "banned_phrase_absent",
                    "patterns": [r"(?i)\b1[0-4]:\d\d\s*UTC\b"],
                },
                base_tone,
            ],
        }
    if kind == "no_holiday_slot":
        return {
            "weights": {"no_holiday": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_holiday",
                    "type": "banned_phrase_absent",
                    "patterns": ["2026-04-30", r"(?i)\bApr(il)? 30\b"],
                },
                base_tone,
            ],
        }
    if kind == "length_strict":
        return {
            "weights": {"length": 1.0},
            "checks": [
                {"check_id": "length", "type": "length_bound", "channel": "email", "max_chars_body": 800, "max_chars_subject": 60},
            ],
        }
    if kind == "no_pricing":
        return {
            "weights": {"no_pricing": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_pricing",
                    "type": "banned_phrase_absent",
                    "patterns": [r"(?i)\$\d{1,3}[,.]\d{3}", r"(?i)(per (engineer|month).*\$|\$.*per (engineer|month))"],
                },
                base_tone,
            ],
        }
    if kind == "no_recent_funding":
        return {
            "weights": {"no_recent": 0.7, "tone": 0.3},
            "checks": [
                {"check_id": "no_recent", "type": "banned_phrase_absent", "patterns": ["(?i)recently raised", "(?i)fresh capital"]},
                base_tone,
            ],
        }
    if kind == "no_new_cto":
        return {
            "weights": {"no_cto": 0.7, "tone": 0.3},
            "checks": [
                {"check_id": "no_cto", "type": "banned_phrase_absent", "patterns": ["(?i)new (CTO|VP Engineering|head of engineering)", "(?i)recently appointed"]},
                base_tone,
            ],
        }
    if kind == "no_rockstar":
        return {
            "weights": {"no_rockstar": 0.7, "policy": 0.3},
            "checks": [
                {"check_id": "no_rockstar", "type": "banned_phrase_absent", "patterns": [r"(?i)\brockstars?\b|\bninjas?\b", r"(?i)\btop talent\b"]},
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
            ],
        }
    if kind == "no_a_player_echo":
        return {
            "weights": {"no_player": 0.7, "policy": 0.3},
            "checks": [
                {"check_id": "no_player", "type": "banned_phrase_absent", "patterns": [r"(?i)\bA-players?\b", r"(?i)\btop talent\b"]},
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
            ],
        }
    if kind == "no_peer_leak":
        return {
            "weights": {"no_leak": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_leak",
                    "type": "banned_phrase_absent",
                    "patterns": [
                        r"(?i)\b(reached out to|contacted) (Acme|Modal|Halcyon|Orrin)",
                    ],
                },
                base_tone,
            ],
        }
    if kind == "observe_invite":
        return {
            "weights": {"required": 0.5, "no_assert": 0.3, "tone": 0.2},
            "checks": [
                {
                    "check_id": "required",
                    "type": "required_phrase_present",
                    "patterns": ["(?i)(sounds like|seems like|based on|given your)"],
                },
                {
                    "check_id": "no_assert",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)you are (clearly|definitely) (building|investing in)"],
                },
                base_tone,
            ],
        }
    if kind == "no_plural_peers":
        return {
            "weights": {"no_plural": 0.7, "tone": 0.3},
            "checks": [
                {
                    "check_id": "no_plural",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)peer (companies|firms) have", "(?i)three (companies|peers|firms)"],
                },
                base_tone,
            ],
        }
    if kind == "icp_plus_dual_control":
        return {
            "weights": {"action": 0.4, "no_seg1": 0.4, "tone": 0.2},
            "checks": [
                {"check_id": "action", "type": "action_class", "expected": "ask_question", "detector": "regex_intent_question"},
                {
                    "check_id": "no_seg1",
                    "type": "banned_phrase_absent",
                    "patterns": ["(?i)scale.*(faster than|in-house) (recruiting|hiring)", "(?i)fresh (funding|capital).*(scale|grow)"],
                },
                base_tone,
            ],
        }
    # Default fallback
    return {
        "weights": {"policy": 0.5, "tone": 0.5},
        "checks": [
            {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
            base_tone,
        ],
    }


def generate(start_idx: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = start_idx
    for adv in ADVERSARIAL:
        out.append(
            {
                "task_id": task_id(idx),
                "source_mode": "hand_authored_adversarial",
                "dimension": adv["dimension"],
                "difficulty": "adversarial",
                "input": {
                    "channel": "email",
                    "scenario": adv["scenario"],
                    "hiring_signal_brief": {"primary_segment_match": "segment_1_series_a_b"},
                    "bench_summary": BENCH_TEMPLATE,
                    "prior_thread": adv["prior_thread"],
                    "expected_action": adv["expected_action"],
                },
                "rubric": _rubric_for(adv["rubric_kind"], adv["scenario"]),
                "metadata": {
                    "author_model": "human",
                    "judge_model": "qwen/qwen3-next-80b-a3b-instruct",
                    "judge_scores": {"input_coherence": 5, "ground_truth_verifiability": 5, "rubric_application_clarity": 5},
                    "exploit_note": adv["exploit_note"],
                    "created_at": now_iso(),
                    "license": "CC-BY-4.0",
                    "public_source_window": "n/a",
                },
            }
        )
        idx += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=600)
    args = p.parse_args()
    rows = generate(args.start)
    print(f"generated {len(rows)} hand-authored adversarial tasks")


if __name__ == "__main__":
    main()
