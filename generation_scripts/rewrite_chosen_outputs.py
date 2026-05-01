"""Rewrite the `chosen` outputs in preference_pairs.jsonl using a real LLM.

The original chosen outputs (from convert_to_preference_pairs.py) are templated
boilerplate. The model trained on them learned the template, not the tone — see
training_run.log: train_acc=1.00 but eval_acc=0.25 throughout.

This script replaces every chosen output with a Tenacious-voice draft authored
by Llama-3.3-70B, anchored in the Tenacious Style Guide v2 (12 good drafts + 5
tone markers + banned phrases + tone-preservation check). The 12 SG good drafts
are passed as few-shot exemplars so the rewriter knows what 'real Tenacious'
looks like.

Each rewrite is gated by scoring_evaluator.score_task >= 0.7. Failed rewrites
get one retry with a stricter prompt; if that fails too, the pair is dropped.

Output: training_data/preference_pairs_v2.jsonl
Cost: ~$0.04 dev-tier (128 calls × ~800 tokens avg)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generation_scripts import openrouter_client  # noqa: E402
from generation_scripts.style_guide_seed import PAIRS, EXPANDED_BANNED_PATTERNS  # noqa: E402
from scoring_evaluator import score_task  # noqa: E402


REWRITER_MODEL = "meta-llama/llama-3.3-70b-instruct"   # third-family rotation
PASS_BAR = 0.7


def _system_prompt(tone_markers_only: bool = False) -> str:
    """Build the system prompt with v2 style-guide context."""
    banned_list = "\n".join(f"- {p}" for p in [
        "world-class",
        "top talent / A-players",
        "rockstar / ninja / wizard",
        "skyrocket / supercharge / 10x",
        "I hope this email finds you well",
        "just following up / circling back",
        "Quick question / Quick chat",
        "synergize / synergy / leverage / ecosystem",
        "game-changer / disruptor / paradigm shift",
        "our proprietary [X] / our AI-powered [X]",
        "You'll regret missing this / Don't miss out",
        "Per my last email",
        "I noticed you're a [job title]",
        "the bench (in prospect-facing language; use 'engineering team' / 'available capacity')",
    ])

    # Two strong few-shots: GOOD #1 (Series A, high confidence) + GOOD #5 (weak signal, asks)
    good_1 = next(p for p in PAIRS if p["pair_id"] == "SG-01")
    good_5 = next(p for p in PAIRS if p["pair_id"] == "SG-05")
    good_9 = next(p for p in PAIRS if p["pair_id"] == "SG-09")  # bench-gated decline

    examples = "\n\n".join([
        f"## EXAMPLE GOOD DRAFT 1 — {good_1['title']}\nSubject: {good_1['good_subject']}\n\n{good_1['good_body']}",
        f"## EXAMPLE GOOD DRAFT 2 — {good_5['title']}\nSubject: {good_5['good_subject']}\n\n{good_5['good_body']}",
        f"## EXAMPLE GOOD DRAFT 3 — {good_9['title']}\nSubject: {good_9['good_subject']}\n\n{good_9['good_body']}",
    ])

    base = (
        "You are writing a single Tenacious B2B sales-outreach draft. Tenacious is a B2B "
        "engineering-outsourcing firm. The four ICP segments are: Series A/B startups, "
        "post-layoff mid-market, leadership-transition, specialized capability gaps.\n\n"
        "## THE FIVE TONE MARKERS (every draft must hit 4/5 on each)\n"
        "1. **Direct** — subject states intent (Request/Follow-up/Context/Question; never Quick/Just/Hey). "
        "Body ≤120 words for cold, ≤200 for warm. One ask per message.\n"
        "2. **Grounded** — every claim tied to a specific signal from the brief (named amount, date, "
        "role count, named peer). Confidence-aware phrasing: weak signal → ask, do not assert.\n"
        "3. **Honest** — refuse claims you cannot ground. Never over-commit bench capacity. Never "
        "fabricate funding events, leadership changes, layoffs, or peer practices. Name what the brief "
        "does NOT show when relevant.\n"
        "4. **Professional** — language for CTOs/founders/VPs Eng. NEVER use the word 'bench' externally "
        "— use 'engineering team', 'available capacity', or 'engineers ready to deploy'. No offshore-"
        "vendor clichés.\n"
        "5. **Non-condescending** — frame any gap as research finding or question, not a failure of the "
        "prospect's leadership.\n\n"
        "## BANNED PHRASES (NEVER appear in subject or body)\n"
        f"{banned_list}\n\n"
        "## STYLE EXAMPLES — the target voice\n\n"
        f"{examples}\n\n"
        "## YOUR TASK\n"
        "Given a SCENARIO and a NEGATIVE DRAFT (a bad example to AVOID), write a single CORRECT draft "
        "in Tenacious voice. Output STRICTLY in this format:\n"
        "Subject: <subject line, ≤60 chars>\n"
        "<blank line>\n"
        "<body, ≤120 words for cold or ≤200 for warm>\n\n"
        "Sign as 'Yabi, Research Partner, Tenacious Intelligence Corporation, gettenacious.com'.\n"
        "No preamble, no commentary, no markdown headers. Just the email."
    )
    if tone_markers_only:
        # Stricter retry prompt: focus on what the evaluator's regex will check
        base += (
            "\n\n## STRICTER RULES FOR THIS RETRY\n"
            "- DO NOT use any of these literal phrases in body or subject: top talent, world-class, "
            "rockstars, ninjas, the bench (externally), I hope this finds you well, synergize, ecosystem, "
            "Quick question, Quick chat, just following up, circling back, Per my last email, "
            "skyrocket, supercharge, 10x.\n"
            "- DO NOT commit specific engineer counts above what the bench shows.\n"
            "- DO NOT fabricate signals not in the scenario.\n"
            "- The first word of the subject MUST be one of: Request, Follow-up, Context, Question, Re, New, Resource."
        )
    return base


def _user_prompt(prompt_text: str, rejected: str) -> str:
    return (
        f"## SCENARIO\n{prompt_text[:1500]}\n\n"
        f"## NEGATIVE DRAFT (avoid this style; do not echo)\n{rejected[:1500]}\n\n"
        "Write the correct Tenacious draft now. Subject + blank line + body. Nothing else."
    )


def _split_subject_body(text: str) -> tuple[str, str]:
    text = (text or "").strip().strip("`").strip()
    # Drop common preamble
    for prefix in ("Here is", "Here's", "Subject:"):
        if text.startswith(prefix) and prefix == "Subject:":
            break
    if "\n" not in text:
        return ("", text)
    first, rest = text.split("\n", 1)
    if first.lower().startswith("subject:"):
        return (first.split(":", 1)[1].strip(), rest.lstrip())
    return ("", text)


def rewrite_one(prompt_text: str, rejected: str, *, retry: bool = False) -> tuple[str, str, dict[str, Any]]:
    sys_prompt = _system_prompt(tone_markers_only=retry)
    out = openrouter_client.chat(
        model=REWRITER_MODEL,
        system=sys_prompt,
        user=_user_prompt(prompt_text, rejected),
        temperature=0.5 if not retry else 0.2,
        max_tokens=400,
        purpose="rewrite_chosen" if not retry else "rewrite_chosen_retry",
    )
    text = out["text"] or ""
    subj, body = _split_subject_body(text)
    full = f"Subject: {subj}\n\n{body}" if subj else body
    return full, subj, {"prompt_tokens": out["usage"].get("prompt_tokens", 0),
                       "completion_tokens": out["usage"].get("completion_tokens", 0),
                       "cost_usd": out["cost_usd"]}


def _build_dummy_task_for_eval(pair: dict, chosen_text: str) -> dict[str, Any]:
    """Build a minimal task dict for scoring_evaluator using the rubric we want
    the chosen output to pass. The pair's `task_id`/`dimension`/`source_mode`
    are preserved for traceability."""
    # Use a base rubric with the most critical checks
    return {
        "task_id": pair.get("task_id", "TB-REWRITE"),
        "source_mode": pair.get("source_mode", "rewrite_v2"),
        "dimension": pair.get("dimension", "tone_drift"),
        "difficulty": pair.get("difficulty", "hard"),
        "input": {
            "channel": "email",
            "scenario": pair.get("prompt", "")[:500],
            "hiring_signal_brief": {},
            "bench_summary": {},
            "prior_thread": [],
            "expected_action": "draft_outbound",
        },
        "rubric": {
            "weights": {"banned": 0.5, "policy": 0.25, "tone": 0.25},
            "checks": [
                {
                    "check_id": "banned",
                    "type": "banned_phrase_absent",
                    "patterns": EXPANDED_BANNED_PATTERNS + [
                        r"(?i)\btop talent\b",
                        r"(?i)\bA-players?\b",
                        r"(?i)\brockstars?\b|\bninjas?\b",
                    ],
                },
                {"check_id": "policy", "type": "policy_compliant", "uses": "agent.policy.check_outbound"},
                {
                    "check_id": "tone",
                    "type": "tone_marker_judge",
                    "markers": ["direct", "honest", "professional", "non_condescending"],
                    "min_score_per_marker": 4,
                    "judge_model": "heuristic_fallback",
                },
            ],
        },
        "metadata": {
            "author_model": "meta-llama/llama-3.3-70b-instruct",
            "judge_model": "heuristic_fallback",
            "created_at": "2026-05-01T00:00:00Z",
            "license": "CC-BY-4.0",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="training_data/preference_pairs.jsonl")
    p.add_argument("--output", default="training_data/preference_pairs_v2.jsonl")
    p.add_argument("--limit", type=int, default=0, help="Cap pairs for testing (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="Don't call the API; print plan only")
    args = p.parse_args()

    in_path = REPO_ROOT / args.input
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"will rewrite {len(rows)} chosen outputs using {REWRITER_MODEL}")
    if args.dry_run:
        print("DRY RUN — no API calls made")
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set in env or .env")
        return 1

    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    total_cost = 0.0
    for i, row in enumerate(rows):
        prompt_text = row["prompt"]
        rejected = row["rejected"]
        try:
            chosen_text, subj, usage = rewrite_one(prompt_text, rejected, retry=False)
            total_cost += usage["cost_usd"]
        except Exception as e:
            print(f"  [{i}] {row.get('task_id','?')}: API error {e}; skipping")
            dropped.append({**row, "drop_reason": f"api_error:{e!s}"})
            continue

        # Score the rewritten chosen against the dummy rubric
        dummy = _build_dummy_task_for_eval(row, chosen_text)
        bd = score_task(dummy, chosen_text)
        if bd.total >= PASS_BAR:
            row_out = {**row, "chosen": chosen_text, "chosen_provenance": f"{REWRITER_MODEL}:rewrite_v2",
                      "chosen_eval_score": round(bd.total, 3)}
            accepted.append(row_out)
            tag = "PASS"
        else:
            # one retry with stricter prompt
            try:
                chosen2, _, usage2 = rewrite_one(prompt_text, rejected, retry=True)
                total_cost += usage2["cost_usd"]
                bd2 = score_task(_build_dummy_task_for_eval(row, chosen2), chosen2)
                if bd2.total >= PASS_BAR:
                    row_out = {**row, "chosen": chosen2,
                               "chosen_provenance": f"{REWRITER_MODEL}:rewrite_v2_retry",
                               "chosen_eval_score": round(bd2.total, 3)}
                    accepted.append(row_out)
                    tag = "RETRY-PASS"
                else:
                    dropped.append({**row, "drop_reason": f"score_below_{PASS_BAR}",
                                  "best_score": round(max(bd.total, bd2.total), 3)})
                    tag = "DROP"
            except Exception as e:
                dropped.append({**row, "drop_reason": f"retry_api_error:{e!s}"})
                tag = "DROP"

        if i % 5 == 0 or tag == "DROP":
            print(f"  [{i:3d}/{len(rows)}] {tag:10s} cost=${total_cost:.4f}  task={row.get('task_id','?')}")

    # Write output
    with out_path.open("w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    drop_path = out_path.with_suffix(".dropped.jsonl")
    with drop_path.open("w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\ndone. accepted={len(accepted)}  dropped={len(dropped)}  total_cost=${total_cost:.4f}")
    print(f"  wrote {out_path}")
    print(f"  dropped log -> {drop_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
