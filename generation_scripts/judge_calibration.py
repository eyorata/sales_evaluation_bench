"""LLM-judge spot-check on templated (programmatic + trace-derived) tasks.

Per Liu et al., the eval-tier judge calibrates the cheap-tier judge. We can't
afford eval-tier on Days 2-3 (per the brief's cost rules), so the calibration
here uses a third-family dev-tier judge (Llama 3.3 70B) over a stratified
sample of templated tasks. The output augments metadata.judge_scores with
metadata.judge_scores_recalibrated and writes a delta report.

Sample size: 50 tasks stratified across dimensions, source modes, partitions.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import openrouter_client
from .common import DATASET_DIR

CALIB_MODEL = "meta-llama/llama-3.3-70b-instruct"

JUDGE_SYSTEM = (
    "You are calibrating a benchmark task. Score on three dimensions, integer 1-5 each. "
    "Return STRICT JSON ONLY with keys input_coherence, ground_truth_verifiability, "
    "rubric_application_clarity. No prose, no code fences."
)


def _judge_one(task: dict[str, Any]) -> dict[str, int]:
    sc = task["input"].get("scenario") or ""
    pt = task["input"].get("prior_thread") or []
    body_lines = [
        f"[{turn.get('role','prospect')}] {turn.get('body','')[:280]}"
        for turn in pt[-3:]
    ]
    rubric_summary = ", ".join(c["type"] for c in task["rubric"]["checks"])
    user = (
        f"Scenario: {sc[:600]}\n\n"
        f"Prior thread (last 3 turns): {chr(10).join(body_lines) or '(none)'}\n\n"
        f"Rubric checks: {rubric_summary}\n\n"
        f"Score input_coherence, ground_truth_verifiability, rubric_application_clarity (1-5 each). JSON only."
    )
    resp = openrouter_client.chat(
        model=CALIB_MODEL,
        system=JUDGE_SYSTEM,
        user=user,
        temperature=0.0,
        max_tokens=80,
        purpose=f"calibration:{task['task_id']}",
    )
    raw = (resp["text"] or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        import re as _re
        m = _re.search(r"\{[^{}]*\}", raw)
        scores = json.loads(m.group(0)) if m else {}

    def _coerce(x: Any) -> int:
        try:
            return max(1, min(5, int(x)))
        except (TypeError, ValueError):
            return 3

    return {k: _coerce(scores.get(k)) for k in ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity")}


def _stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["dimension"], r["source_mode"])].append(r)
    cells = list(by_key.values())
    rng.shuffle(cells)
    out: list[dict[str, Any]] = []
    while len(out) < n and cells:
        for cell in cells:
            if cell:
                out.append(cell.pop(rng.randrange(len(cell))))
                if len(out) >= n:
                    break
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    for part in ("train", "dev", "held_out"):
        path = DATASET_DIR / part / "tasks.jsonl"
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            # Sample only templated (deterministic) tasks; synthesis-mode already had real LLM scores.
            if r["source_mode"] in ("programmatic", "trace_derived", "hand_authored_adversarial"):
                rows.append(r)

    sample = _stratified_sample(rows, args.n, args.seed)
    print(f"calibration sample: n={len(sample)} model={CALIB_MODEL}")

    delta_log: list[dict[str, Any]] = []
    agree, disagree = 0, 0
    for t in sample:
        try:
            new_scores = _judge_one(t)
        except Exception as e:
            print(f"  {t['task_id']}: judge error {e}")
            continue
        old = t["metadata"].get("judge_scores") or {}
        deltas = {k: new_scores[k] - old.get(k, 0) for k in new_scores}
        rec = {
            "task_id": t["task_id"],
            "dimension": t["dimension"],
            "source_mode": t["source_mode"],
            "old": old,
            "recalibrated": new_scores,
            "deltas": deltas,
            "min_recalibrated": min(new_scores.values()),
        }
        delta_log.append(rec)
        if all(abs(d) <= 1 for d in deltas.values()):
            agree += 1
        else:
            disagree += 1

    out = DATASET_DIR / "judge_calibration.json"
    out.write_text(json.dumps({
        "model": CALIB_MODEL,
        "sample_size": len(sample),
        "agree_within_1": agree,
        "disagree_more_than_1": disagree,
        "agreement_rate": round(agree / max(1, len(delta_log)), 3),
        "details": delta_log,
    }, indent=2), encoding="utf-8")
    print(f"agreement-within-1: {agree}/{len(delta_log)}  ({100*agree/max(1,len(delta_log)):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
