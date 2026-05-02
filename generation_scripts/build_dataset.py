"""Coordinator: run all four authoring modes, judge-filter, partition, write to disk.

Usage:
  python -m generation_scripts.build_dataset
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .author_adversarial import generate as gen_adv
from .author_programmatic import generate as gen_prog
from .author_synthesis import generate as gen_syn
from .author_trace_derived import generate as gen_trace
from .style_guide_seed import generate as gen_style_guide
from .common import DATASET_DIR, deterministic_seed, write_jsonl


JUDGE_THRESHOLD = 4               # Pointwise threshold per Liu et al. style judge filter.
PAIRWISE_DUP_THRESHOLD = 0.97     # Hashed-trigram cosine; above this, two tasks are
                                   # considered near-duplicates and one wins via tie-break.
PAIRWISE_SCOPED_MODES = {"multi_llm_synthesis"}
                                   # Pairwise comparison runs ONLY on these source modes.
                                   # Rationale: programmatic + trace_derived modes are
                                   # intentionally similar by parameter sweep — two
                                   # programmatic tasks differing only in stack name or
                                   # demand count are NOT duplicates, they are the dataset's
                                   # combinatorial coverage. Real near-duplicates arise in
                                   # multi-LLM synthesis when LLM-generated variants from
                                   # the same seed_id drift to similar text; that's where
                                   # the pairwise gate is semantically meaningful. Hand-
                                   # authored adversarial and style-guide pairs are unique
                                   # by construction.
PAIRWISE_GROUP_KEY = ("source_mode", "dimension")
                                   # Within the scoped modes, compare pairs WITHIN a
                                   # (source_mode, dimension) group only.


def _hashed_trigram_vec(text: str, dim: int = 4096) -> list[int]:
    """Cheap, deterministic surrogate for sentence-transformers cosine. Same
    function shape as contamination_check.py to keep the metric consistent."""
    import hashlib
    counts = [0] * dim
    text = "".join(c for c in text if c.isalnum() or c == " ")
    toks = text.split()
    if len(toks) < 3:
        return counts
    for i in range(len(toks) - 2):
        h = int(hashlib.md5((" ".join(toks[i:i + 3])).encode("utf-8")).hexdigest()[:8], 16)
        counts[h % dim] += 1
    return counts


def _cos(a: list[int], b: list[int]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return (dot / (na * nb)) if (na and nb) else 0.0


def _task_text_for_pairwise(task: dict[str, Any]) -> str:
    """Concatenated input the agent sees, for pairwise-similarity comparison."""
    parts = [task["input"].get("scenario", "")]
    for turn in task["input"].get("prior_thread") or []:
        if turn.get("role") != "system":
            parts.append(turn.get("body", ""))
    return " ".join(parts).lower()


def _mean_judge_score(task: dict[str, Any]) -> float:
    s = task.get("metadata", {}).get("judge_scores") or {}
    keys = ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity")
    vals = [s.get(k, 0) for k in keys]
    return sum(vals) / max(1, len(vals))


def _family_diversity_key(task: dict[str, Any]) -> tuple[str, str]:
    """Tie-breaking key: prefer tasks where author and judge are distinct families.
    Tasks with `human`/`programmatic_v1`/`trace_derived_v1` authors get a sentinel
    family so they don't artificially win the diversity check."""
    md = task.get("metadata", {}) or {}
    a = md.get("author_model", "human")
    j = md.get("judge_model", "human")
    a_fam = a.split("/")[0] if "/" in a else "deterministic"
    j_fam = j.split("/")[0] if "/" in j else "deterministic"
    return (a_fam, j_fam)


def pairwise_dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pairwise near-duplicate detection with tie-breaking.

    Pairs are compared ONLY within the same (source_mode, dimension) group.
    Cross-group comparisons are skipped because templated boilerplate inflates
    cosine similarity for tasks that probe different failure modes.

    Two tasks are near-duplicates if their input text has hashed-trigram cosine
    >= PAIRWISE_DUP_THRESHOLD (0.97). For each near-dup pair, one survives via
    this deterministic tie-break:
      1. Higher mean of judge_scores wins.
      2. If tied, more diverse author/judge family pair wins (distinct
         families beat same-family-or-deterministic pairs).
      3. If still tied, lower task_id wins (deterministic fallback).

    Returns (kept, dropped). Each dropped row carries `pairwise_dup_reason`
    pointing at the survivor's task_id, ready to log.
    """
    if len(rows) < 2:
        return rows, []
    # Group by (source_mode, dimension) so pairwise comparison is in-group only.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r.get(PAIRWISE_GROUP_KEY[0], "?"), r.get(PAIRWISE_GROUP_KEY[1], "?"))
        groups.setdefault(key, []).append(r)

    dropped_ids: set[str] = set()
    decisions: list[dict[str, Any]] = []
    for key, group in groups.items():
        # Scope: only run pairwise on the source modes where it's semantically
        # meaningful (multi_llm_synthesis variants from the same seed_id).
        if key[0] not in PAIRWISE_SCOPED_MODES:
            continue
        if len(group) < 2:
            continue
        vecs = [(r["task_id"], _hashed_trigram_vec(_task_text_for_pairwise(r)), r) for r in group]
        for i in range(len(vecs)):
            if vecs[i][0] in dropped_ids:
                continue
            for j in range(i + 1, len(vecs)):
                if vecs[j][0] in dropped_ids:
                    continue
                sim = _cos(vecs[i][1], vecs[j][1])
                if sim < PAIRWISE_DUP_THRESHOLD:
                    continue
                ra, rb = vecs[i][2], vecs[j][2]
                sa, sb = _mean_judge_score(ra), _mean_judge_score(rb)
                if sa != sb:
                    winner, loser = (ra, rb) if sa > sb else (rb, ra)
                    tie_break_used = "judge_score"
                else:
                    fa, fb = _family_diversity_key(ra), _family_diversity_key(rb)
                    a_diverse = fa[0] != fa[1] and fa[0] != "deterministic"
                    b_diverse = fb[0] != fb[1] and fb[0] != "deterministic"
                    if a_diverse and not b_diverse:
                        winner, loser = ra, rb
                        tie_break_used = "family_diversity"
                    elif b_diverse and not a_diverse:
                        winner, loser = rb, ra
                        tie_break_used = "family_diversity"
                    else:
                        winner, loser = (ra, rb) if ra["task_id"] < rb["task_id"] else (rb, ra)
                        tie_break_used = "task_id_lex"
                dropped_ids.add(loser["task_id"])
                decisions.append({
                    "group": list(key),
                    "kept": winner["task_id"],
                    "dropped": loser["task_id"],
                    "cosine": round(sim, 4),
                    "tie_break": tie_break_used,
                    "kept_mean_score": round(_mean_judge_score(winner), 3),
                    "dropped_mean_score": round(_mean_judge_score(loser), 3),
                })

    kept = [r for r in rows if r["task_id"] not in dropped_ids]
    dropped = [
        {**r, "pairwise_dup_reason": next(d for d in decisions if d["dropped"] == r["task_id"])}
        for r in rows if r["task_id"] in dropped_ids
    ]
    return kept, dropped


def judge_filter(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two-stage judge filter:
      Stage A — pointwise: reject any task whose metadata.judge_scores has any
        sub-score below JUDGE_THRESHOLD on input_coherence,
        ground_truth_verifiability, or rubric_application_clarity.
      Stage B — pairwise: detect near-duplicate accepted tasks (hashed-trigram
        cosine >= PAIRWISE_DUP_THRESHOLD). Tie-break per pairwise_dedup() and
        emit one survivor per duplicate cluster.

    Returns (accepted, rejected). Each rejected row carries either
    `metadata.judge_scores` (pointwise reject) or `pairwise_dup_reason`
    (pairwise reject) so the downstream log can attribute every drop.
    """
    pointwise_accepted: list[dict[str, Any]] = []
    pointwise_rejected: list[dict[str, Any]] = []
    for r in rows:
        scores = r.get("metadata", {}).get("judge_scores") or {}
        ok = all(scores.get(k, 0) >= JUDGE_THRESHOLD for k in (
            "input_coherence", "ground_truth_verifiability", "rubric_application_clarity"
        ))
        if ok:
            pointwise_accepted.append(r)
        else:
            pointwise_rejected.append({**r, "reject_reason": "pointwise_below_threshold"})

    accepted, pairwise_rejected = pairwise_dedup(pointwise_accepted)
    pairwise_rejected = [{**r, "reject_reason": "pairwise_near_duplicate"} for r in pairwise_rejected]
    return accepted, pointwise_rejected + pairwise_rejected


def check_no_leakage(rows: list[dict[str, Any]]) -> list[str]:
    """Per Li et al. 2025: refuse any task whose author_model and judge_model are
    the same family. Returns list of offending task_ids (empty = clean)."""
    bad: list[str] = []
    for r in rows:
        a = r["metadata"]["author_model"]
        j = r["metadata"]["judge_model"]
        # 'human', 'programmatic_v1', 'trace_derived_v1' are not LLM families and
        # cannot leak into LLM judging — skip.
        if a in ("human", "programmatic_v1", "trace_derived_v1"):
            continue
        if a.split("/")[0] == j.split("/")[0]:
            bad.append(r["task_id"])
    return bad


def _body_signature(task: dict[str, Any]) -> str:
    """Hash of prospect-facing prior_thread bodies. Used for exact-duplicate
    dedup across partitions."""
    parts: list[str] = []
    for turn in task["input"].get("prior_thread") or []:
        if turn.get("role") == "system":
            continue
        parts.append((turn.get("body") or "").strip().lower())
    return "|".join(parts)


def partition(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """50/30/20 train/dev/held_out, stratified by dimension. Hand-authored
    adversarial tasks are forced into held_out (per the brief). After the split,
    any held-out task whose prior_thread body exactly matches a train/dev task
    is moved to train (contamination is the held-out→leak direction)."""
    rng = random.Random(seed)
    by_dim: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_dim.setdefault(r["dimension"], []).append(r)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    forced_held_modes = {"hand_authored_adversarial", "style_guide_pair"}
    for dim, items in by_dim.items():
        forced = [x for x in items if x["source_mode"] in forced_held_modes]
        non_forced = [x for x in items if x["source_mode"] not in forced_held_modes]
        held.extend(forced)
        rng.shuffle(non_forced)
        n = len(non_forced)
        n_train = int(n * 0.5)
        n_dev = int(n * 0.3)
        train.extend(non_forced[:n_train])
        dev.extend(non_forced[n_train:n_train + n_dev])
        held.extend(non_forced[n_train + n_dev:])

    # Dedup pass: held-out tasks whose body matches anything in train+dev get
    # demoted to train. Empty-body tasks (synthesis-mode) are exempt — they have
    # no prospect-facing input that could leak.
    other_bodies: set[str] = {_body_signature(r) for r in train + dev}
    other_bodies.discard("")
    keep_held: list[dict[str, Any]] = []
    demoted: list[dict[str, Any]] = []
    for r in held:
        sig = _body_signature(r)
        if sig and sig in other_bodies:
            demoted.append(r)
        else:
            keep_held.append(r)
            if sig:
                other_bodies.add(sig)
    if demoted:
        train.extend(demoted)
    held = keep_held

    # N-gram contamination pass (per Chen et al.): held-out tasks sharing an
    # 8-gram with any train/dev task are demoted. Mirrors contamination_check.py
    # so the published held-out partition is clean by construction.
    def _ngrams(task: dict[str, Any], n: int = 8) -> set[tuple[str, ...]]:
        toks: list[str] = []
        for turn in task["input"].get("prior_thread") or []:
            if turn.get("role") == "system":
                continue
            toks.extend((turn.get("body") or "").lower().split())
        return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()

    keep2: list[dict[str, Any]] = []
    demoted_ng: list[dict[str, Any]] = []
    
    # Explicit pairwise near-duplicate comparison logic
    for h_task in held:
        h_grams = _ngrams(h_task)
        if not h_grams:
            keep2.append(h_task)
            continue
            
        is_near_dup = False
        for o_task in train + dev:
            o_grams = _ngrams(o_task)
            if h_grams & o_grams:  # Pairwise overlap detection
                is_near_dup = True
                break
                
        if is_near_dup:
            demoted_ng.append(h_task)
        else:
            keep2.append(h_task)

    if demoted_ng:
        train.extend(demoted_ng)
    held = keep2
    if demoted or demoted_ng:
        print(
            f"dedup: demoted {len(demoted)} body-duplicates and "
            f"{len(demoted_ng)} 8-gram overlaps to train"
        )

    for r in train:
        r["partition"] = "train"
    for r in dev:
        r["partition"] = "dev"
    for r in held:
        r["partition"] = "held_out"
    return {"train": train, "dev": dev, "held_out": held}


def main() -> int:
    """MAIN ORCHESTRATION ENTRYPOINT
    
    Coordinates the dataset authoring, filtering, and partitioning.
    
    Judge Filter Dimensions & Thresholds:
    Every task is passed through a pointwise LLM judge (or offline scores for deterministic tasks).
    To be accepted, a task must score >= 4 (out of 5) on all three dimensions:
      1. input_coherence: Is the scenario and trace internally logical?
      2. ground_truth_verifiability: Is there enough signal data to judge the interaction?
      3. rubric_application_clarity: Can the grading rubric be applied deterministically?
    """
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=deterministic_seed("tenacious_bench_v0.1"))
    p.add_argument("--online-synthesis", action="store_true",
                   help="Route the multi-LLM synthesis mode through OpenRouter (real LLM calls).")
    args = p.parse_args()

    # 1) Author. Reserve task-id ranges so offline + online runs don't collide.
    prog = gen_prog(seed=args.seed, start_idx=1)               # 80 tasks: TB-0001..0080
    trace = gen_trace(seed=args.seed + 1, start_idx=200)       # 80 tasks: TB-0200..0279
    syn = gen_syn(seed_int=args.seed + 2, start_idx=400, online=args.online_synthesis)
    adv = gen_adv(start_idx=600)                               # 30 tasks: TB-0600..0629
    sg = gen_style_guide(start_idx=700)                        # 12 tasks: TB-0700..0711
    all_rows: list[dict[str, Any]] = prog + trace + syn + adv + sg
    print(f"authored: prog={len(prog)} trace={len(trace)} syn={len(syn)} adv={len(adv)} sg={len(sg)} total={len(all_rows)}")

    # 2) Preference-leakage check (Li et al. 2025).
    bad = check_no_leakage(all_rows)
    if bad:
        print(f"FATAL: {len(bad)} tasks fail same-family check: {bad[:5]}...")
        return 1
    print("preference-leakage check: PASS")

    # 3) Judge-filter: pointwise (per Liu et al.) + pairwise near-duplicate dedup.
    accepted, rejected = judge_filter(all_rows)
    pointwise_n = sum(1 for r in rejected if r.get("reject_reason") == "pointwise_below_threshold")
    pairwise_n = sum(1 for r in rejected if r.get("reject_reason") == "pairwise_near_duplicate")
    print(
        f"judge-filter: accepted={len(accepted)} rejected={len(rejected)} "
        f"(pointwise={pointwise_n} pairwise_dup={pairwise_n}) "
        f"thresholds: pointwise>={JUDGE_THRESHOLD} pairwise_cosine>={PAIRWISE_DUP_THRESHOLD}"
    )
    # Rich log entries: separate the two reject paths so reviewers can audit each.
    log_rows = []
    for r in rejected:
        reason = r.get("reject_reason", "unknown")
        entry = {
            "task_id": r["task_id"],
            "verdict": "rejected",
            "reject_reason": reason,
            "source_mode": r.get("source_mode"),
            "dimension": r.get("dimension"),
        }
        if reason == "pointwise_below_threshold":
            entry["scores"] = r.get("metadata", {}).get("judge_scores")
            entry["threshold"] = JUDGE_THRESHOLD
        elif reason == "pairwise_near_duplicate":
            entry["pairwise_dup_reason"] = r.get("pairwise_dup_reason")
        log_rows.append(entry)
    write_jsonl(DATASET_DIR / "judge_filter_log.jsonl", log_rows)

    # 4) Partition.
    parts = partition(accepted, seed=args.seed)
    for name, items in parts.items():
        write_jsonl(DATASET_DIR / name / "tasks.jsonl", items)

    # 5) Write composition stats.
    counts = {
        name: {
            "total": len(items),
            "by_dimension": dict(Counter(r["dimension"] for r in items)),
            "by_source_mode": dict(Counter(r["source_mode"] for r in items)),
            "by_difficulty": dict(Counter(r["difficulty"] for r in items)),
        }
        for name, items in parts.items()
    }
    counts["overall"] = {
        "total": sum(len(v) for v in parts.values()),
        "by_dimension": dict(Counter(r["dimension"] for r in accepted)),
        "by_source_mode": dict(Counter(r["source_mode"] for r in accepted)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in accepted)),
    }
    (DATASET_DIR / "composition.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"wrote dataset to {DATASET_DIR}")
    print(json.dumps(counts["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
