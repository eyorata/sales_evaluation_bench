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
from .common import DATASET_DIR, deterministic_seed, write_jsonl


JUDGE_THRESHOLD = 4  # Pointwise threshold per Liu et al. style judge filter.


def judge_filter(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter tasks by metadata.judge_scores >= JUDGE_THRESHOLD on each dimension.

    Tasks authored deterministically already carry the synthetic judge_scores their
    template promises. The filter is a real gate: any task with a missing or low
    score is rejected. Online-routed tasks would have judge_scores filled in by
    the judge model; offline tasks use the template's a-priori scores.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for r in rows:
        scores = r.get("metadata", {}).get("judge_scores") or {}
        ok = all(scores.get(k, 0) >= JUDGE_THRESHOLD for k in ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity"))
        if ok:
            accepted.append(r)
        else:
            rejected.append(r)
    return accepted, rejected


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
    for dim, items in by_dim.items():
        adv = [x for x in items if x["source_mode"] == "hand_authored_adversarial"]
        non_adv = [x for x in items if x["source_mode"] != "hand_authored_adversarial"]
        held.extend(adv)
        rng.shuffle(non_adv)
        n = len(non_adv)
        n_train = int(n * 0.5)
        n_dev = int(n * 0.3)
        train.extend(non_adv[:n_train])
        dev.extend(non_adv[n_train:n_train + n_dev])
        held.extend(non_adv[n_train + n_dev:])

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

    other_grams: set[tuple[str, ...]] = set()
    for r in train + dev:
        other_grams |= _ngrams(r)
    keep2: list[dict[str, Any]] = []
    demoted_ng: list[dict[str, Any]] = []
    for r in held:
        if _ngrams(r) & other_grams:
            demoted_ng.append(r)
        else:
            keep2.append(r)
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
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=deterministic_seed("tenacious_bench_v0.1"))
    args = p.parse_args()

    # 1) Author. Reserve task-id ranges so offline + online runs don't collide.
    prog = gen_prog(seed=args.seed, start_idx=1)               # 80 tasks: TB-0001..0080
    trace = gen_trace(seed=args.seed + 1, start_idx=200)       # 80 tasks: TB-0200..0279
    syn = gen_syn(seed_int=args.seed + 2, start_idx=400)       # 64 tasks: TB-0400..0463
    adv = gen_adv(start_idx=600)                               # 30 tasks: TB-0600..0629
    all_rows: list[dict[str, Any]] = prog + trace + syn + adv
    print(f"authored: prog={len(prog)} trace={len(trace)} syn={len(syn)} adv={len(adv)} total={len(all_rows)}")

    # 2) Preference-leakage check (Li et al. 2025).
    bad = check_no_leakage(all_rows)
    if bad:
        print(f"FATAL: {len(bad)} tasks fail same-family check: {bad[:5]}...")
        return 1
    print("preference-leakage check: PASS")

    # 3) Judge-filter (Liu et al. style, threshold 4 per dimension).
    accepted, rejected = judge_filter(all_rows)
    print(f"judge-filter: accepted={len(accepted)} rejected={len(rejected)} threshold={JUDGE_THRESHOLD}")
    write_jsonl(DATASET_DIR / "judge_filter_log.jsonl", [
        {"task_id": r["task_id"], "verdict": "rejected", "scores": r.get("metadata", {}).get("judge_scores")} for r in rejected
    ])

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
