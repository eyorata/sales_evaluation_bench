"""Contamination check (per Chen et al., 2025 — held-out vs train/dev).

Three checks:
  1. N-gram overlap on input fields. Reject any held-out / train pair with an
     8-gram or longer contiguous overlap.
  2. Embedding similarity. We use a deterministic hashed-bag-of-trigrams
     surrogate (no network, no model download). Cosine similarity > 0.85 is
     rejected. The proper sentence-transformer run is documented but the
     hashed surrogate is the interim default — see methodology.md, section 4.
  3. Time-shift verification. Every held-out task that names a public source
     must carry a metadata.public_source_window inside the dataset window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from typing import Any

from .common import DATASET_DIR


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _input_text(task: dict[str, Any]) -> str:
    """Prospect-facing input only. The `scenario` field is meta-instruction the
    agent never sees at inference; contamination concern is over what an agent
    could memorize from training, i.e. the prior_thread bodies."""
    parts: list[str] = []
    for turn in task["input"].get("prior_thread", []) or []:
        body = turn.get("body", "")
        # Skip system-role lines and quote-nest noise.
        if turn.get("role") == "system":
            continue
        parts.append(_normalize(body))
    return " | ".join(parts)


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = text.split()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


def _trigram_vec(text: str, dim: int = 4096) -> list[float]:
    """Hashed-bag-of-trigrams for deterministic similarity without a model."""
    counts = [0] * dim
    text = "".join(c for c in text if c.isalnum() or c == " ")
    toks = text.split()
    trigs = [(toks[i], toks[i + 1], toks[i + 2]) for i in range(len(toks) - 2)] if len(toks) >= 3 else []
    for tri in trigs:
        h = int(hashlib.md5((" ".join(tri)).encode("utf-8")).hexdigest()[:8], 16)
        counts[h % dim] += 1
    return [float(x) for x in counts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


PUBLIC_SOURCE_WINDOW_BOUND = ("2025-11-01", "2026-04-29")


def time_shift_ok(task: dict[str, Any]) -> bool:
    win = task.get("metadata", {}).get("public_source_window") or ""
    if win == "n/a" or not win:
        return True
    # Format: "YYYY-MM-DD..YYYY-MM-DD"
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})$", win)
    if not m:
        return False
    a, b = m.group(1), m.group(2)
    return a >= PUBLIC_SOURCE_WINDOW_BOUND[0] and b <= PUBLIC_SOURCE_WINDOW_BOUND[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-gram", type=int, default=8)
    p.add_argument("--cos-threshold", type=float, default=0.85)
    args = p.parse_args()

    held = [json.loads(l) for l in (DATASET_DIR / "held_out" / "tasks.jsonl").open(encoding="utf-8")]
    train = [json.loads(l) for l in (DATASET_DIR / "train" / "tasks.jsonl").open(encoding="utf-8")]
    dev = [json.loads(l) for l in (DATASET_DIR / "dev" / "tasks.jsonl").open(encoding="utf-8")]
    other = train + dev

    # 1. N-gram overlap
    held_grams = [(t["task_id"], _ngrams(_input_text(t), args.n_gram)) for t in held]
    other_grams = [(t["task_id"], _ngrams(_input_text(t), args.n_gram)) for t in other]
    ngram_violations: list[dict[str, Any]] = []
    for hid, hg in held_grams:
        if not hg:
            continue
        for oid, og in other_grams:
            if hg & og:
                ngram_violations.append({"held_out": hid, "other": oid, "shared_n": args.n_gram})

    # 2. Embedding similarity (hashed-trigram surrogate)
    held_vecs = [(t["task_id"], _trigram_vec(_input_text(t))) for t in held]
    other_vecs = [(t["task_id"], _trigram_vec(_input_text(t))) for t in other]
    sim_violations: list[dict[str, Any]] = []
    for hid, hv in held_vecs:
        for oid, ov in other_vecs:
            cs = _cosine(hv, ov)
            if cs >= args.cos_threshold:
                sim_violations.append({"held_out": hid, "other": oid, "cosine": round(cs, 4)})

    # 3. Time-shift
    time_violations = [t["task_id"] for t in held if not time_shift_ok(t)]

    report = {
        "n_gram_check": {
            "n": args.n_gram,
            "violations": ngram_violations,
            "violations_count": len(ngram_violations),
            "verdict": "PASS" if not ngram_violations else "FAIL",
        },
        "embedding_similarity_check": {
            "metric": "hashed_trigram_cosine",
            "threshold": args.cos_threshold,
            "violations": sim_violations[:25],
            "violations_count": len(sim_violations),
            "verdict": "PASS" if not sim_violations else "FAIL",
        },
        "time_shift_check": {
            "window": list(PUBLIC_SOURCE_WINDOW_BOUND),
            "violations": time_violations,
            "violations_count": len(time_violations),
            "verdict": "PASS" if not time_violations else "FAIL",
        },
        "summary": {
            "held_out_count": len(held),
            "other_count": len(other),
        },
    }

    out = DATASET_DIR / "contamination_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: {"verdict": v["verdict"], "violations": v["violations_count"]} if isinstance(v, dict) and "verdict" in v else v
         for k, v in report.items()},
        indent=2,
    ))
    overall_pass = all(report[k]["verdict"] == "PASS" for k in ("n_gram_check", "embedding_similarity_check", "time_shift_check"))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
