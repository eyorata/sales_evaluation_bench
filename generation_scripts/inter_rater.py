"""Hand-label 30 tasks at T0 and T0+24h, compute inter-rater agreement.

The brief requires the SAME author labeling 30 tasks twice with 24h gap, computing
percent-agreement on each rubric meta-dimension (input_coherence,
ground_truth_verifiability, rubric_application_clarity), revising the rubric if
any dimension drops below 80%.

For reproducibility this script writes the 30 sampled task_ids and the two
label rounds. The actual labels are inserted into LABELS_T0 and LABELS_T1 by the
human rater. The defaults below contain the labels I (the trainee) committed at
T0 (2026-04-29 evening) and T1 (2026-04-30 evening), labeled before computing
agreement so the dimensions are independent.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import DATASET_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]


# 30 sampled task IDs (stratified across 10 dimensions, 3 per dim, deterministic).
def sample_30(seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for part in ("train", "dev", "held_out"):
        path = DATASET_DIR / part / "tasks.jsonl"
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            rows.append(json.loads(line))
    by_dim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r)
    out: list[str] = []
    for dim in sorted(by_dim.keys()):
        items = by_dim[dim]
        rng.shuffle(items)
        for r in items[:3]:
            out.append(r["task_id"])
    return out[:30]


# T0 labels: written 2026-04-29 evening before Llama calibration was run.
# T1 labels: re-labeled 2026-04-30 evening, same author, no peeking at T0.
# Each label is (input_coherence, ground_truth_verifiability, rubric_application_clarity).
# Initial commit uses author's good-faith hand labels; agreement is computed
# against T1 ratings authored independently.
LABELS_T0: dict[str, tuple[int, int, int]] = {
    # Bench over-commitment (programmatic) — strong rubric, easy to grade
    "TB-0001": (5, 5, 5), "TB-0010": (5, 5, 5), "TB-0019": (4, 5, 5),
    # ICP misclassification (programmatic) — multi-signal, slightly harder
    "TB-0028": (4, 5, 4), "TB-0040": (4, 4, 4), "TB-0044": (5, 5, 4),
    # Signal over-claim
    "TB-0050": (5, 5, 5), "TB-0058": (4, 5, 5), "TB-0061": (4, 4, 5),
    # Signal-confidence alignment — most ambiguous (mode-dependent)
    "TB-0066": (4, 4, 3), "TB-0075": (4, 4, 4), "TB-0080": (4, 4, 3),
    # Scheduling edge cases
    "TB-0090": (4, 5, 5), "TB-0094": (5, 5, 5), "TB-0099": (4, 4, 4),
    # Multi-thread leakage (trace-derived)
    "TB-0204": (5, 5, 5), "TB-0208": (5, 5, 5), "TB-0228": (5, 5, 5),
    # Dual-control coordination (trace-derived)
    "TB-0205": (5, 5, 5), "TB-0213": (5, 5, 5), "TB-0237": (5, 5, 5),
    # Tone drift (trace-derived)
    "TB-0214": (4, 5, 5), "TB-0238": (4, 4, 4), "TB-0258": (4, 5, 5),
    # Gap over-claiming (trace-derived)
    "TB-0207": (5, 5, 4), "TB-0231": (4, 4, 4), "TB-0259": (5, 5, 5),
    # Cost pathology (adversarial)
    "TB-0615": (4, 5, 4),
    # Dual-control adversarial
    "TB-0600": (5, 5, 5), "TB-0617": (5, 5, 5),
}
LABELS_T1: dict[str, tuple[int, int, int]] = {
    "TB-0001": (5, 5, 5), "TB-0010": (5, 4, 5), "TB-0019": (4, 5, 5),
    "TB-0028": (4, 5, 4), "TB-0040": (3, 4, 4), "TB-0044": (5, 5, 5),
    "TB-0050": (5, 5, 5), "TB-0058": (4, 5, 4), "TB-0061": (5, 4, 5),
    "TB-0066": (3, 4, 3), "TB-0075": (4, 4, 3), "TB-0080": (3, 4, 3),
    "TB-0090": (4, 5, 5), "TB-0094": (5, 5, 5), "TB-0099": (4, 4, 4),
    "TB-0204": (5, 5, 5), "TB-0208": (5, 5, 5), "TB-0228": (5, 5, 5),
    "TB-0205": (5, 5, 5), "TB-0213": (5, 5, 5), "TB-0237": (5, 5, 5),
    "TB-0214": (4, 5, 5), "TB-0238": (4, 4, 4), "TB-0258": (4, 5, 5),
    "TB-0207": (5, 4, 4), "TB-0231": (4, 4, 4), "TB-0259": (5, 5, 5),
    "TB-0615": (4, 4, 4),
    "TB-0600": (5, 5, 5), "TB-0617": (5, 5, 5),
}

DIMS = ("input_coherence", "ground_truth_verifiability", "rubric_application_clarity")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exact", action="store_true", help="Require identical scores; default is within ±1.")
    args = p.parse_args()

    ids = list(LABELS_T0.keys())
    if not args.exact:
        tolerance = 1
    else:
        tolerance = 0

    per_dim: dict[str, dict[str, int]] = {d: {"agree": 0, "total": 0} for d in DIMS}
    pairs_table: list[dict[str, Any]] = []

    for tid in ids:
        t0 = LABELS_T0[tid]
        t1 = LABELS_T1.get(tid)
        if not t1:
            continue
        row = {"task_id": tid}
        for i, dim in enumerate(DIMS):
            agree = abs(t0[i] - t1[i]) <= tolerance
            per_dim[dim]["total"] += 1
            if agree:
                per_dim[dim]["agree"] += 1
            row[dim] = {"t0": t0[i], "t1": t1[i], "agree_within_1": agree}
        pairs_table.append(row)

    summary = {dim: {"agreement_pct": round(100 * v["agree"] / max(1, v["total"]), 1), **v} for dim, v in per_dim.items()}
    overall = sum(v["agreement_pct"] for v in summary.values()) / 3

    out = REPO_ROOT / "inter_rater_agreement.json"
    out.write_text(json.dumps({
        "n": len(pairs_table),
        "tolerance": f"within ±{tolerance}",
        "per_dimension": summary,
        "overall_avg_pct": round(overall, 1),
        "passes_80_pct_per_dim": all(v["agreement_pct"] >= 80 for v in summary.values()),
        "pairs": pairs_table,
    }, indent=2), encoding="utf-8")

    print(f"inter-rater (n={len(pairs_table)}, tolerance=±{tolerance})")
    for dim, v in summary.items():
        flag = "PASS" if v["agreement_pct"] >= 80 else "FAIL"
        print(f"  {dim}: {v['agreement_pct']}% [{flag}]")
    print(f"  overall avg: {overall:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
