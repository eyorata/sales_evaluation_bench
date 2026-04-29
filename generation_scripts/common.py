"""Shared types and constants for the four authoring-mode scripts."""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "reference"
DATASET_DIR = REPO_ROOT / "tenacious_bench_v0.1"

DIMENSIONS = [
    "icp_misclassification",
    "signal_over_claiming",
    "bench_over_commitment",
    "tone_drift",
    "multi_thread_leakage",
    "dual_control_coordination",
    "scheduling_edge_cases",
    "signal_confidence_alignment",
    "gap_over_claiming",
    "cost_pathology",
]

# Same Week 10 bench (frozen reference). Real bench available at
# reference/week10_bench_summary.json.
BENCH_TEMPLATE = {
    "stacks": {
        "python": {"available_engineers": 7},
        "go": {"available_engineers": 3},
        "data": {"available_engineers": 9},
        "ml": {"available_engineers": 5},
        "infra": {"available_engineers": 4},
        "frontend": {"available_engineers": 6},
        "fullstack_nestjs": {"available_engineers": 2, "note": "Committed on Modo Compass through Q3 2026."},
    },
    "leadership": {
        "fractional_architects_available": 2,
        "fractional_cto_available": 1,
    },
}

# Public-source window for time-shift contamination check.
PUBLIC_SOURCE_WINDOW = {
    "crunchbase_funding": "2025-11-01..2026-04-29",
    "layoffs_fyi": "2025-12-01..2026-04-29",
    "builtin_jobs": "2026-02-01..2026-04-29",
}

# Models for rotation (per methodology.md).
AUTHOR_MODELS = ["deepseek/deepseek-v3.2", "qwen/qwen3-next-80b-a3b-instruct"]
JUDGE_MODELS = ["qwen/qwen3-next-80b-a3b-instruct", "deepseek/deepseek-v3.2"]
THIRD_FAMILY_JUDGE = "meta-llama/llama-3.3-70b-instruct"


def task_id(n: int) -> str:
    return f"TB-{n:04d}"


def deterministic_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pick_pair(rng: random.Random, mode: str) -> tuple[str, str]:
    """Return (author_model, judge_model) such that the two come from different
    families. `mode` is one of: 'synthesis', 'programmatic', 'rewrite'."""
    if mode == "rewrite":
        return rng.choice(AUTHOR_MODELS), THIRD_FAMILY_JUDGE
    a = rng.choice(AUTHOR_MODELS)
    j = rng.choice([m for m in JUDGE_MODELS if not _same_family(m, a)])
    return a, j


def _same_family(a: str, b: str) -> bool:
    return a.split("/")[0] == b.split("/")[0]
