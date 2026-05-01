"""Publish Tenacious-Bench v0.1 to HuggingFace Hub.

The brief requires the held-out partition to be sealed at publish time:
> Held-out is in a separate file, gitignored from training scripts, and not
> committed in unencrypted form to the public repo. Sealed-slice tasks released
> only after the leaderboard is published.

This script:
1. Stages a fresh copy of `tenacious_bench_v0.1/` to a temp dir.
2. REMOVES the held_out/tasks.jsonl from the staged copy.
3. Adds a `held_out/SEALED.md` placeholder explaining the seal.
4. Verifies datasheet, contamination_check, composition.json are present.
5. Pushes to https://huggingface.co/datasets/<HF_USERNAME>/tenacious_bench_v0.1.
6. Tags the HF release as v0.1.

Required env vars:
- HF_TOKEN: write-scope HuggingFace token
- HF_USERNAME: your HuggingFace handle (default: eyorata)

Usage:
  python -m generation_scripts.publish_hf_dataset --dry-run   # plan only
  python -m generation_scripts.publish_hf_dataset             # actually push
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "tenacious_bench_v0.1"

# Load HF_TOKEN / HF_USERNAME from .env (re-uses the loader from openrouter_client).
sys.path.insert(0, str(REPO_ROOT))
from generation_scripts.openrouter_client import _load_env  # noqa: E402

_load_env()


REQUIRED_FILES = [
    "datasheet.md",
    "composition.json",
    "contamination_check.json",
    "judge_filter_log.jsonl",
    "judge_calibration.json",
    "train/tasks.jsonl",
    "dev/tasks.jsonl",
]

SEALED_README_TEMPLATE = """# held_out/ — sealed until leaderboard publish

The held-out partition (75 tasks) is **not** included in this HuggingFace dataset
release. Per the Week 11 brief and the dataset's stated contamination protocol,
the sealed slice is released only after a public leaderboard is established.

If you want to evaluate your agent against the held-out partition, please open
an issue at:

https://github.com/eyorata/sales_evaluation_bench

and we'll walk through running the scoring evaluator against your model with the
sealed tasks under a non-disclosure agreement on the labels.

The 12 ground-truth-bearing pairs (from the Tenacious Style Guide v2) form the
canonical preference-pair eval slice. The other 63 held-out tasks are
hand-authored adversarial, multi-LLM-synthesis, programmatic, and trace-derived.

Provenance and counts:

- 30 hand_authored_adversarial
- 16 multi_llm_synthesis
- 12 style_guide_pair (verbatim from Style Guide v2)
- 9 programmatic
- 8 trace_derived

License: CC-BY-4.0.
"""


def _stage(dest: Path) -> None:
    """Copy dataset to dest, drop held_out/tasks.jsonl, add SEALED.md."""
    dest.mkdir(parents=True, exist_ok=True)
    src = DATASET_DIR
    for item in src.iterdir():
        if item.name == "held_out":
            held_dst = dest / "held_out"
            held_dst.mkdir(exist_ok=True)
            (held_dst / "SEALED.md").write_text(SEALED_README_TEMPLATE, encoding="utf-8")
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name)
        else:
            shutil.copy2(item, dest / item.name)


def _verify_staging(staged: Path) -> list[str]:
    """Return list of missing files."""
    missing = []
    for rel in REQUIRED_FILES:
        if not (staged / rel).exists():
            missing.append(rel)
    if not (staged / "held_out" / "SEALED.md").exists():
        missing.append("held_out/SEALED.md")
    return missing


def _hf_dataset_card(staged: Path, hf_repo: str) -> str:
    comp = json.loads((staged / "composition.json").read_text(encoding="utf-8"))
    overall = comp.get("overall", {})
    total = overall.get("total", "?")
    per_dim = overall.get("by_dimension", {})
    per_mode = overall.get("by_source_mode", {})

    return f"""---
license: cc-by-4.0
language: en
task_categories:
- text-classification
- text-generation
size_categories:
- n<1K
tags:
- benchmark
- preference-pairs
- b2b-sales
- agent-evaluation
- tenacious-bench
---

# Tenacious-Bench v0.1

A 266-task evaluation benchmark for B2B sales-outreach agents, grounded in the
Tenacious (B2B engineering-outsourcing) workflow. 10 failure dimensions, 5
authoring source modes, mechanically-gradable rubric (no human in the loop).

**Held-out partition is sealed** — see `held_out/SEALED.md` for details.

## Dataset summary

- Total tasks (train + dev): {total - 75 if isinstance(total, int) else 'TBD'}
- Held-out (sealed): 75
- Source modes: {len(per_mode)}
- Failure dimensions: {len(per_dim)}
- License: CC-BY-4.0

## Quickstart

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ds = load_dataset("{hf_repo}", split="train")
print(f"first task:", ds[0]["task_id"], ds[0]["dimension"])
```

To run the scoring evaluator:

```bash
git clone https://github.com/eyorata/sales_evaluation_bench
cd sales_evaluation_bench
python scoring_evaluator.py --self-test
```

## Composition

| Source mode | n |
|---|---:|
{chr(10).join(f"| {k} | {v} |" for k, v in per_mode.items())}

| Dimension | n |
|---|---:|
{chr(10).join(f"| {k} | {v} |" for k, v in per_dim.items())}

## Citation

```bibtex
@dataset{{tenacious_bench_v01_2026,
  title  = {{Tenacious-Bench: a B2B sales-outreach evaluation benchmark}},
  author = {{Yorat, Eyoel and 10Academy TRP1 cohort}},
  year   = 2026, version = {{0.1}}, license = {{CC-BY-4.0}}, publisher = {{HuggingFace}}
}}
```

## Companion artifacts

- Trained Path B SimPO judge: https://huggingface.co/eyorata/tenacious-judge-simpo-qwen25-3b
- Source code + reproduction: https://github.com/eyorata/sales_evaluation_bench
- Blog post: see GitHub README for the latest URL.

See `datasheet.md` (Gebru + Pushkarna) for full provenance, motivation, collection,
preprocessing, uses, distribution, and maintenance details.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Stage and verify but don't push")
    p.add_argument("--hf-username", default=os.environ.get("HF_USERNAME", "eyorata"))
    p.add_argument("--repo-name", default="tenacious_bench_v0.1")
    args = p.parse_args()

    hf_repo = f"{args.hf_username}/{args.repo_name}"
    print(f"target HF repo: https://huggingface.co/datasets/{hf_repo}")

    if not DATASET_DIR.exists():
        print(f"ERROR: {DATASET_DIR} does not exist")
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / args.repo_name
        _stage(staged)

        # Write the HF-flavored README (with frontmatter)
        (staged / "README.md").write_text(_hf_dataset_card(staged, hf_repo), encoding="utf-8")

        missing = _verify_staging(staged)
        if missing:
            print("ERROR: staging is missing files:")
            for m in missing:
                print(f"  - {m}")
            return 1

        print(f"staged at: {staged}")
        print(f"  files: {sum(1 for _ in staged.rglob('*') if _.is_file())}")
        print(f"  size:  {sum(p.stat().st_size for p in staged.rglob('*') if p.is_file()) / 1024 / 1024:.2f} MB")
        print(f"  held_out/tasks.jsonl present? {(staged / 'held_out' / 'tasks.jsonl').exists()}  (must be False)")

        if args.dry_run:
            print("\nDRY RUN — not pushing. Re-run without --dry-run to publish.")
            return 0

        token = os.environ.get("HF_TOKEN")
        if not token:
            env_path = REPO_ROOT / ".env"
            print(f"ERROR: HF_TOKEN not set in environment or in {env_path}")
            if env_path.exists():
                lines = [ln.rstrip() for ln in env_path.read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.lstrip().startswith("#")]
                keys_seen = [ln.split("=", 1)[0].strip() for ln in lines if "=" in ln]
                print(f"  .env exists; non-commented keys seen: {keys_seen}")
                if "HF_TOKEN" not in keys_seen:
                    print("  -> HF_TOKEN line is missing or still commented out (begins with '#')")
            else:
                print("  .env does not exist at all")
            return 1
        if not token.startswith("hf_"):
            print(f"ERROR: HF_TOKEN value does not start with 'hf_'. Found prefix: {token[:5]!r}")
            print("  Likely cause: stray quote, trailing comma (turns string into tuple), or wrong token type.")
            return 1

        try:
            from huggingface_hub import HfApi, create_repo
        except ImportError:
            print("ERROR: pip install huggingface_hub first")
            return 1

        api = HfApi(token=token)
        create_repo(repo_id=hf_repo, repo_type="dataset", token=token, exist_ok=True)
        api.upload_folder(
            folder_path=str(staged),
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message="Tenacious-Bench v0.1 release (held-out sealed)",
        )
        print(f"\npushed: https://huggingface.co/datasets/{hf_repo}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
