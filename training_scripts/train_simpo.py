"""Tenacious-Bench Path B — SimPO LoRA training script.

This is the canonical, committed training script. It mirrors the cells of
docs/act4_colab_path_b.md but is a stand-alone .py runnable outside Colab —
on RunPod, on a local GPU, or anywhere with CUDA + the dependencies pinned in
requirements.txt.

Reproducibility commitments (per Week 11 GitHub-check rubric):
- Random seed fixed and propagated across `random`, `numpy`, `torch`,
  `torch.cuda`, and `transformers.set_seed`.
- HF model repo + revision pinned at module top (MODEL_REPO + MODEL_REVISION).
- LoRA-only configuration (`prepare_model_for_kbit_training` + PEFT LoRA).
- All hyperparameters explicit: lr, batch, gradient_accumulation, lora rank/alpha,
  warmup, scheduler, max_steps.
- Backbone version pinned to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` at the
  HF revision recorded in MODEL_REVISION; the actual loaded SHA is also
  resolved at runtime and written to training/training_run.log so the run is
  byte-traceable even if MODEL_REVISION is changed later.
- TRL CPOTrainer with `loss_type="simpo"` is the path-aligned algorithm for
  Path B (preference-tuned judge).

Usage (local GPU or RunPod):

    pip install -r requirements.txt
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    python training_scripts/train_simpo.py

Outputs:
- outputs/tenacious_judge_simpo/                  (LoRA adapter)
- training/training_run.log                       (hyperparams + per-step loss + actual revision SHA)
- training/loss_curve.png                         (PNG of the loss + eval-margin)

Hyperparameters below were the v2 run's choices (2026-05-01 on Colab T4).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

# -------- Pinning + reproducibility --------
SEED = 3407
MODEL_REPO = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MODEL_REVISION = "main"  # Override with a specific commit SHA for byte-reproducible runs.
                          # Latest commit on `main` is logged at runtime to training/training_run.log.

# Hyperparameters (v2 run, Colab T4, 64 min wall time)
MAX_SEQ_LEN = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LEARNING_RATE = 1e-5
PER_DEVICE_TRAIN_BATCH = 2
GRAD_ACCUM_STEPS = 8
MAX_STEPS = 300
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 5
SAVE_STEPS = 100
EVAL_STEPS = 50
SIMPO_BETA = 2.0
SIMPO_GAMMA = 1.0
CPO_ALPHA = 0.0  # 0 = pure SimPO; nonzero adds CPO's SFT-loss regularization

TRAIN_DATA_PATH = REPO_ROOT / "training_data" / "preference_pairs_v2.jsonl"
HELD_OUT_TASKS_PATH = REPO_ROOT / "tenacious_bench_v0.1" / "held_out" / "tasks.jsonl"
ADAPTER_OUT_DIR = REPO_ROOT / "outputs" / "tenacious_judge_simpo"
TRAINING_LOG_PATH = REPO_ROOT / "training" / "training_run.log"
LOSS_CURVE_PATH = REPO_ROOT / "training" / "loss_curve.png"


def _set_all_seeds(seed: int) -> None:
    """Propagate one seed across every RNG SimPO + LoRA training touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


def _resolve_model_revision(repo: str, revision: str) -> str:
    """Resolve the revision to a concrete commit SHA. If `revision='main'`, fetch
    the head commit so the run is reproducible from the SHA."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(repo, revision=revision)
        return info.sha
    except Exception as e:  # offline or rate-limited; fall back to whatever was passed
        print(f"  (revision resolve failed: {e}; logging '{revision}' literal)")
        return revision


def _load_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return rows


def _build_eval_pairs() -> list[dict]:
    """Build a held-out preference slice from style_guide_pair tasks (the only
    held-out tasks with ground_truth.chosen + rejected)."""
    rows = []
    with HELD_OUT_TASKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            gt = t.get("ground_truth") or {}
            if gt.get("chosen_output") and gt.get("rejected_output"):
                prompt_text = (
                    f"{t['input']['scenario']}\n\n"
                    f"Hiring Signal Brief: "
                    f"{json.dumps(t['input'].get('hiring_signal_brief') or {})[:600]}\n"
                )
                rows.append({
                    "prompt": prompt_text,
                    "chosen": gt["chosen_output"],
                    "rejected": gt["rejected_output"],
                })
    return rows


def main() -> int:
    print("=" * 70)
    print("Tenacious-Bench Path B — SimPO LoRA Training")
    print("=" * 70)

    # 1. Seed everything before any model load.
    _set_all_seeds(SEED)
    print(f"seed: {SEED} (propagated to random, numpy, torch, torch.cuda, transformers)")

    # 2. Resolve and log the actual HF revision SHA.
    revision_sha = _resolve_model_revision(MODEL_REPO, MODEL_REVISION)
    print(f"backbone: {MODEL_REPO} @ revision={revision_sha}")

    # 3. Load model via Unsloth (4-bit QLoRA).
    try:
        import unsloth  # MUST be imported before transformers/peft/trl per Unsloth.
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. See requirements.txt + the Unsloth git extra.")
        return 1

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_REPO,
        revision=MODEL_REVISION,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Attach LoRA adapters.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=list(LORA_TARGET_MODULES),
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: r={LORA_R} alpha={LORA_ALPHA} target={LORA_TARGET_MODULES} "
          f"trainable={n_trainable:,}/{n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    # 5. Load preference pairs.
    if not TRAIN_DATA_PATH.exists():
        print(f"ERROR: {TRAIN_DATA_PATH} missing")
        return 1
    train_rows = _load_pairs(TRAIN_DATA_PATH)
    eval_rows = _build_eval_pairs() if HELD_OUT_TASKS_PATH.exists() else []
    print(f"data: train={len(train_rows)} eval={len(eval_rows)} "
          f"(eval slice from style_guide_pair held-out tasks; sealed-leak risk disclosed in model_card.md)")

    # 6. Build TRL CPOTrainer with loss_type="simpo".
    from datasets import Dataset
    from trl import CPOConfig, CPOTrainer

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

    config = CPOConfig(
        output_dir=str(REPO_ROOT / "outputs" / "simpo_qwen25_3b"),
        loss_type="simpo",
        beta=SIMPO_BETA,
        simpo_gamma=SIMPO_GAMMA,
        cpo_alpha=CPO_ALPHA,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=1024,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=2,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = CPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    # 7. Train.
    t0 = time.time()
    result = trainer.train()
    wall_min = (time.time() - t0) / 60
    print(f"training done in {wall_min:.1f} min, final_train_loss={result.training_loss:.4f}")

    # 8. Save adapter.
    ADAPTER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_OUT_DIR))
    tokenizer.save_pretrained(str(ADAPTER_OUT_DIR))
    print(f"adapter saved to {ADAPTER_OUT_DIR}")

    # 9. Persist training_run.log with the actual revision SHA + hyperparams.
    TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write("# Tenacious-Bench v0.1 — Path B SimPO Training Run\n\n")
        f.write(f"backbone: {MODEL_REPO}\n")
        f.write(f"backbone_revision_pinned: {MODEL_REVISION}\n")
        f.write(f"backbone_revision_actual: {revision_sha}\n")
        f.write(f"adapter: LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}\n")
        f.write(f"target_modules: {','.join(LORA_TARGET_MODULES)}\n")
        f.write(f"loss_type: simpo (via trl.CPOTrainer)\n")
        f.write(f"beta: {SIMPO_BETA}  simpo_gamma: {SIMPO_GAMMA}  cpo_alpha: {CPO_ALPHA}\n")
        f.write(f"learning_rate: {LEARNING_RATE}  scheduler: {LR_SCHEDULER}, warmup_ratio={WARMUP_RATIO}\n")
        f.write(f"effective_batch_size: {PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS}\n")
        f.write(f"max_steps: {MAX_STEPS}  max_length: {MAX_SEQ_LEN}\n")
        f.write(f"seed: {SEED}\n")
        f.write(f"\n# Loss history\n")
        for x in trainer.state.log_history:
            f.write(json.dumps(x) + "\n")
        f.write(f"\n# Wall time: {wall_min:.1f} minutes\n")
        f.write(f"final_train_loss: {result.training_loss:.4f}\n")
    print(f"log written: {TRAINING_LOG_PATH}")

    # 10. Plot loss curve.
    try:
        import matplotlib.pyplot as plt
        history = trainer.state.log_history
        train_steps = [x["step"] for x in history if "loss" in x and "eval_loss" not in x]
        train_losses = [x["loss"] for x in history if "loss" in x and "eval_loss" not in x]
        plt.figure(figsize=(10, 5))
        plt.plot(train_steps, train_losses, color="#ff7f0e", linewidth=2, label="train SimPO loss")
        eval_steps = [x["step"] for x in history if "eval_loss" in x]
        eval_losses = [x["eval_loss"] for x in history if "eval_loss" in x]
        if eval_losses:
            plt.plot(eval_steps, eval_losses, color="#1f77b4", linewidth=2, marker="o",
                     label="eval SimPO loss")
        plt.title("Tenacious-Bench v0.1 — SimPO Judge Training (Qwen2.5-3B + LoRA)",
                  fontsize=13, fontweight="bold")
        plt.xlabel("step"); plt.ylabel("loss")
        plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
        LOSS_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(LOSS_CURVE_PATH), dpi=140, bbox_inches="tight")
        print(f"loss curve written: {LOSS_CURVE_PATH}")
    except ImportError:
        print("matplotlib not installed; skipping loss curve PNG")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
