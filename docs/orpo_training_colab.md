# Act IV (v3) — Path B ORPO Judge on Google Colab T4

**ORPO alternative run.** Same data, same backbone, same ablation structure as the SimPO run — but using ORPO (Odds-Ratio Preference Optimization, Hong et al. EMNLP 2024) instead. ORPO is a monolithic, reference-free method that folds the SFT and preference-alignment losses into a single training step, which can yield better calibration on small datasets. Run this if your SimPO Delta B was negative and you want to see whether ORPO flips it.

> **Why ORPO might do better:** ORPO penalizes the rejected response *within the same forward pass* as the SFT objective, rather than relying on a margin signal between two separate passes. On small preference datasets (n≈128) where the margin signal is noisy, this coupled loss can be more stable. See `synthesis_memos/orpo_memo.md` for the full argument.

---

## What you'll produce

| File | Lands at | Purpose |
|---|---|---|
| `training_run_orpo.log` | `training/training_run_orpo.log` | Hyperparameters + per-step loss |
| `ablation_results_orpo.json` | `ablations/ablation_results_orpo.json` | Delta A, Delta B, Delta C, Cost-Pareto with 95% CI |
| `held_out_traces_orpo.jsonl` | `ablations/held_out_traces_orpo.jsonl` | Per-task raw judge scores |
| `model_card_orpo.md` | `model_card_orpo.md` | Backbone, hyperparameters, intended use, limitations |
| HF model | `<your-username>/tenacious-judge-orpo-qwen25-3b` | LoRA adapter only |

---

## Prerequisites (verify before starting)

| Item | Pass condition |
|---|---|
| Colab T4 connected | Runtime → Change runtime type → T4 GPU. Run `!nvidia-smi` and confirm 16 GB. |
| HuggingFace write token | Get from https://huggingface.co/settings/tokens. Scope: **write**. Add as Colab Secret named `HF_TOKEN`. |
| Repo cloned | `git clone https://github.com/eyorata/sales_evaluation_bench.git` |
| Preference pairs present | `training_data/preference_pairs_v2.jsonl` has 128 lines |
| Held-out pairs available | `tenacious_bench_v0.1/held_out/tasks.jsonl` has ≥75 tasks; ≥12 carry `ground_truth.chosen_output` + `rejected_output` |

---

## How to use this file

1. Open a new notebook in Colab.
2. Copy each fenced code block into a separate cell, **in order**.
3. Run cells one at a time and check the printed output before proceeding.
4. Cells marked **(takes a while)** are the ones that warrant going to make coffee.

---

## Cell 1 — Install Unsloth + TRL + dependencies

```python
# ~3-4 minutes
!pip install --upgrade pip -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps xformers trl peft accelerate bitsandbytes datasets -q
!pip install matplotlib -q
print("install done")
```

Sanity check:

```python
import torch, trl, peft, transformers
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("trl:", trl.__version__, "peft:", peft.__version__, "transformers:", transformers.__version__)
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Verify ORPOTrainer is available in this TRL version
from trl import ORPOTrainer, ORPOConfig
print("ORPOTrainer: OK")
```

If `torch.cuda.is_available()` is `False`: switch the runtime to T4 and re-run.

---

## Cell 2 — Clone the Tenacious-Bench repo (data + scoring evaluator)

```python
import os
if not os.path.exists("/content/sales_evaluation_bench"):
    !git clone https://github.com/eyorata/sales_evaluation_bench.git /content/sales_evaluation_bench
else:
    !cd /content/sales_evaluation_bench && git fetch origin master && git reset --hard origin/master
%cd /content/sales_evaluation_bench

# Verify required files (v2: LLM-rewritten chosen outputs)
TRAIN_PATH = "training_data/preference_pairs_v2.jsonl"
assert os.path.exists(TRAIN_PATH), f"{TRAIN_PATH} missing — pull latest from origin/master"
assert os.path.exists("tenacious_bench_v0.1/held_out/tasks.jsonl"), "held_out/tasks.jsonl missing"

import json
n_train = sum(1 for _ in open(TRAIN_PATH, encoding="utf-8"))
n_held  = sum(1 for _ in open("tenacious_bench_v0.1/held_out/tasks.jsonl", encoding="utf-8"))
print(f"train preference pairs (v2): {n_train}  (expect 128)")
print(f"held-out tasks: {n_held}  (expect 75)")

sample = json.loads(open(TRAIN_PATH, encoding="utf-8").readline())
prov = sample.get("chosen_provenance", "MISSING")
assert "rewrite_v2" in prov, f"chosen_provenance is '{prov}', expected 'rewrite_v2' — wrong file?"
print(f"first-row chosen_provenance: {prov}  (rewrite_v2 confirmed)")
```

---

## Cell 3 — Load Qwen2.5-3B in 4-bit via Unsloth

```python
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 2048
BACKBONE    = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = BACKBONE,
    max_seq_length= MAX_SEQ_LEN,
    load_in_4bit  = True,
    dtype         = None,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("backbone loaded:", BACKBONE)
print("vocab:", len(tokenizer), "pad:", tokenizer.pad_token, "eos:", tokenizer.eos_token)
```

---

## Cell 4 — Initialize LoRA adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r          = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

You should see ~0.5–1.5% trainable.

---

## Cell 5 — Load the preference pairs

ORPO's `ORPOTrainer` uses the same `{prompt, chosen, rejected}` triple format as CPOTrainer/SimPO. No change to the data loading.

```python
import json
from datasets import Dataset

def load_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "prompt":   r["prompt"],
                "chosen":   r["chosen"],
                "rejected": r["rejected"],
            })
    return rows

train_rows = load_pairs("training_data/preference_pairs_v2.jsonl")
print(f"loaded {len(train_rows)} train pairs")

# Build held-out dev slice (ground_truth-bearing tasks: style_guide_pair slice + others)
held = [json.loads(l) for l in open("tenacious_bench_v0.1/held_out/tasks.jsonl", encoding="utf-8")]
dev_rows = []
for t in held:
    gt = t.get("ground_truth") or {}
    if gt.get("chosen_output") and gt.get("rejected_output"):
        prompt_text = (
            f"{t['input']['scenario']}\n\n"
            f"Hiring Signal Brief: {json.dumps(t['input'].get('hiring_signal_brief') or {})[:600]}\n"
        )
        dev_rows.append({
            "prompt":   prompt_text,
            "chosen":   gt["chosen_output"],
            "rejected": gt["rejected_output"],
        })
print(f"held-out eval pairs: {len(dev_rows)}")

train_ds = Dataset.from_list(train_rows)
dev_ds   = Dataset.from_list(dev_rows) if dev_rows else None

print("\nFirst train pair:")
print("  prompt[:160]:  ", train_rows[0]["prompt"][:160].replace("\n", " "))
print("  chosen[:120]:  ", train_rows[0]["chosen"][:120].replace("\n", " "))
print("  rejected[:120]:", train_rows[0]["rejected"][:120].replace("\n", " "))
```

---

## Cell 6 — Configure ORPO via TRL's `ORPOTrainer` **(takes a while)**

ORPO's key hyperparameter is `beta` (λ in the paper), the weight of the odds-ratio penalty term relative to the SFT cross-entropy loss. The paper reports β=0.1 as a stable default across many tasks. We tune up slightly to β=0.15 to give the preference signal a bit more weight on our small dataset.

**Key differences from SimPO:**
- Uses `ORPOConfig` + `ORPOTrainer` instead of `CPOConfig` + `CPOTrainer`.
- No `simpo_gamma` — ORPO uses an odds-ratio-based margin, not a length-normalized reward margin.
- No `cpo_alpha` — the SFT loss is built in to ORPO by design (it's the monolithic part).
- `beta` here scales the odds-ratio penalty; lower = more SFT-dominant, higher = more preference-dominant.

```python
from trl import ORPOConfig, ORPOTrainer

OUTPUT_DIR = "outputs/orpo_qwen25_3b"

config = ORPOConfig(
    output_dir              = OUTPUT_DIR,
    beta                    = 0.15,          # odds-ratio penalty weight (λ in the paper)
    learning_rate           = 8e-6,          # slightly lower than SimPO; ORPO is more sensitive
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.1,

    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_train_epochs            = 1,
    max_steps                   = 300,       # same as SimPO v2 run for fair comparison
    max_length                  = MAX_SEQ_LEN,
    max_prompt_length           = 1024,

    logging_steps    = 5,
    save_steps       = 100,
    save_total_limit = 2,

    bf16 = torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),

    eval_strategy = "steps" if dev_ds is not None else "no",
    eval_steps    = 50,
    per_device_eval_batch_size = 2,

    report_to              = "none",
    remove_unused_columns  = False,
    seed                   = 3407,
)

trainer = ORPOTrainer(
    model         = model,
    args          = config,
    train_dataset = train_ds,
    eval_dataset  = dev_ds,
    tokenizer     = tokenizer,
)

print("ORPOTrainer ready. Starting training...")
print(f"  train pairs: {len(train_ds)}  dev pairs: {len(dev_ds) if dev_ds else 0}")
print(f"  beta (odds-ratio weight): {config.beta}")
print(f"  max_steps: {config.max_steps}  effective batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
```

---

## Cell 7 — Train **(30–45 min on T4)**

```python
import time

t0 = time.time()
result = trainer.train()
wall_min = (time.time() - t0) / 60
print(f"\ntraining done in {wall_min:.1f} min")
print(f"final loss: {result.training_loss:.4f}")
```

**Watch for:**
- Loss should drop from ~0.7–0.9 toward 0.2–0.4 over the first 100 steps. ORPO loss is typically higher than SimPO loss in absolute magnitude because it includes the SFT cross-entropy component.
- If loss is flat after step 50: check that `train_rows[0]["chosen"]` and `train_rows[0]["rejected"]` differ meaningfully.
- If you hit OOM: drop `MAX_SEQ_LEN` to 1024 and `per_device_train_batch_size` to 1.

---

## Cell 8 — Plot loss + save `training_run_orpo.log`

```python
import matplotlib.pyplot as plt
import json, os

os.makedirs("training", exist_ok=True)

history     = trainer.state.log_history
train_steps = [x["step"] for x in history if "loss" in x and "eval_loss" not in x]
train_losses= [x["loss"] for x in history if "loss" in x and "eval_loss" not in x]
eval_steps  = [x["step"] for x in history if "eval_loss" in x]
eval_losses = [x["eval_loss"] for x in history if "eval_loss" in x]

plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_losses, color="#2ca02c", linewidth=2, label="train ORPO loss")
if eval_losses:
    plt.plot(eval_steps, eval_losses, color="#9467bd", linewidth=2, marker="o", label="eval ORPO loss")
plt.title("Tenacious-Bench v0.1 — ORPO Judge Training (Qwen2.5-3B + LoRA)", fontsize=13, fontweight="bold")
plt.xlabel("step"); plt.ylabel("loss")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
plt.savefig("training/loss_curve_orpo.png", dpi=140, bbox_inches="tight")
plt.show()

with open("training/training_run_orpo.log", "w", encoding="utf-8") as f:
    f.write("# Tenacious-Bench v0.1 — Path B ORPO Training Run\n\n")
    f.write(f"backbone: {BACKBONE}\n")
    f.write(f"adapter:  LoRA r=16, alpha=16, dropout=0\n")
    f.write(f"target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\n")
    f.write(f"loss_type: orpo (via trl.ORPOTrainer)\n")
    f.write(f"beta (odds-ratio weight): {config.beta}\n")
    f.write(f"learning_rate: {config.learning_rate}  scheduler: cosine, warmup_ratio={config.warmup_ratio}\n")
    f.write(f"effective_batch_size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}\n")
    f.write(f"max_steps: {config.max_steps}  max_length: {config.max_length}\n")
    f.write(f"seed: {config.seed}\n")
    f.write(f"\n# Loss history\n")
    for x in history:
        f.write(json.dumps(x) + "\n")
    f.write(f"\n# Wall time: {wall_min:.1f} minutes\n")
    f.write(f"final_train_loss: {result.training_loss:.4f}\n")

print("wrote training/training_run_orpo.log")
```

---

## Cell 9 — Save adapter locally + push to HuggingFace

**Setup (one-time):** in Colab's left sidebar, click the 🔑 (key icon) → **Secrets** → **Add new secret**. Name it `HF_TOKEN`, paste your write-scope token from https://huggingface.co/settings/tokens, toggle "Notebook access" ON.

```python
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN    = userdata.get("HF_TOKEN")
HF_USERNAME = "your-username"                 # <-- replace
HF_REPO     = f"{HF_USERNAME}/tenacious-judge-orpo-qwen25-3b"

assert HF_TOKEN and HF_TOKEN.startswith("hf_"), "HF_TOKEN secret missing or wrong format"
login(token=HF_TOKEN)

ADAPTER_PATH = "outputs/tenacious_judge_orpo"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print("local save done:", ADAPTER_PATH)

model.push_to_hub(HF_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
print("pushed to:", f"https://huggingface.co/{HF_REPO}")
```

---

## Cell 10 — Held-out preference accuracy (judge scoring helper)

Same scoring logic as the SimPO notebook: completion log-likelihood comparison.

```python
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model)

@torch.no_grad()
def completion_logprob(model, tokenizer, prompt, completion, length_normalize=True):
    """Return mean log-likelihood per completion token."""
    full = prompt + completion
    enc_full   = tokenizer(full, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)["input_ids"].shape[1]

    out    = model(**enc_full)
    logits = out.logits[0, :-1, :]
    labels = enc_full["input_ids"][0, 1:]

    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[prompt_len - 1:] = True

    logprobs  = torch.log_softmax(logits, dim=-1)
    chosen_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    chosen_lp = chosen_lp[mask]

    if chosen_lp.numel() == 0:
        return float("-inf")
    return (chosen_lp.mean() if length_normalize else chosen_lp.sum()).item()


def judge_pair(model, tokenizer, prompt, chosen, rejected):
    """Return (chosen_lp, rejected_lp, prefers_chosen)."""
    cl = completion_logprob(model, tokenizer, prompt, chosen)
    rl = completion_logprob(model, tokenizer, prompt, rejected)
    return cl, rl, cl > rl


if dev_rows:
    cl, rl, ok = judge_pair(model, tokenizer, dev_rows[0]["prompt"], dev_rows[0]["chosen"], dev_rows[0]["rejected"])
    print(f"smoke test → chosen_lp={cl:.4f}  rejected_lp={rl:.4f}  prefers_chosen={ok}")
```

---

## Cell 11 — Delta A: trained ORPO judge vs untrained-backbone baseline

```python
import gc

# 1. Score all held-out pairs with the TRAINED ORPO model
trained_records = []
for t in dev_rows:
    cl, rl, ok = judge_pair(model, tokenizer, t["prompt"], t["chosen"], t["rejected"])
    trained_records.append({
        "prompt_hash": hash(t["prompt"]) & 0xffffffff,
        "chosen_lp": cl, "rejected_lp": rl, "prefers_chosen": int(ok),
    })

trained_acc = sum(r["prefers_chosen"] for r in trained_records) / len(trained_records)
print(f"trained ORPO judge accuracy: {trained_acc:.3f} ({sum(r['prefers_chosen'] for r in trained_records)}/{len(trained_records)})")

# 2. Load fresh untrained backbone for comparison
del model, trainer
gc.collect()
torch.cuda.empty_cache()

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name    = BACKBONE,
    max_seq_length= MAX_SEQ_LEN,
    load_in_4bit  = True,
    dtype         = None,
)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token
FastLanguageModel.for_inference(base_model)

base_records = []
for t in dev_rows:
    cl, rl, ok = judge_pair(base_model, base_tokenizer, t["prompt"], t["chosen"], t["rejected"])
    base_records.append({
        "prompt_hash": hash(t["prompt"]) & 0xffffffff,
        "chosen_lp": cl, "rejected_lp": rl, "prefers_chosen": int(ok),
    })

base_acc = sum(r["prefers_chosen"] for r in base_records) / len(base_records)
print(f"untrained-backbone baseline accuracy: {base_acc:.3f}")
print(f"delta A (raw): {trained_acc - base_acc:+.3f} pp")
```

---

## Cell 12 — Delta B: trained ORPO judge vs prompt-engineered same-backbone

```python
TONE_JUDGE_SYSTEM = (
    "You score B2B sales-outreach drafts for Tenacious. The five tone markers are:\n"
    "1. Direct (subject states intent; no filler; one ask; ≤120 words cold)\n"
    "2. Grounded (claims tied to specific signals from brief; confidence-aware)\n"
    "3. Honest (refuses unsupported claims; no over-commit; no fabrication)\n"
    "4. Professional (no banned phrases: 'top talent', 'world-class', 'rockstars', 'A-players', 'bench' externally; no consultant jargon)\n"
    "5. Non-condescending (frames gaps as research findings, not failures)\n\n"
    "Given a SCENARIO and two candidate drafts A and B, output a single character: 'A' if A is the better Tenacious draft, 'B' if B is. Output nothing else."
)

@torch.no_grad()
def prompt_judge_pair(model, tokenizer, scenario, draft_a, draft_b):
    msg = (
        f"<|im_start|>system\n{TONE_JUDGE_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nSCENARIO:\n{scenario[:800]}\n\n"
        f"DRAFT A:\n{draft_a[:1200]}\n\n"
        f"DRAFT B:\n{draft_b[:1200]}\n\nWhich is better? Output only 'A' or 'B'.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN - 4).to(model.device)
    out    = model.generate(**inputs, max_new_tokens=2, do_sample=False, temperature=0.0)
    text   = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return text[:1]


import random
random.seed(3407)
prompt_records = []
for t in dev_rows:
    swap = random.random() < 0.5
    a_text = t["rejected"] if swap else t["chosen"]
    b_text = t["chosen"]   if swap else t["rejected"]
    pick   = prompt_judge_pair(base_model, base_tokenizer, t["prompt"], a_text, b_text)
    prefers_chosen = int((pick == "B") if swap else (pick == "A"))
    prompt_records.append({"prompt_hash": hash(t["prompt"]) & 0xffffffff, "pick": pick, "swap": swap, "prefers_chosen": prefers_chosen})

prompt_acc = sum(r["prefers_chosen"] for r in prompt_records) / len(prompt_records)
print(f"prompt-engineered baseline accuracy: {prompt_acc:.3f}")
print(f"delta B (raw): {trained_acc - prompt_acc:+.3f} pp")
```

---

## Cell 13 — Cost-Pareto

```python
import time

def time_n(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.time(); fn(); ts.append(time.time() - t0)
    return sum(ts) / n

sample = dev_rows[0]

del base_model, base_tokenizer
gc.collect(); torch.cuda.empty_cache()

trained_model_eval, trained_tok_eval = FastLanguageModel.from_pretrained(
    model_name    = ADAPTER_PATH,
    max_seq_length= MAX_SEQ_LEN,
    load_in_4bit  = True,
)
if trained_tok_eval.pad_token is None:
    trained_tok_eval.pad_token = trained_tok_eval.eos_token
FastLanguageModel.for_inference(trained_model_eval)

trained_lat = time_n(lambda: judge_pair(trained_model_eval, trained_tok_eval, sample["prompt"], sample["chosen"], sample["rejected"]))

del trained_model_eval, trained_tok_eval
gc.collect(); torch.cuda.empty_cache()

base_model2, base_tok2 = FastLanguageModel.from_pretrained(
    model_name    = BACKBONE,
    max_seq_length= MAX_SEQ_LEN,
    load_in_4bit  = True,
)
if base_tok2.pad_token is None:
    base_tok2.pad_token = base_tok2.eos_token
FastLanguageModel.for_inference(base_model2)

prompt_lat = time_n(lambda: prompt_judge_pair(base_model2, base_tok2, sample["prompt"], sample["chosen"], sample["rejected"]))

RUNPOD_4090_USD_PER_HR = 0.34
trained_cost = trained_lat * RUNPOD_4090_USD_PER_HR / 3600
prompt_cost  = prompt_lat  * RUNPOD_4090_USD_PER_HR / 3600

print(f"ORPO trained-judge latency: {trained_lat*1000:.1f} ms / judgment   (~${trained_cost:.5f})")
print(f"prompt-judge       latency: {prompt_lat*1000:.1f} ms / judgment   (~${prompt_cost:.5f})")
```

---

## Cell 14 — Paired bootstrap (95% CI on Delta A and Delta B)

```python
import numpy as np

def paired_bootstrap(a, b, n=10000, seed=42):
    """a, b: lists of 0/1 correctness for the same N pairs in the same order."""
    rng  = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(a), size=len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low":    float(np.percentile(diffs,  2.5)),
        "ci_high":   float(np.percentile(diffs, 97.5)),
        "p_value":   float((diffs <= 0).mean()),
    }

a_correct = [r["prefers_chosen"] for r in trained_records]
b_correct = [r["prefers_chosen"] for r in base_records]
p_correct = [r["prefers_chosen"] for r in prompt_records]

delta_a = paired_bootstrap(a_correct, b_correct)
delta_b = paired_bootstrap(a_correct, p_correct)

print("Delta A (ORPO-trained vs untrained backbone):", json.dumps(delta_a, indent=2))
print("Delta B (ORPO-trained vs prompt-engineered): ", json.dumps(delta_b, indent=2))
```

**Interpretation:** if `ci_low > 0` AND `p_value < 0.05`, the lift is statistically significant. Compare Delta A and Delta B here against the SimPO run in `ablations/ablation_results.json` to decide which algorithm to deploy.

---

## Cell 15 — Write `ablation_results_orpo.json`

```python
import os
os.makedirs("ablations", exist_ok=True)

WEEK10_TAU2_PASS_AT_1    = 0.7267
WEEK10_TAU2_CI           = [0.6504, 0.7917]
WEEK10_TAU2_BASELINE_COMMIT = "d11a97072c49d093f7b5a3e4fe9da95b490d43ba"

ablation = {
    "version":   "v0.1-orpo",
    "backbone":  BACKBONE,
    "algorithm": "orpo",
    "training": {
        "loss_type":           "orpo (trl.ORPOTrainer)",
        "beta":                config.beta,
        "learning_rate":       config.learning_rate,
        "max_steps":           config.max_steps,
        "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        "wall_time_min":       round(wall_min, 1),
        "final_train_loss":    round(result.training_loss, 4),
    },
    "held_out": {
        "n_pairs": len(dev_rows),
        "source_modes_eligible": "tasks with ground_truth.chosen_output AND rejected_output (style_guide_pair slice + others)",
    },
    "delta_a": {
        "name":         "ORPO-trained vs Week-10-baseline (untrained backbone, same scale)",
        "trained_acc":  round(trained_acc, 4),
        "baseline_acc": round(base_acc, 4),
        "raw_lift_pp":  round(trained_acc - base_acc, 4),
        **{f"bootstrap_{k}": v for k, v in delta_a.items()},
        "verdict": (
            "POSITIVE_SIGNIFICANT"     if delta_a["ci_low"] > 0 and delta_a["p_value"] < 0.05
            else "POSITIVE_NOT_SIGNIFICANT" if delta_a["mean_diff"] > 0
            else "FLAT_OR_NEGATIVE"
        ),
    },
    "delta_b": {
        "name":       "ORPO-trained vs prompt-engineered same-backbone",
        "trained_acc": round(trained_acc, 4),
        "prompt_acc":  round(prompt_acc, 4),
        "raw_lift_pp": round(trained_acc - prompt_acc, 4),
        **{f"bootstrap_{k}": v for k, v in delta_b.items()},
        "verdict": (
            "POSITIVE_SIGNIFICANT"     if delta_b["ci_low"] > 0 and delta_b["p_value"] < 0.05
            else "POSITIVE_NOT_SIGNIFICANT" if delta_b["mean_diff"] > 0
            else "FLAT_OR_NEGATIVE"
        ),
        "honest_note": "Per the brief: a flat or negative Delta B is a legitimate, publishable finding. Report as-is.",
    },
    "delta_c_informational": {
        "name":                    "Week 10 τ²-Bench retail (informational only, NO re-run per brief)",
        "week10_pass_at_1":        WEEK10_TAU2_PASS_AT_1,
        "week10_ci":               WEEK10_TAU2_CI,
        "week10_baseline_commit":  WEEK10_TAU2_BASELINE_COMMIT,
        "interpretation": (
            "Tenacious-Bench is a different distribution; ORPO was not trained on retail tasks. "
            "Delta-C lift is not expected or claimed. Tenacious-Bench lift is Tenacious-specific by construction."
        ),
    },
    "cost_pareto": {
        "trained_judge_latency_ms": round(trained_lat * 1000, 1),
        "prompt_judge_latency_ms":  round(prompt_lat * 1000, 1),
        "trained_cost_usd_per_judgment_at_runpod_4090": round(trained_cost, 6),
        "prompt_cost_usd_per_judgment_at_runpod_4090":  round(prompt_cost, 6),
        "training_one_time_cost_usd": 0.0,
    },
}

with open("ablations/ablation_results_orpo.json", "w", encoding="utf-8") as f:
    json.dump(ablation, f, indent=2)
print("wrote ablations/ablation_results_orpo.json")

trace_rows = []
for i, t in enumerate(dev_rows):
    trace_rows.append({
        "pair_idx":          i,
        "trained_orpo":      trained_records[i],
        "untrained":         base_records[i],
        "prompt_engineered": prompt_records[i],
    })
with open("ablations/held_out_traces_orpo.jsonl", "w", encoding="utf-8") as f:
    for r in trace_rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote ablations/held_out_traces_orpo.jsonl ({len(trace_rows)} rows)")
```

---

## Cell 16 — Write `model_card_orpo.md`

```python
model_card = f"""---
license: cc-by-4.0
base_model: {BACKBONE}
tags:
- preference-optimization
- orpo
- judge-model
- b2b-sales
- tenacious-bench
datasets:
- {HF_USERNAME}/tenacious_bench_v0.1
---

# Tenacious-Judge ORPO (Qwen2.5-3B + LoRA)

A small preference-tuned judge (critic) for Tenacious-style B2B sales outreach. Trained via ORPO on 128 (chosen, rejected) pairs from Tenacious-Bench v0.1. Compare with the SimPO adapter at `{HF_USERNAME}/tenacious-judge-simpo-qwen25-3b`.

## Intended use

Deployed as a **rejection-sampling layer** in front of a Tenacious sales-outreach generator. The generator produces a draft; this judge scores the (chosen, candidate) pair via completion log-likelihood; the orchestrator routes back to regeneration if the judge prefers a known-bad pattern.

## Algorithm rationale (ORPO vs SimPO)

ORPO (Hong et al., EMNLP 2024) folds the SFT cross-entropy and preference odds-ratio penalty into a single training forward pass. This eliminates the need for a reference model (like DPO) and avoids the length-biased margin term in SimPO. On small datasets (n≈128), ORPO's coupled SFT+preference objective can yield better calibration because the model never fully forgets how to generate fluent output in the process of learning the preference boundary.

## Training

- Backbone: `{BACKBONE}` (4-bit QLoRA via Unsloth)
- Algorithm: ORPO (TRL `ORPOTrainer`)
- Hyperparameters: β={config.beta} (odds-ratio weight), lr={config.learning_rate}, cosine scheduler, warmup {config.warmup_ratio}, effective batch {config.per_device_train_batch_size * config.gradient_accumulation_steps}, max_steps={config.max_steps}
- Wall time: {wall_min:.1f} min on Colab T4 (free)
- Training data: 128 preference pairs from `tenacious_bench_v0.1/train` (Tenacious-Bench v0.1, v2 LLM-rewritten chosen outputs)

## Held-out evaluation

| Metric | ORPO-trained | Untrained backbone | Prompt-engineered same backbone |
|---|---:|---:|---:|
| Held-out preference accuracy | {trained_acc:.3f} | {base_acc:.3f} | {prompt_acc:.3f} |

- Delta A: {trained_acc - base_acc:+.3f} pp (95% CI [{delta_a['ci_low']:.3f}, {delta_a['ci_high']:.3f}], p={delta_a['p_value']:.4f})
- Delta B: {trained_acc - prompt_acc:+.3f} pp (95% CI [{delta_b['ci_low']:.3f}, {delta_b['ci_high']:.3f}], p={delta_b['p_value']:.4f})

## Limitations

- Held-out eval slice is small (n={len(dev_rows)}) — wide CI bands apply to conclusions.
- Judge is trained on Tenacious-specific failure modes; **not** a general B2B preference scorer.
- ORPO's coupled SFT loss means lower β may produce outputs that are fluent but less preference-sharp; higher β risks training instability on small data.

## Citation

```bibtex
@misc{{tenacious_judge_orpo_2026,
  title={{Tenacious-Judge ORPO: a small preference-tuned judge for B2B sales outreach}},
  author={{Nebiyu, Eyoel}},
  year={{2026}},
  howpublished={{HuggingFace}},
  license={{CC-BY-4.0}}
}}
```
"""

with open("model_card_orpo.md", "w", encoding="utf-8") as f:
    f.write(model_card)
print("wrote model_card_orpo.md")
```

---

## Cell 17 — Download outputs + push to GitHub

```python
# Zip the committable artifacts (logs, results, model card — NOT the adapter weights)
!rm -f /content/act4_orpo_outputs.zip
!cd /content/sales_evaluation_bench && zip -r /content/act4_orpo_outputs.zip \
    training/training_run_orpo.log \
    training/loss_curve_orpo.png \
    ablations/ablation_results_orpo.json \
    ablations/held_out_traces_orpo.jsonl \
    model_card_orpo.md

!ls -lh /content/act4_orpo_outputs.zip

# Download to browser
from google.colab import files
files.download("/content/act4_orpo_outputs.zip")
```

After downloading: unzip into your local repo, then from your laptop:

```bash
cd sales_evaluation_bench
git add training/training_run_orpo.log training/loss_curve_orpo.png \
        ablations/ablation_results_orpo.json ablations/held_out_traces_orpo.jsonl \
        model_card_orpo.md
git commit -m "results: ORPO v0.1 training run — Delta A/B/C + Cost-Pareto"
git push
```

---

## Quick comparison table (fill in after running)

| Metric | SimPO (v2) | ORPO (v3) | Winner |
|---|---:|---:|---|
| Delta A (trained vs untrained) | 0.417 | — | — |
| Delta B (trained vs prompt-eng) | −0.42 | — | — |
| Final train loss | — | — | — |
| Wall time (min, T4) | ~55 | — | — |
| Latency per judgment (ms) | 417 | — | — |

Update this table after Cell 15 and commit it to `docs/orpo_training_colab.md` so the comparison lives next to the runs.