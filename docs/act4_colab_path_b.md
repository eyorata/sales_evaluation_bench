# Act IV — Path B (SimPO Judge) on Google Colab T4

**One core training run + three ablations + one cost measurement.** Total wall time: 60–90 min. Cost: $0 (Colab T4 is free; HF push is free).

This notebook trains a small Tenacious-specific *judge* (preference-tuned critic) per the Week 11 brief's Path B. The judge will be deployed in production as a rejection-sampling layer in front of the Week 10 generator.

> **Path-A note:** The reference notebook you have (`trp1_week11_unsloth.py`) is for Path A (SFT) — it uses `SFTTrainer` and the `{instruction, input, output}` format. **Ignore those cells.** Path B uses `CPOTrainer` with `loss_type="simpo"` and the `{prompt, chosen, rejected}` format, which is what your `training_data/preference_pairs.jsonl` already has.

---

## What you'll produce (Act IV deliverables)

| File | Lands at | Purpose |
|---|---|---|
| `training_run.log` | `training/training_run.log` | Hyperparameters + per-step loss |
| `ablation_results.json` | `ablations/ablation_results.json` | Delta A, Delta B, Delta C, Cost-Pareto with 95% CI |
| `held_out_traces.jsonl` | `ablations/held_out_traces.jsonl` | Per-task raw judge scores |
| `model_card.md` | `model_card.md` | Backbone, hyperparameters, intended use, limitations |
| HF model | `<your-username>/tenacious-judge-simpo-qwen25` | LoRA adapter only |

---

## Prerequisites (Day 0 — verify before you start the run)

| Item | Pass condition |
|---|---|
| Colab T4 connected | Runtime → Change runtime type → T4 GPU. Run `!nvidia-smi` and confirm 16 GB. |
| HuggingFace write token | Get from https://huggingface.co/settings/tokens. Scope: **write**. |
| Repo cloned | `git clone https://github.com/eyorata/sales_evaluation_bench.git` |
| Preference pairs present | `training_data/preference_pairs.jsonl` has 128 lines |
| Held-out pairs available | `tenacious_bench_v0.1/held_out/tasks.jsonl` has 75 tasks; ≥12 carry `ground_truth.chosen_output` + `rejected_output` (the style_guide_pair slice) |
| OpenRouter key (optional) | Only needed if you also want to run the dev-tier prompt-baseline arm of Delta B via API. The local-Qwen prompt baseline is free and is what this notebook uses. |

If any item fails, fix it before launching the run. The brief explicitly says: *if it is not converging by 30 minutes, kill it and check your data — do not throw more compute at it.*

---

## How to use this file

1. Open a new notebook in Colab.
2. Copy each fenced code block below into a separate cell, **in order**.
3. Run cells one at a time and check the printed output before proceeding.
4. Cells marked **(takes a while)** are the ones that warrant going to make coffee.

---

## Cell 1 — Install Unsloth + TRL + dependencies

```python
# ~3 minutes
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
```

If `torch.cuda.is_available()` is `False`: switch the runtime to T4 and re-run.

---

## Cell 2 — Clone the Tenacious-Bench repo (data + scoring evaluator)

```python
import os
if not os.path.exists("/content/sales_evaluation_bench"):
    !git clone https://github.com/eyorata/sales_evaluation_bench.git /content/sales_evaluation_bench
%cd /content/sales_evaluation_bench

# Verify required files (v2: real LLM-rewritten chosen outputs)
TRAIN_PATH = "training_data/preference_pairs_v2.jsonl"
assert os.path.exists(TRAIN_PATH), f"{TRAIN_PATH} missing — pull latest from origin/master"
assert os.path.exists("tenacious_bench_v0.1/held_out/tasks.jsonl"), "held_out/tasks.jsonl missing"

import json
n_train = sum(1 for _ in open(TRAIN_PATH, encoding="utf-8"))
n_held = sum(1 for _ in open("tenacious_bench_v0.1/held_out/tasks.jsonl", encoding="utf-8"))
print(f"train preference pairs (v2): {n_train}  (expect 128)")
print(f"held-out tasks: {n_held}  (expect 75)")

# Sanity-check: confirm chosen outputs were rewritten by Llama-3.3-70B (not the v1 templates)
sample = json.loads(open(TRAIN_PATH, encoding="utf-8").readline())
prov = sample.get("chosen_provenance", "MISSING")
assert "rewrite_v2" in prov, f"chosen_provenance is '{prov}', expected 'rewrite_v2' — wrong file?"
print(f"first-row chosen_provenance: {prov}  (rewrite_v2 confirmed)")
```

---

## Cell 3 — Load Qwen2.5-3B in 4-bit (upgraded from 1.5B for v2 run)

**v2 update (2026-05-01):** the v1 run with Qwen2.5-1.5B trained perfectly on the templated preference data (train accuracy 1.0) but scored 25% on the held-out style-guide pairs — the model overfit the template, not the tone. v2 fixes both halves: (a) chosen outputs are now LLM-rewritten in real Tenacious voice via Llama-3.3-70B with the v2 style guide as context (`training_data/preference_pairs_v2.jsonl`), and (b) backbone is bumped to **Qwen2.5-3B-Instruct-bnb-4bit** for more capacity to learn the 5-marker discrimination. 3B fits T4 with margin (~7 GB VRAM in 4-bit), trains in ~50–60 min, and is the closest match to the brief's permitted Qwen 3.5 4B class.

```python
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 2048  # covers the longest Tenacious draft (style-guide pair worst case ~1100 chars)
BACKBONE = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BACKBONE,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
    dtype = None,
)

# Set pad token if missing (Qwen models often have eos but no pad)
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
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Confirm LoRA is wired
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

You should see ~0.5–1.5% trainable.

---

## Cell 5 — Load the preference pairs (no SFT formatting; SimPO uses raw triples)

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

train_rows = load_pairs("training_data/preference_pairs_v2.jsonl")  # v2: real LLM-rewritten chosens
print(f"loaded {len(train_rows)} train pairs")

# Build a small held-out dev slice for eval-during-training: use the 12 style-guide pairs
# from held_out (which carry real ground_truth) plus any other held_out task with chosen+rejected.
held = [json.loads(l) for l in open("tenacious_bench_v0.1/held_out/tasks.jsonl", encoding="utf-8")]
dev_rows = []
for t in held:
    gt = t.get("ground_truth") or {}
    if gt.get("chosen_output") and gt.get("rejected_output"):
        # Same prompt shape as training
        prompt_text = (
            f"{t['input']['scenario']}\n\n"
            f"Hiring Signal Brief: {json.dumps(t['input'].get('hiring_signal_brief') or {})[:600]}\n"
        )
        dev_rows.append({
            "prompt": prompt_text,
            "chosen": gt["chosen_output"],
            "rejected": gt["rejected_output"],
        })
print(f"held-out eval pairs: {len(dev_rows)}")

train_ds = Dataset.from_list(train_rows)
dev_ds   = Dataset.from_list(dev_rows) if dev_rows else None

# Sanity print
print("\nFirst train pair:")
print("  prompt[:160]:  ", train_rows[0]["prompt"][:160].replace("\n", " "))
print("  chosen[:120]:  ", train_rows[0]["chosen"][:120].replace("\n", " "))
print("  rejected[:120]:", train_rows[0]["rejected"][:120].replace("\n", " "))
```

---

## Cell 6 — Configure SimPO via TRL's CPOTrainer **(takes a while)**

TRL's `CPOTrainer` with `loss_type="simpo"` is the canonical SimPO entry point in 2026. SimPO's hyperparameters that matter:

- `beta = 2.0` — preference margin scaling. SimPO recommended.
- `simpo_gamma = 1.0` — target reward margin between chosen and rejected.
- `cpo_alpha = 0.0` — disable the SFT regularization term that CPO adds (pure SimPO has no SFT term).
- `learning_rate = 8e-6` — standard for preference fine-tuning at this scale.

```python
from trl import CPOConfig, CPOTrainer
import torch

OUTPUT_DIR = "outputs/simpo_qwen25_15b"

config = CPOConfig(
    output_dir              = OUTPUT_DIR,
    loss_type               = "simpo",
    beta                    = 2.0,
    simpo_gamma             = 1.0,
    cpo_alpha               = 0.0,
    learning_rate           = 1e-5,               # v2: bumped from 8e-6 for 3B + real data
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.1,

    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,
    num_train_epochs            = 1,
    max_steps                   = 300,            # v2: bumped from 200; ~50-60 min on T4
    max_length                  = MAX_SEQ_LEN,
    max_prompt_length           = 1024,

    logging_steps = 5,
    save_steps    = 100,
    save_total_limit = 2,

    bf16 = torch.cuda.is_bf16_supported(),
    fp16 = not torch.cuda.is_bf16_supported(),

    eval_strategy = "steps" if dev_ds is not None else "no",
    eval_steps    = 50,
    per_device_eval_batch_size = 2,

    report_to = "none",
    remove_unused_columns = False,
    seed = 3407,
)

trainer = CPOTrainer(
    model         = model,
    args          = config,
    train_dataset = train_ds,
    eval_dataset  = dev_ds,
    tokenizer     = tokenizer,
)

print("CPOTrainer ready. Starting training...")
print(f"  train pairs: {len(train_ds)}  dev pairs: {len(dev_ds) if dev_ds else 0}")
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
- Loss should drop from ~0.6 toward 0.2–0.3 over the first 100 steps. If it's flat after step 50, kill the cell and inspect `train_rows[0]` — your data is the problem, not the algorithm.
- If you hit OOM: drop `MAX_SEQ_LEN` to 1024 and `per_device_train_batch_size` to 1.

---

## Cell 8 — Plot loss + save `training_run.log`

```python
import matplotlib.pyplot as plt
import json
import os

os.makedirs("training", exist_ok=True)   # must exist before plt.savefig

history = trainer.state.log_history
train_steps = [x["step"] for x in history if "loss" in x and "eval_loss" not in x]
train_losses = [x["loss"] for x in history if "loss" in x and "eval_loss" not in x]
eval_steps   = [x["step"] for x in history if "eval_loss" in x]
eval_losses  = [x["eval_loss"] for x in history if "eval_loss" in x]

plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_losses, color="#ff7f0e", linewidth=2, label="train SimPO loss")
if eval_losses:
    plt.plot(eval_steps, eval_losses, color="#1f77b4", linewidth=2, marker="o", label="eval SimPO loss")
plt.title("Tenacious-Bench v0.1 — SimPO Judge Training (Qwen2.5-3B + LoRA)", fontsize=13, fontweight="bold")
plt.xlabel("step"); plt.ylabel("loss")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend()
plt.savefig("training/loss_curve.png", dpi=140, bbox_inches="tight")
plt.show()

# Persist training_run.log
with open("training/training_run.log", "w", encoding="utf-8") as f:
    f.write("# Tenacious-Bench v0.1 — Path B SimPO Training Run\n\n")
    f.write(f"backbone: {BACKBONE}\n")
    f.write(f"adapter:  LoRA r=16, alpha=16, dropout=0\n")
    f.write(f"target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\n")
    f.write(f"loss_type: simpo (via trl.CPOTrainer)\n")
    f.write(f"beta: {config.beta}  simpo_gamma: {config.simpo_gamma}  cpo_alpha: {config.cpo_alpha}\n")
    f.write(f"learning_rate: {config.learning_rate}  scheduler: cosine, warmup_ratio={config.warmup_ratio}\n")
    f.write(f"effective_batch_size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}\n")
    f.write(f"max_steps: {config.max_steps}  max_length: {config.max_length}\n")
    f.write(f"seed: {config.seed}\n")
    f.write(f"\n# Loss history\n")
    for x in history:
        f.write(json.dumps(x) + "\n")
    f.write(f"\n# Wall time: {wall_min:.1f} minutes\n")
    f.write(f"final_train_loss: {result.training_loss:.4f}\n")

print("wrote training/training_run.log")
```

---

## Cell 9 — Save adapter locally + push to HuggingFace

**Setup (one-time):** in Colab's left sidebar, click the 🔑 (key icon) → **Secrets** → **Add new secret**. Name it `HF_TOKEN`, paste your write-scope token from https://huggingface.co/settings/tokens, toggle "Notebook access" ON. Never paste the token directly into a cell — secrets are scoped to the runtime and survive restarts; pasted tokens leak into logs and version history.

```python
from huggingface_hub import login
from google.colab import userdata

HF_TOKEN    = userdata.get("HF_TOKEN")        # pulled from Colab secrets
HF_USERNAME = "your-username"                 # <-- replace
HF_REPO     = f"{HF_USERNAME}/tenacious-judge-simpo-qwen25-3b"

assert HF_TOKEN and HF_TOKEN.startswith("hf_"), "HF_TOKEN secret missing or wrong format"
login(token=HF_TOKEN)

ADAPTER_PATH = "outputs/tenacious_judge_simpo"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print("local save done:", ADAPTER_PATH)

model.push_to_hub(HF_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
print("pushed to:", f"https://huggingface.co/{HF_REPO}")
```

> **Token-handling rule:** if you ever paste a real `hf_...` token into a cell or chat (which makes it a tuple-bug magnet too — a stray trailing comma turns it into a tuple and `login()` errors with `'tuple' object has no attribute 'startswith'`), revoke it at https://huggingface.co/settings/tokens, generate a fresh one, and switch to Colab Secrets before continuing.

---

## Cell 10 — Held-out preference accuracy (judge scoring helper)

The held-out judge accuracy = % of pairs where the trained judge prefers `chosen` over `rejected`. We compute the **completion log-likelihood** under the model — SimPO's reward signal at training time IS this normalized log-likelihood, so this is the natural eval metric.

```python
from unsloth import FastLanguageModel

# Switch to inference mode (Unsloth optimization)
FastLanguageModel.for_inference(model)

@torch.no_grad()
def completion_logprob(model, tokenizer, prompt, completion, length_normalize=True):
    """Return mean log-likelihood per completion token (SimPO-style)."""
    full = prompt + completion
    enc_full   = tokenizer(full, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)["input_ids"].shape[1]

    out    = model(**enc_full)
    logits = out.logits[0, :-1, :]              # predicting next token
    labels = enc_full["input_ids"][0, 1:]       # shifted

    # mask: only score completion tokens
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[prompt_len-1:] = True

    logprobs = torch.log_softmax(logits, dim=-1)
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


# Smoke test on the first held-out pair
if dev_rows:
    cl, rl, ok = judge_pair(model, tokenizer, dev_rows[0]["prompt"], dev_rows[0]["chosen"], dev_rows[0]["rejected"])
    print(f"smoke test → chosen_lp={cl:.4f}  rejected_lp={rl:.4f}  prefers_chosen={ok}")
```

---

## Cell 11 — Delta A: trained judge vs untrained-backbone baseline

The Week 10 baseline for the judge component is "no judge." For a measurable comparison the brief asks for, the equivalent is: **same backbone, no LoRA**. We load it fresh, run the same scoring loop, and compare.

```python
import gc, json, time

# 1. Score all held-out pairs with the TRAINED model (already loaded)
trained_records = []
for t in dev_rows:
    cl, rl, ok = judge_pair(model, tokenizer, t["prompt"], t["chosen"], t["rejected"])
    trained_records.append({
        "prompt_hash": hash(t["prompt"]) & 0xffffffff,
        "chosen_lp": cl, "rejected_lp": rl, "prefers_chosen": int(ok),
    })

trained_acc = sum(r["prefers_chosen"] for r in trained_records) / len(trained_records)
print(f"trained judge accuracy: {trained_acc:.3f} ({sum(r['prefers_chosen'] for r in trained_records)}/{len(trained_records)})")

# 2. Free the trained model and load a fresh untrained backbone for the baseline
del model, trainer
gc.collect()
torch.cuda.empty_cache()

base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = BACKBONE,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
    dtype = None,
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

## Cell 12 — Delta B: trained judge vs prompt-engineered same-backbone

Delta B asks: did SimPO training beat what a careful prompt could do on the same backbone?

We use the *same untrained backbone* as the baseline arm above, but instead of just scoring chosen/rejected log-likelihood, we **prompt** it with a careful 5-marker rubric and ask it to pick. This is the "no-training, prompt-engineered" comparator the brief explicitly asks for.

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
    inputs = tokenizer(msg, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN-4).to(model.device)
    out = model.generate(**inputs, max_new_tokens=2, do_sample=False, temperature=0.0)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return text[:1]


# Run prompt-engineered judge on held-out pairs
# Randomize A/B order per pair to remove position bias
import random
random.seed(3407)
prompt_records = []
for t in dev_rows:
    swap = random.random() < 0.5
    a_text = t["rejected"] if swap else t["chosen"]
    b_text = t["chosen"] if swap else t["rejected"]
    pick = prompt_judge_pair(base_model, base_tokenizer, t["prompt"], a_text, b_text)
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

# Re-load trained model for timing (we freed it earlier)
del base_model, base_tokenizer
gc.collect(); torch.cuda.empty_cache()

trained_model_eval, trained_tok_eval = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_PATH,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
)
if trained_tok_eval.pad_token is None:
    trained_tok_eval.pad_token = trained_tok_eval.eos_token
FastLanguageModel.for_inference(trained_model_eval)

trained_lat = time_n(lambda: judge_pair(trained_model_eval, trained_tok_eval, sample["prompt"], sample["chosen"], sample["rejected"]))

del trained_model_eval, trained_tok_eval
gc.collect(); torch.cuda.empty_cache()

base_model2, base_tok2 = FastLanguageModel.from_pretrained(
    model_name = BACKBONE,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit = True,
)
if base_tok2.pad_token is None:
    base_tok2.pad_token = base_tok2.eos_token
FastLanguageModel.for_inference(base_model2)

prompt_lat = time_n(lambda: prompt_judge_pair(base_model2, base_tok2, sample["prompt"], sample["chosen"], sample["rejected"]))

# Cost on Colab T4 = $0 (free tier). On a paid GPU host like RunPod 4090 (~$0.34/hr),
# we report the per-judgment $ cost as latency × hourly_rate / 3600.
RUNPOD_4090_USD_PER_HR = 0.34
trained_cost_per_judgment = trained_lat * RUNPOD_4090_USD_PER_HR / 3600
prompt_cost_per_judgment  = prompt_lat  * RUNPOD_4090_USD_PER_HR / 3600

print(f"trained-judge latency: {trained_lat*1000:.1f} ms / judgment   (~${trained_cost_per_judgment:.5f})")
print(f"prompt-judge  latency: {prompt_lat*1000:.1f} ms / judgment   (~${prompt_cost_per_judgment:.5f})")
```

---

## Cell 14 — Paired bootstrap (95% CI on Delta A and Delta B)

```python
import numpy as np

def paired_bootstrap(a, b, n=10000, seed=42):
    """a, b: lists of 0/1 correctness for the same N pairs in the same order."""
    rng = np.random.default_rng(seed)
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
        "p_value":   float((diffs <= 0).mean()),  # one-sided: P(trained <= baseline)
    }

a_correct = [r["prefers_chosen"] for r in trained_records]
b_correct = [r["prefers_chosen"] for r in base_records]
p_correct = [r["prefers_chosen"] for r in prompt_records]

delta_a = paired_bootstrap(a_correct, b_correct)
delta_b = paired_bootstrap(a_correct, p_correct)

print("Delta A (trained vs untrained backbone):", json.dumps(delta_a, indent=2))
print("Delta B (trained vs prompt-engineered):", json.dumps(delta_b, indent=2))
```

**Interpretation:** if `ci_low > 0` AND `p_value < 0.05`, the lift is statistically significant. The brief requires this for Delta A but explicitly says Delta B may fail (and that's a publishable finding).

---

## Cell 15 — Write `ablation_results.json` + `held_out_traces.jsonl`

```python
import os
os.makedirs("ablations", exist_ok=True)

# Delta C: informational. Reuse Week 10 τ²-Bench retail score (no re-run; brief forbids it).
WEEK10_TAU2_PASS_AT_1 = 0.7267
WEEK10_TAU2_CI = [0.6504, 0.7917]
WEEK10_TAU2_BASELINE_COMMIT = "d11a97072c49d093f7b5a3e4fe9da95b490d43ba"

ablation = {
    "version": "v0.1",
    "backbone": BACKBONE,
    "training": {
        "loss_type": "simpo",
        "beta": config.beta, "simpo_gamma": config.simpo_gamma, "cpo_alpha": config.cpo_alpha,
        "learning_rate": config.learning_rate,
        "max_steps": config.max_steps,
        "effective_batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        "wall_time_min": round(wall_min, 1),
        "final_train_loss": round(result.training_loss, 4),
    },
    "held_out": {
        "n_pairs": len(dev_rows),
        "source_modes_eligible": "tasks with ground_truth.chosen_output AND rejected_output (style_guide_pair + a few others)",
    },
    "delta_a": {
        "name": "trained vs Week-10-baseline (untrained backbone, same scale)",
        "trained_acc": round(trained_acc, 4),
        "baseline_acc": round(base_acc, 4),
        "raw_lift_pp": round(trained_acc - base_acc, 4),
        **{f"bootstrap_{k}": v for k, v in delta_a.items()},
        "verdict": "POSITIVE_SIGNIFICANT" if (delta_a["ci_low"] > 0 and delta_a["p_value"] < 0.05)
                  else "POSITIVE_NOT_SIGNIFICANT" if delta_a["mean_diff"] > 0
                  else "FLAT_OR_NEGATIVE",
    },
    "delta_b": {
        "name": "trained vs prompt-engineered same-backbone",
        "trained_acc": round(trained_acc, 4),
        "prompt_acc": round(prompt_acc, 4),
        "raw_lift_pp": round(trained_acc - prompt_acc, 4),
        **{f"bootstrap_{k}": v for k, v in delta_b.items()},
        "verdict": "POSITIVE_SIGNIFICANT" if (delta_b["ci_low"] > 0 and delta_b["p_value"] < 0.05)
                  else "POSITIVE_NOT_SIGNIFICANT" if delta_b["mean_diff"] > 0
                  else "FLAT_OR_NEGATIVE",
        "honest_note": "Per the brief: a flat or negative Delta B is a legitimate, publishable finding. Report as-is.",
    },
    "delta_c_informational": {
        "name": "Week 10 τ²-Bench retail (informational only, NO re-run per brief)",
        "week10_pass_at_1": WEEK10_TAU2_PASS_AT_1,
        "week10_ci": WEEK10_TAU2_CI,
        "week10_baseline_commit": WEEK10_TAU2_BASELINE_COMMIT,
        "interpretation": (
            "Tenacious-Bench is a different distribution. The trained judge was not trained "
            "on retail-domain tasks; we do not expect or claim Delta-C lift. Tenacious-Bench "
            "lift is Tenacious-specific by construction."
        ),
    },
    "cost_pareto": {
        "trained_judge_latency_ms": round(trained_lat * 1000, 1),
        "prompt_judge_latency_ms":  round(prompt_lat * 1000, 1),
        "trained_cost_usd_per_judgment_at_runpod_4090": round(trained_cost_per_judgment, 6),
        "prompt_cost_usd_per_judgment_at_runpod_4090":  round(prompt_cost_per_judgment, 6),
        "training_one_time_cost_usd": 0.0,
    },
}

with open("ablations/ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(ablation, f, indent=2)
print("wrote ablations/ablation_results.json")

# held_out_traces.jsonl: one row per held-out pair with all three judge scores
trace_rows = []
for i, t in enumerate(dev_rows):
    trace_rows.append({
        "pair_idx": i,
        "trained":         trained_records[i],
        "untrained":       base_records[i],
        "prompt_engineered": prompt_records[i],
    })
with open("ablations/held_out_traces.jsonl", "w", encoding="utf-8") as f:
    for r in trace_rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote ablations/held_out_traces.jsonl ({len(trace_rows)} rows)")
```

---

## Cell 16 — Write `model_card.md`

```python
model_card = f"""---
license: cc-by-4.0
base_model: {BACKBONE}
tags:
- preference-optimization
- simpo
- judge-model
- b2b-sales
- tenacious-bench
datasets:
- {HF_USERNAME}/tenacious_bench_v0.1
---

# Tenacious-Judge SimPO (Qwen2.5-3B + LoRA)

A small preference-tuned judge (critic) for Tenacious-style B2B sales outreach. Trained via SimPO on 128 (chosen, rejected) pairs from Tenacious-Bench v0.1.

## Intended use

Deployed as a **rejection-sampling layer** in front of a Tenacious sales-outreach generator. The generator produces a draft; this judge scores the draft (or the (chosen, rejected) candidate set); the orchestrator routes back to regeneration if the judge prefers an alternate.

## Training

- Backbone: `{BACKBONE}` (4-bit QLoRA via Unsloth)
- Algorithm: SimPO (TRL `CPOTrainer` with `loss_type='simpo'`)
- Hyperparameters: β={config.beta}, simpo_γ={config.simpo_gamma}, lr={config.learning_rate}, cosine scheduler, warmup {config.warmup_ratio}, effective batch {config.per_device_train_batch_size * config.gradient_accumulation_steps}, max_steps={config.max_steps}
- Wall time: {wall_min:.1f} min on Colab T4 (free)
- Training data: 128 preference pairs from `tenacious_bench_v0.1/train` (Tenacious-Bench v0.1)

## Held-out evaluation

| Metric | Trained | Untrained backbone | Prompt-engineered same backbone |
|---|---:|---:|---:|
| Held-out preference accuracy | {trained_acc:.3f} | {base_acc:.3f} | {prompt_acc:.3f} |
| Raw lift vs untrained (Delta A) | — | — | — |

- Delta A: {trained_acc - base_acc:+.3f} pp (95% CI [{delta_a['ci_low']:.3f}, {delta_a['ci_high']:.3f}], p={delta_a['p_value']:.4f})
- Delta B: {trained_acc - prompt_acc:+.3f} pp (95% CI [{delta_b['ci_low']:.3f}, {delta_b['ci_high']:.3f}], p={delta_b['p_value']:.4f})
- Delta C (informational): not measured here; Week 10 τ²-retail pass@1 = {WEEK10_TAU2_PASS_AT_1} on file.

## Limitations

- Held-out eval slice is small (n={len(dev_rows)}) — wide CI bands.
- Judge is trained on Tenacious-specific failure modes; **not** a general B2B preference scorer.
- Tone-marker calibration uses heuristic regex fallback when the eval-tier judge is offline; some tone failures may pass.
- The 12 style-guide pairs anchor the eval slice; remaining held-out tasks lack ground_truth and are not used here.

## Inference (rejection-sampling pattern)

```python
from peft import PeftModel
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "{BACKBONE}", max_seq_length=2048, load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "{HF_REPO}")
FastLanguageModel.for_inference(model)

# Then score (prompt, chosen, rejected) triples via completion log-likelihood.
```

## Citation

```bibtex
@misc{{tenacious_judge_simpo_2026,
  title={{Tenacious-Judge SimPO: a small preference-tuned judge for B2B sales outreach}},
  author={{Yorat, Eyoel}},
  year={{2026}},
  howpublished={{HuggingFace}},
  license={{CC-BY-4.0}}
}}
```
"""

with open("model_card.md", "w", encoding="utf-8") as f:
    f.write(model_card)
print("wrote model_card.md")
```

---

## Cell 17 — Download outputs to your laptop, review, then push

The flow is: **Colab → zip → browser download → unzip into local repo → review → git push from your laptop.** This keeps you in the loop on what gets committed (no surprise 100 MB adapters in git) and the LoRA adapter stays on HuggingFace where it belongs.

### Cell 17a — Zip the *small* artifacts only (committable)

```python
# Small artifacts that DO go into git (logs, results, model card)
!rm -f /content/act4_outputs.zip
!cd /content/sales_evaluation_bench && zip -r /content/act4_outputs.zip \
    training/training_run.log \
    training/loss_curve.png \
    ablations/ablation_results.json \
    ablations/held_out_traces.jsonl \
    model_card.md

!ls -lh /content/act4_outputs.zip

from google.colab import files
files.download("/content/act4_outputs.zip")  # triggers browser download to your Downloads/
print("downloaded — see next cell for the local-side steps")
```

### Cell 17b — (Optional) Zip the LoRA adapter separately for safekeeping

The adapter is already on HuggingFace (Cell 9). Only download a local copy if you want one for offline inference. **Don't commit it to git** — it's >50 MB.

```python
!cd /content/sales_evaluation_bench && zip -r /content/tenacious_judge_adapter.zip outputs/tenacious_judge_simpo
!ls -lh /content/tenacious_judge_adapter.zip
from google.colab import files
files.download("/content/tenacious_judge_adapter.zip")
```

### What to do on your laptop

1. The browser drops `act4_outputs.zip` into `C:\Users\user\Downloads\`.
2. Unpack into the repo (PowerShell):
   ```powershell
   Expand-Archive -Force "$env:USERPROFILE\Downloads\act4_outputs.zip" `
                  -DestinationPath "c:\Users\user\Documents\tenx_academy\sales_evaluation_bench"
   ```
   Or in Git Bash:
   ```bash
   cd "c:/Users/user/Documents/tenx_academy/sales_evaluation_bench"
   unzip -o "$USERPROFILE/Downloads/act4_outputs.zip"
   ```
3. Review the diff:
   ```bash
   git status
   git diff training/training_run.log
   cat ablations/ablation_results.json | head -30
   ```
4. Commit + push:
   ```bash
   git add training/ ablations/ model_card.md
   git commit -m "Act IV: SimPO judge training + Delta A/B/C + Cost-Pareto"
   git push origin master
   ```

### One-time `.gitignore` addition (so adapter dirs never get accidentally committed)

If you also extracted `tenacious_judge_adapter.zip`, add this line to `.gitignore` once:
```
outputs/
```
The HuggingFace push from Cell 9 is the canonical store for the adapter; the local copy is just for re-loading without re-downloading.

---

## Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| OOM during training | Long sequences or too-large batch | Set `MAX_SEQ_LEN=1024`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` |
| Loss flat after step 50 | Data quality issue, not algorithm | Check `train_rows[0]` — is `chosen` actually different from `rejected`? Are they Tenacious-shaped or noise? |
| HF push 401 | Token has read scope only | Re-create with **write** scope at https://huggingface.co/settings/tokens |
| Delta A negative | Either training under-cooked OR data noise | Don't throw more compute. Inspect 5 random pairs; if `chosen` outputs are templated and degenerate, regenerate them with the dev-tier rewrite pass before re-training |
| Delta B negative | Prompt-engineered baseline beat training | This is a **publishable** finding per the brief. Report honestly in the blog. |
| Loss curve plot empty | `trainer.state.log_history` is empty | Make sure you ran the training cell to completion (not killed mid-step) |
| `dev_rows` empty | No held-out tasks have ground_truth | The 12 style_guide_pair tasks should always have it; if 0 pairs, your repo clone is stale — re-pull |

---

## What this run does NOT do

- **Doesn't run the full Week 10 agent** to generate fresh drafts on held-out scenarios. The judge accuracy here is a *judge-component* metric. The full agent+judge end-to-end on Tenacious-Bench is a Day-7 follow-up if time permits — for the brief's grading, judge accuracy + the three deltas + cost-Pareto are sufficient.
- **Doesn't re-run τ²-Bench retail.** The brief explicitly forbids it; we reuse the Week 10 number ($0 spend, no compute).
- **Doesn't use the eval-tier model (Claude Sonnet 4.6 / GPT-5).** The brief permits 3–4 eval-tier passes on Day 6 within the $2–3 envelope; if you want to add a Claude-judged pairwise comparison as a fourth ablation row, that's the right place to spend it. Optional.

---

## Mapping to brief deliverables

| Brief deliverable | Where it lands |
|---|---|
| `ablation_results.json` | Cell 15 → `ablations/ablation_results.json` |
| `held_out_traces.jsonl` | Cell 15 → `ablations/held_out_traces.jsonl` |
| `model_card.md` (Path A or C; optional for Path B) | Cell 16 → `model_card.md` |
| `training_run.log` with hyperparameters and loss curves | Cell 8 → `training/training_run.log` + `training/loss_curve.png` |
| Delta A positive on sealed held-out with 95% CI separation, p<0.05 | Cell 14 → `delta_a.bootstrap_ci_low > 0 AND delta_a.bootstrap_p_value < 0.05` |
| Delta B reported honestly (even if negative) | Cell 14 → `delta_b` block |
| Delta C informational (no re-run) | Cell 15 → `delta_c_informational` |
| Cost-Pareto with vs without trained component | Cell 13 → `cost_pareto` block |
