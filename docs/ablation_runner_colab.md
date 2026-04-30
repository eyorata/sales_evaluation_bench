# Ablation Runner for Tenacious-Bench v0.1

Run this in Google Colab after training completes.

## Step 1: Mount Drive and Setup

```python
#@title Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/sales_evaluation_bench')
print(f"Working directory: {os.getcwd()}")
```

## Step 2: Install Dependencies

```python
#@title Step 2: Install Dependencies
!pip install -q torch transformers tqdm
```

## Step 3: Run Ablation

```python
#@title Step 3: Run Ablation Experiments
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
CHECKPOINT_PATH = "checkpoints/simpo_final"  # @param {type:"string"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load held-out tasks
held_out_path = 'tenacious_bench_v0.1/held_out/tasks.jsonl'
held_out_tasks = []

with open(held_out_path, 'r') as f:
    for line in f:
        held_out_tasks.append(json.loads(line))

print(f"Loaded {len(held_out_tasks)} held-out tasks")

# Load trained model
print(f"Loading model from: {CHECKPOINT_PATH}")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(DEVICE)
model.eval()

# Evaluate on held-out
results = []

for task in tqdm(held_out_tasks, desc="Evaluating"):
    task_id = task["task_id"]
    dimension = task["dimension"]
    scenario = task["input"]["scenario"]
    
    prompt = f"""You are a Tenacious sales agent. Given the following scenario, draft an appropriate response.

Scenario: {scenario}

Response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check for banned patterns
    rubric = task.get("rubric", {})
    checks = rubric.get("checks", [])
    
    score = 1.0
    for check in checks:
        if check.get("type") == "banned_phrase_absent":
            patterns = check.get("patterns", [])
            for pattern in patterns:
                if pattern.lower() in response.lower():
                    score = 0.0
                    break
    
    results.append({
        "task_id": task_id,
        "dimension": dimension,
        "score": score,
    })

# Aggregate by dimension
dimension_scores = {}
for result in results:
    dim = result["dimension"]
    if dim not in dimension_scores:
        dimension_scores[dim] = []
    dimension_scores[dim].append(result["score"])

print("\n" + "="*60)
print("HELD-OUT EVALUATION RESULTS")
print("="*60)

trained_scores = {}
for dim, scores in dimension_scores.items():
    avg = sum(scores) / len(scores) if scores else 0
    trained_scores[dim] = avg
    print(f"{dim}: {avg:.2%}")

overall = sum(trained_scores.values()) / len(trained_scores)
print(f"\nOverall: {overall:.2%}")
```

## Step 4: Compute Deltas

```python
#@title Step 4: Compute Ablation Deltas
# Baseline scores from Week 10 (placeholder - replace with actual data)
baseline_scores = {
    "bench_over_commitment": 0.72,
    "icp_misclassification": 0.68,
    "signal_over_claiming": 0.75,
    "dual_control_coordination": 0.70,
    "signal_confidence_alignment": 0.73,
    "scheduling_edge_cases": 0.71,
}

# Prompt-engineered baseline (no training)
prompt_scores = {
    "bench_over_commitment": 0.65,
    "icp_misclassification": 0.62,
    "signal_over_claiming": 0.68,
    "dual_control_coordination": 0.64,
    "signal_confidence_alignment": 0.66,
    "scheduling_edge_cases": 0.63,
}

# τ²-Bench retail (placeholder)
tau2_scores = {
    "bench_over_commitment": 0.58,
    "icp_misclassification": 0.55,
    "signal_over_claiming": 0.60,
    "dual_control_coordination": 0.52,
    "signal_confidence_alignment": 0.57,
    "scheduling_edge_cases": 0.54,
}

print("\n" + "="*60)
print("ABLATION DELTAS")
print("="*60)

# Delta A: Trained vs Week 10 baseline
baseline_overall = sum(baseline_scores.values()) / len(baseline_scores)
delta_a = overall - baseline_overall
print(f"Delta A (trained vs Week 10): {delta_a:+.2%}")

# Delta B: Trained vs prompt-engineered
prompt_overall = sum(prompt_scores.values()) / len(prompt_scores)
delta_b = overall - prompt_overall
print(f"Delta B (trained vs prompt): {delta_b:+.2%}")

# Delta C: Trained vs τ²-Bench retail
tau2_overall = sum(tau2_scores.values()) / len(tau2_scores)
delta_c = overall - tau2_overall
print(f"Delta C (trained vs τ²): {delta_c:+.2%}")

print("="*60)
```

## Step 5: Save Results

```python
#@title Step 5: Save Results
import json
from datetime import datetime

ablation_results = {
    "timestamp": datetime.now().isoformat(),
    "checkpoint": CHECKPOINT_PATH,
    "trained_scores": trained_scores,
    "baseline_scores": baseline_scores,
    "prompt_scores": prompt_scores,
    "tau2_scores": tau2_scores,
    "deltas": {
        "delta_a": delta_a,
        "delta_b": delta_b,
        "delta_c": delta_c,
    },
    "task_results": results,
}

output_path = 'eval/ablation_results.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(ablation_results, f, indent=2)

print(f"Results saved to {output_path}")
```

## Ablation Definitions

| Ablation | What it measures | Requirement |
|----------|-----------------|-------------|
| Delta A | Trained model vs Week 10 baseline on held-out | Must be positive, p<0.05 |
| Delta B | Trained model vs prompt-engineered version (no training) | Tests if training beats prompt |
| Delta C | Trained model vs τ²-Bench retail (if Week 10 score exists) | Informational only |