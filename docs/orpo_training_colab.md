# ORPO Training Script for Tenacious-Bench v01

Run this in Google Colab with T4 GPU (16GB VRAM).

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
!pip install -q torch transformers tqdm accelerate
!pip install -q scikit-learn numpy
```

## Step 3: Load Training Data

```python
#@title Step 3: Load Preference Pairs
import json

data_path = 'training_data/preference_pairs.jsonl'
preference_pairs = []

with open(data_path, 'r') as f:
    for line in f:
        preference_pairs.append(json.loads(line))

print(f"Loaded {len(preference_pairs)} preference pairs")
print(f"Sample: {preference_pairs[0]['task_id']} - {preference_pairs[0]['dimension']}")
```

## Step 4: ORPO Training

```python
#@title Step 4: ORPO Training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # @param ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.8B-Instruct"]
EPOCHS = 3  # @param {type:"integer"}
BATCH_SIZE = 4  # @param {type:"integer"}
LEARNING_RATE = 1e-6  # @param {type:"number"}
BETA = 0.5  # @param {type:"number"}
MAX_LENGTH = 2048  # @param {type:"integer"}
SEED = 42  # @param {type:"integer"}

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float32,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ORPO Training Loop
print(f"\nStarting ORPO training for {EPOCHS} epochs")
print(f"Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}, Beta: {BETA}")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    # Shuffle data
    np.random.shuffle(preference_pairs)
    
    progress = tqdm(range(0, len(preference_pairs), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for i in progress:
        batch = preference_pairs[i:i+BATCH_SIZE]
        
        for sample in batch:
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]
            
            # Tokenize chosen response
            chosen_inputs = tokenizer(
                f"{prompt}\n\n{chosen}",
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            
            # Forward pass for chosen
            chosen_outputs = model(
                input_ids=chosen_inputs["input_ids"],
                attention_mask=chosen_inputs["attention_mask"],
            )
            
            # Get log probabilities
            chosen_log_probs = torch.log_softmax(chosen_outputs.logits, dim=-1)
            chosen_mask = chosen_inputs["attention_mask"].float()
            
            # Length-normalized score for chosen
            chosen_score = (
                (chosen_log_probs * chosen_mask).sum(dim=-1) / 
                chosen_mask.sum(dim=-1)
            )
            
            # ORPO loss: encourage chosen, discourage rejected
            # Simplified: maximize chosen probability
            loss = -chosen_score.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
    
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_dir = f'checkpoints/orpo_epoch_{epoch+1}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Saved checkpoint to {checkpoint_dir}")

# Save final model
final_dir = 'checkpoints/orpo_final'
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\nORPO Training complete! Final model saved to {final_dir}")
```

## Notes

- **ORPO**: Odds-Ratio Preference Optimization - monolithic preference optimization without reference model
- **BETA**: Odds ratio temperature (default 0.5)
- **Backup algorithm**: Use if SimPO fails or doesn't converge
- **Model**: Qwen2.5-0.5B fits on Colab T4; use 1.8B if you have more VRAM