"""SimPO Training Script for Tenacious-Bench v0.1

Runs SimPO (Simple Preference Optimization) on the preference pairs.
Designed to run on Colab T4 (16GB VRAM) with limited resources.

Usage:
  python -m training_scripts.train_simpo \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-6
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints"


class PreferenceDataset(Dataset):
    """Dataset for SimPO preference pairs."""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> list[dict]:
        """Load preference pairs from JSONL."""
        data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} preference pairs")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single preference pair."""
        item = self.data[idx]
        
        # Format: prompt + chosen/rejected
        # SimPO uses length-normalized reward
        chosen_text = f"{item['prompt']}\n\n{item['chosen']}"
        rejected_text = f"{item['prompt']}\n\n{item['rejected']}"
        
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "task_id": item["task_id"],
            "dimension": item["dimension"],
        }


def compute_simpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Compute SimPO loss.
    
    SimPO (Simple Preference Optimization):
    - Reference-free, length-normalized preference optimization
    - Uses difference between chosen and rejected log probabilities
    - Normalized by sequence length
    
    Args:
        policy_logits: Logits from policy model (batch, seq_len, vocab)
        ref_logits: Logits from reference model (batch, seq_len, vocab)
        labels: Token labels (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        gamma: Target reward margin (default 0.5)
    
    Returns:
        Loss tensor
    """
    # Get log probabilities
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    
    # Gather log probs for labels
    # (batch, seq_len)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply attention mask
    token_log_probs = token_log_probs * attention_mask
    
    # Sum over sequence (length-normalized)
    # (batch,)
    seq_log_probs = token_log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)
    
    # For simplicity, use the prompt+chosen as "chosen" and prompt+rejected as "rejected"
    # In practice, you'd compute this for both and take the difference
    
    # SimPO loss: -log(sigmoid(reward - gamma))
    # where reward = log_prob(chosen) - log_prob(rejected)
    reward = seq_log_probs  # Simplified - would be difference in practice
    
    loss = -torch.log(torch.sigmoid(reward - gamma))
    
    return loss.mean()


def train_simpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    data_path: str = None,
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-6,
    max_length: int = 2048,
    gamma: float = 0.5,
    seed: int = 42,
):
    """Run SimPO training."""
    
    # Set paths
    if data_path is None:
        data_path = str(TRAINING_DATA_DIR / "preference_pairs.jsonl")
    if output_dir is None:
        output_dir = str(OUTPUT_DIR / "simpo_run")
    
    # Set seed
    torch.manual_seed(seed)
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for stability
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading training data from: {data_path}")
    dataset = PreferenceDataset(data_path, tokenizer, max_length)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # Custom collate
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            # Process each sample in batch
            # In practice, you'd batch this more efficiently
            for sample in batch:
                prompt = sample["prompt"]
                chosen = sample["chosen"]
                rejected = sample["rejected"]
                
                # Tokenize
                chosen_inputs = tokenizer(
                    f"{prompt}\n\n{chosen}",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_inputs = tokenizer(
                    f"{prompt}\n\n{rejected}",
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                # Move to device
                chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
                
                # Forward pass for chosen
                chosen_outputs = model(
                    input_ids=chosen_inputs["input_ids"],
                    attention_mask=chosen_inputs["attention_mask"],
                )
                
                # Forward pass for rejected
                rejected_outputs = model(
                    input_ids=rejected_inputs["input_ids"],
                    attention_mask=rejected_inputs["attention_mask"],
                )
                
                # Compute SimPO loss
                # Simplified: use log probability difference
                chosen_logits = chosen_outputs.logits
                rejected_logits = rejected_outputs.logits
                
                # Get log probs
                chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
                rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
                
                # Length-normalized scores
                chosen_mask = chosen_inputs["attention_mask"].float()
                rejected_mask = rejected_inputs["attention_mask"].float()
                
                chosen_score = (
                    (chosen_log_probs * chosen_mask).sum(dim=-1) / 
                    chosen_mask.sum(dim=-1)
                )
                rejected_score = (
                    (rejected_log_probs * rejected_mask).sum(dim=-1) / 
                    rejected_mask.sum(dim=-1)
                )
                
                # SimPO loss: prefer chosen over rejected
                reward = chosen_score - rejected_score
                loss = -torch.log(torch.sigmoid(reward - gamma)).mean()
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path(output_dir) / f"epoch_{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")
    
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="SimPO Training for Tenacious-Bench")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_simpo(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gamma=args.gamma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()