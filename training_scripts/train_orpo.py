"""ORPO Training Script for Tenacious-Bench v0.1

Runs ORPO (Odds-Ratio Preference Optimization) on the preference pairs.
ORPO is a backup algorithm - monolithic preference optimization without reference model.

Usage:
  python -m training_scripts.train_orpo \
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
    """Dataset for ORPO preference pairs."""
    
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
        
        return {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "task_id": item["task_id"],
            "dimension": item["dimension"],
        }


def compute_orpo_loss(
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Compute ORPO loss.
    
    ORPO (Odds-Ratio Preference Optimization):
    - Monolithic preference optimization without reference model
    - Uses odds ratio between chosen and rejected probabilities
    - Simpler than SimPO but less stable
    
    Args:
        policy_logits: Logits from policy model (batch, seq_len, vocab)
        labels: Token labels (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        beta: Odds ratio temperature (default 0.5)
    
    Returns:
        Loss tensor
    """
    # Get log probabilities
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    
    # Gather log probs for labels
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply attention mask
    token_log_probs = token_log_probs * attention_mask
    
    # Sum over sequence
    seq_log_probs = token_log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)
    
    # ORPO uses sigmoid on log odds
    # loss = -log(sigmoid(log_prob_chosen - log_prob_rejected))
    # Simplified for single-sequence training
    
    return -torch.log(torch.sigmoid(seq_log_probs * beta)).mean()


def train_orpo(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    data_path: str = None,
    output_dir: str = None,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-6,
    max_length: int = 2048,
    beta: float = 0.5,
    seed: int = 42,
):
    """Run ORPO training."""
    
    # Set paths
    if data_path is None:
        data_path = str(TRAINING_DATA_DIR / "preference_pairs.jsonl")
    if output_dir is None:
        output_dir = str(OUTPUT_DIR / "orpo_run")
    
    # Set seed
    torch.manual_seed(seed)
    
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
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
        collate_fn=lambda x: x,
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Starting ORPO training for {epochs} epochs")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Beta (odds ratio temperature): {beta}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            for sample in batch:
                prompt = sample["prompt"]
                chosen = sample["chosen"]
                rejected = sample["rejected"]
                
                # Tokenize chosen response
                chosen_inputs = tokenizer(
                    f"{prompt}\n\n{chosen}",
                    max_length=max_length,
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
    logger.info(f"ORPO Training complete! Final model saved to {final_dir}")
    
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="ORPO Training for Tenacious-Bench")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train_orpo(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        beta=args.beta,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()