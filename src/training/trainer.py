"""
Training loop for T5 NLU model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datetime import datetime


class NLUTrainer:
    """Trainer for T5 NLU model."""

    def __init__(self,
                 model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 500,
                 weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'models'):
        """
        Initialize the trainer.

        Args:
            model: T5NLUModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            save_dir: Directory to save models
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm

        # Move model to device
        self.model.to(device)

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup scheduler
        total_steps = len(train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training state
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        print(f"Trainer initialized on device: {device}")
        print(f"Trainable parameters: {model.count_parameters():,}")

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs['loss'].item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def train(self, num_epochs: int, save_every: int = 1):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")

            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(
                self.scheduler.get_last_lr()[0]
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"✓ New best model saved! (val_loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "=" * 80)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save training history
        self.save_training_history()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            save_path = self.save_dir / "best_model"
        else:
            save_path = self.save_dir / f"checkpoint_epoch_{epoch}"

        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))

        # Save training state
        state = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        torch.save(state, save_path / 'training_state.pt')

    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        from src.models.t5_nlu import T5NLUModel

        # Load model
        self.model = T5NLUModel.from_pretrained(checkpoint_path)
        self.model.to(self.device)

        # Load training state
        state_path = Path(checkpoint_path) / 'training_state.pt'
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.best_val_loss = state['best_val_loss']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            print(f"Warning: Training state not found at {state_path}")


if __name__ == "__main__":
    # This will be called from the training script
    print("Trainer module loaded successfully")
