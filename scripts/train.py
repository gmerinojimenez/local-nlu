"""
Main training script for T5 NLU model.
"""
import sys
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel
from src.data.dataset import create_data_loaders
from src.training.trainer import NLUTrainer


def main():
    """Main training function."""
    print("=" * 80)
    print("T5 NLU MODEL TRAINING")
    print("=" * 80)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract configs
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    paths = config['paths']

    print("\nConfiguration:")
    print(f"  Model: {model_config['name']}")
    print(f"  Batch size: {training_config['batch_size']}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Epochs: {training_config['num_epochs']}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Initialize model
    print("\nInitializing model...")
    model = T5NLUModel(
        model_name=model_config['name'],
        dropout=model_config['dropout']
    )

    # Create data loaders
    print("\nCreating data loaders...")
    processed_path = data_config['processed_data_path']
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=f"{processed_path}/train.csv",
        val_path=f"{processed_path}/val.csv",
        test_path=f"{processed_path}/test.csv",
        tokenizer=model.tokenizer,
        batch_size=training_config['batch_size'],
        max_input_length=model_config['max_input_length'],
        max_output_length=model_config['max_output_length'],
        num_workers=2  # Reduce if you have memory issues
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = NLUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        max_grad_norm=training_config['max_grad_norm'],
        save_dir=paths['model_save_dir']
    )

    # Train
    trainer.train(
        num_epochs=training_config['num_epochs'],
        save_every=1
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest model saved to: {paths['model_save_dir']}/best_model")
    print("\nNext steps:")
    print("  1. Evaluate the model: python scripts/evaluate.py")
    print("  2. Test inference: python scripts/inference.py")


if __name__ == "__main__":
    main()
