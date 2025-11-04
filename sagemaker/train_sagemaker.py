"""
SageMaker training entry point script.
This script runs inside the SageMaker training container.
"""
import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path

# Add the code directory to Python path
# SageMaker extracts code to /opt/ml/code
CODE_DIR = '/opt/ml/code'
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

print(f"Python path: {sys.path}")
print(f"Code directory contents: {os.listdir(CODE_DIR)}")

# SageMaker expects data in /opt/ml/input/data/
# and outputs to go to /opt/ml/model/


def train_model(args):
    """Main training function for SageMaker."""
    # Import after adding to path
    # Note: In SageMaker, src/ structure is flattened to /opt/ml/code/
    from models.t5_nlu import T5NLUModel
    from data.dataset import create_data_loaders
    from training.trainer import NLUTrainer

    print("=" * 80)
    print("SAGEMAKER TRAINING JOB")
    print("=" * 80)

    # SageMaker paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.model_dir)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # List data files
    print(f"\nAvailable data files:")
    for file in data_dir.glob('*.csv'):
        print(f"  - {file.name}")

    # Initialize model
    print(f"\nInitializing {args.model_name}...")
    model = T5NLUModel(
        model_name=args.model_name,
        dropout=args.dropout
    )

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path=str(data_dir / "train.csv"),
        val_path=str(data_dir / "val.csv"),
        test_path=str(data_dir / "test.csv"),
        tokenizer=model.tokenizer,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        num_workers=args.num_workers
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = NLUTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_dir=str(output_dir)
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.num_epochs, save_every=1)

    # Save final model
    print(f"\nSaving final model to {output_dir}...")
    model.save_pretrained(str(output_dir / "final_model"))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker specific parameters
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    # Model parameters
    parser.add_argument('--model-name', type=str, default='t5-base')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-input-length', type=int, default=128)
    parser.add_argument('--max-output-length', type=int, default=256)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    train_model(args)
