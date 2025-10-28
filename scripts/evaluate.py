"""
Evaluation script for T5 NLU model.
"""
import sys
from pathlib import Path
import yaml
import torch
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel
from src.data.dataset import create_data_loaders
from src.training.evaluator import NLUEvaluator


def main():
    """Main evaluation function."""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    paths = config['paths']

    # Load best model
    model_path = paths['model_save_dir'] + '/best_model'
    print(f"\nLoading model from: {model_path}")

    model = T5NLUModel.from_pretrained(model_path)

    # Create data loaders
    print("Creating data loaders...")
    processed_path = data_config['processed_data_path']
    _, val_loader, test_loader = create_data_loaders(
        train_path=f"{processed_path}/train.csv",
        val_path=f"{processed_path}/val.csv",
        test_path=f"{processed_path}/test.csv",
        tokenizer=model.tokenizer,
        batch_size=training_config['batch_size'],
        max_input_length=model_config['max_input_length'],
        max_output_length=model_config['max_output_length'],
        num_workers=2
    )

    # Initialize evaluator
    evaluator = NLUEvaluator(
        model=model,
        tokenizer=model.tokenizer
    )

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)
    val_metrics = evaluator.evaluate_batch(val_loader)
    evaluator.print_metrics(val_metrics)

    # Save validation metrics
    val_metrics_path = paths['model_save_dir'] + '/val_metrics.json'
    with open(val_metrics_path, 'w') as f:
        # Convert per_intent_accuracy to serializable format
        metrics_to_save = val_metrics.copy()
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nValidation metrics saved to: {val_metrics_path}")

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    test_metrics = evaluator.evaluate_batch(test_loader)
    evaluator.print_metrics(test_metrics)

    # Save test metrics
    test_metrics_path = paths['model_save_dir'] + '/test_metrics.json'
    with open(test_metrics_path, 'w') as f:
        metrics_to_save = test_metrics.copy()
        json.dump(metrics_to_save, f, indent=2)
    print(f"\nTest metrics saved to: {test_metrics_path}")


if __name__ == "__main__":
    main()
