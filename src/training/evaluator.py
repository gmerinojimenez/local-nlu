"""
Evaluation metrics for NLU model.
"""
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import pandas as pd
from collections import defaultdict


class NLUEvaluator:
    """Evaluator for T5 NLU model."""

    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evaluator.

        Args:
            model: T5NLUModel instance
            tokenizer: T5 tokenizer
            device: Device to run evaluation on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_batch(self, data_loader: DataLoader) -> Dict:
        """
        Evaluate model on a data loader.

        Args:
            data_loader: DataLoader instance

        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_references = []
        all_intents_pred = []
        all_intents_true = []

        print("Evaluating...")
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Generate predictions
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=256,
                    num_beams=4
                )

                # Decode predictions and references
                for pred_ids, label_ids in zip(output_ids, labels):
                    # Decode prediction
                    pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    all_predictions.append(pred_text)

                    # Decode reference
                    label_ids = label_ids[label_ids != -100]
                    ref_text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
                    all_references.append(ref_text)

                    # Parse JSONs to extract intents
                    try:
                        pred_json = json.loads(pred_text)
                        all_intents_pred.append(pred_json.get('intent', 'UNKNOWN'))
                    except:
                        all_intents_pred.append('PARSE_ERROR')

                    try:
                        ref_json = json.loads(ref_text)
                        all_intents_true.append(ref_json.get('intent', 'UNKNOWN'))
                    except:
                        all_intents_true.append('UNKNOWN')

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_references,
            all_intents_pred,
            all_intents_true
        )

        return metrics

    def _calculate_metrics(self,
                          predictions: List[str],
                          references: List[str],
                          intents_pred: List[str],
                          intents_true: List[str]) -> Dict:
        """Calculate evaluation metrics."""
        # Exact match (full JSON match)
        exact_matches = sum(p == r for p, r in zip(predictions, references))
        exact_match_rate = exact_matches / len(predictions) * 100

        # Intent accuracy
        intent_correct = sum(p == t for p, t in zip(intents_pred, intents_true))
        intent_accuracy = intent_correct / len(intents_pred) * 100

        # Parse success rate
        parse_success = sum(1 for p in intents_pred if p != 'PARSE_ERROR')
        parse_success_rate = parse_success / len(intents_pred) * 100

        # Parameter-level metrics
        param_metrics = self._calculate_param_metrics(predictions, references)

        # Per-intent accuracy
        per_intent_acc = self._calculate_per_intent_accuracy(intents_pred, intents_true)

        metrics = {
            'exact_match': exact_match_rate,
            'intent_accuracy': intent_accuracy,
            'parse_success_rate': parse_success_rate,
            'param_f1': param_metrics['f1'],
            'param_precision': param_metrics['precision'],
            'param_recall': param_metrics['recall'],
            'per_intent_accuracy': per_intent_acc,
            'total_samples': len(predictions)
        }

        return metrics

    def _calculate_param_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate precision, recall, and F1 for parameters."""
        total_correct = 0
        total_predicted = 0
        total_reference = 0

        for pred_text, ref_text in zip(predictions, references):
            try:
                pred_json = json.loads(pred_text)
                ref_json = json.loads(ref_text)

                pred_params = pred_json.get('params', {})
                ref_params = ref_json.get('params', {})

                # Flatten nested dicts for comparison
                pred_flat = self._flatten_dict(pred_params)
                ref_flat = self._flatten_dict(ref_params)

                # Count matches
                for key, value in pred_flat.items():
                    if key in ref_flat and ref_flat[key] == value:
                        total_correct += 1

                total_predicted += len(pred_flat)
                total_reference += len(ref_flat)

            except:
                continue

        precision = total_correct / total_predicted * 100 if total_predicted > 0 else 0
        recall = total_correct / total_reference * 100 if total_reference > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _calculate_per_intent_accuracy(self,
                                      intents_pred: List[str],
                                      intents_true: List[str]) -> Dict:
        """Calculate accuracy per intent."""
        intent_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for pred, true in zip(intents_pred, intents_true):
            intent_stats[true]['total'] += 1
            if pred == true:
                intent_stats[true]['correct'] += 1

        # Calculate accuracy for each intent
        per_intent_acc = {}
        for intent, stats in intent_stats.items():
            acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            per_intent_acc[intent] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }

        return per_intent_acc

    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a formatted way."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nOverall Metrics:")
        print(f"  Exact Match:           {metrics['exact_match']:.2f}%")
        print(f"  Intent Accuracy:       {metrics['intent_accuracy']:.2f}%")
        print(f"  Parse Success Rate:    {metrics['parse_success_rate']:.2f}%")
        print(f"  Parameter F1:          {metrics['param_f1']:.2f}%")
        print(f"  Parameter Precision:   {metrics['param_precision']:.2f}%")
        print(f"  Parameter Recall:      {metrics['param_recall']:.2f}%")
        print(f"  Total Samples:         {metrics['total_samples']}")

        print(f"\nTop 10 Intent Accuracies:")
        per_intent = sorted(
            metrics['per_intent_accuracy'].items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]

        for intent, stats in per_intent:
            print(f"  {intent:<30} {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")

        print(f"\nBottom 5 Intent Accuracies:")
        bottom_intents = sorted(
            metrics['per_intent_accuracy'].items(),
            key=lambda x: x[1]['accuracy']
        )[:5]

        for intent, stats in bottom_intents:
            print(f"  {intent:<30} {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    print("Evaluator module loaded successfully")
