"""
Comprehensive model diagnosis script.
"""
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel


def analyze_training_history():
    """Analyze training loss curves."""
    history_path = Path("models/training_history.json")

    if not history_path.exists():
        print("⚠ Training history not found")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    print("\n" + "=" * 80)
    print("TRAINING HISTORY ANALYSIS")
    print("=" * 80)

    train_losses = history['train_loss']
    val_losses = history['val_loss']

    print(f"\nEpoch-by-epoch losses:")
    print(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15} {'Difference':<15}")
    print("-" * 60)

    for i, (train, val) in enumerate(zip(train_losses, val_losses), 1):
        diff = val - train
        status = "⚠ Overfitting" if diff > 0.5 else "✓ Good" if diff < 0.3 else "→ OK"
        print(f"{i:<10} {train:<15.4f} {val:<15.4f} {diff:<15.4f} {status}")

    # Check for convergence issues
    if len(train_losses) > 2:
        last_improvement = train_losses[-1] - train_losses[-2]
        print(f"\nLast epoch improvement: {last_improvement:.4f}")

        if abs(last_improvement) < 0.01:
            print("⚠ Training has plateaued - may need more epochs or learning rate adjustment")
        elif last_improvement > 0:
            print("⚠ Loss increased - possible learning rate too high or overfitting")

    # Final metrics
    print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Generalization Gap: {val_losses[-1] - train_losses[-1]:.4f}")

    return history


def test_model_samples():
    """Test model on diverse samples."""
    print("\n" + "=" * 80)
    print("MODEL OUTPUT QUALITY CHECK")
    print("=" * 80)

    model = T5NLUModel.from_pretrained('models/best_model')

    # Diverse test cases
    test_cases = {
        "Simple commands": [
            "volume up",
            "volume down",
            "play music",
            "pause"
        ],
        "Timer/Reminder": [
            "set a timer for 5 minutes",
            "remind me to call mom at 3pm",
            "set an alarm for 7am"
        ],
        "Questions": [
            "what is the weather",
            "what time is it",
            "who is the president"
        ],
        "Search/Open": [
            "open YouTube",
            "search Google for cats",
            "go to Wikipedia"
        ],
        "Complex utterances": [
            "remind me to email john about the meeting tomorrow at 2pm",
            "search Google Drive for my presentation about climate change",
            "set brightness to 75 percent"
        ]
    }

    issues = {
        'json_parse_error': 0,
        'missing_params': 0,
        'wrong_intent': 0,
        'total': 0
    }

    for category, utterances in test_cases.items():
        print(f"\n{category}:")
        print("-" * 60)

        for utterance in utterances:
            issues['total'] += 1
            result = model.predict(utterance)

            intent = result.get('intent', 'UNKNOWN')
            params = result.get('params', {})
            has_error = 'raw_output' in result or 'error' in result

            # Check for issues
            if has_error:
                issues['json_parse_error'] += 1
                print(f"✗ \"{utterance}\"")
                print(f"  Intent: {intent}")
                print(f"  Issue: JSON parsing failed")
                if 'raw_output' in result:
                    print(f"  Raw: {result['raw_output'][:100]}...")
            elif intent in ['UNSUPPORTED', 'PARSE_ERROR', 'UNKNOWN']:
                issues['wrong_intent'] += 1
                print(f"⚠ \"{utterance}\"")
                print(f"  Intent: {intent} (possibly incorrect)")
                print(f"  Params: {params}")
            elif not params and 'timer' in utterance.lower():
                issues['missing_params'] += 1
                print(f"⚠ \"{utterance}\"")
                print(f"  Intent: {intent}")
                print(f"  Issue: Expected parameters but got none")
            else:
                print(f"✓ \"{utterance}\"")
                print(f"  Intent: {intent}")
                if params:
                    print(f"  Params: {json.dumps(params)[:80]}...")

    # Summary
    print("\n" + "=" * 80)
    print("ISSUE SUMMARY")
    print("=" * 80)
    print(f"Total tested: {issues['total']}")
    print(f"JSON parse errors: {issues['json_parse_error']} ({issues['json_parse_error']/issues['total']*100:.1f}%)")
    print(f"Missing parameters: {issues['missing_params']} ({issues['missing_params']/issues['total']*100:.1f}%)")
    print(f"Wrong intent: {issues['wrong_intent']} ({issues['wrong_intent']/issues['total']*100:.1f}%)")

    return issues


def check_generation_settings():
    """Check if generation settings need tuning."""
    print("\n" + "=" * 80)
    print("GENERATION SETTINGS CHECK")
    print("=" * 80)

    print("\nCurrent settings in model.predict():")
    print("  - max_length: 256")
    print("  - num_beams: 4")
    print("  - early_stopping: True")

    print("\nRecommendations:")
    print("  1. Try increasing num_beams to 8 for better quality")
    print("  2. Try temperature sampling (add temperature=0.7)")
    print("  3. Consider increasing max_length to 512 for complex params")


def main():
    """Main diagnosis function."""
    print("=" * 80)
    print("MODEL DIAGNOSIS TOOL")
    print("=" * 80)

    # 1. Check training history
    history = analyze_training_history()

    # 2. Test model outputs
    issues = test_model_samples()

    # 3. Check generation settings
    check_generation_settings()

    # 4. Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if history:
        final_val_loss = history['val_loss'][-1]

        if final_val_loss > 1.0:
            print("\n🔴 HIGH LOSS - Model needs more training")
            print("   → Increase num_epochs to 10-15")
            print("   → Check if learning rate is appropriate")
            print("   → Verify data quality")
        elif final_val_loss > 0.5:
            print("\n🟡 MODERATE LOSS - Model partially trained")
            print("   → Train for 5-10 more epochs")
            print("   → Fine-tune generation parameters")
        else:
            print("\n🟢 LOW LOSS - Model well trained")
            print("   → Focus on generation parameter tuning")

    if issues:
        if issues['json_parse_error'] / issues['total'] > 0.3:
            print("\n🔴 HIGH JSON PARSE ERROR RATE")
            print("   → Model hasn't learned JSON format well")
            print("   → Train for MORE epochs (10-15)")
            print("   → Consider data augmentation with more diverse JSON examples")

        if issues['missing_params'] / issues['total'] > 0.2:
            print("\n🟡 MISSING PARAMETERS ISSUE")
            print("   → Increase max_output_length to 512")
            print("   → Train for more epochs")
            print("   → Check training data has parameter examples")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. If training loss is still decreasing:")
    print("   → Continue training: Edit configs/config.yaml, increase num_epochs to 15")
    print("   → Resume: python scripts/train.py")

    print("\n2. If model outputs are malformed:")
    print("   → The model needs more training epochs")
    print("   → JSON generation is complex and needs 10+ epochs")

    print("\n3. Run full evaluation:")
    print("   → python scripts/evaluate.py")
    print("   → Check exact match rate and per-intent accuracy")


if __name__ == "__main__":
    main()
