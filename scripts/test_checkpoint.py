"""
Test a specific checkpoint during or after training.
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel


def main():
    """Test a checkpoint."""
    print("=" * 80)
    print("TEST CHECKPOINT")
    print("=" * 80)

    # List available checkpoints
    models_dir = Path("models")
    checkpoints = []

    if (models_dir / "best_model").exists():
        checkpoints.append("best_model")

    for path in models_dir.glob("checkpoint_epoch_*"):
        checkpoints.append(path.name)

    if not checkpoints:
        print("No checkpoints found in models/ directory")
        return

    print("\nAvailable checkpoints:")
    for i, checkpoint in enumerate(sorted(checkpoints), 1):
        print(f"  {i}. {checkpoint}")

    # Let user choose or use command line argument
    if len(sys.argv) > 1:
        checkpoint_name = sys.argv[1]
    else:
        print(f"\nUsage: python scripts/test_checkpoint.py <checkpoint_name>")
        print(f"Example: python scripts/test_checkpoint.py checkpoint_epoch_1")
        print(f"         python scripts/test_checkpoint.py best_model")

        # Auto-select first checkpoint for convenience
        checkpoint_name = sorted(checkpoints)[0] if checkpoints else None
        if checkpoint_name:
            print(f"\nAuto-selecting: {checkpoint_name}")

    if not checkpoint_name:
        return

    # Load model
    model_path = f"models/{checkpoint_name}"
    print(f"\nLoading model from: {model_path}")

    try:
        model = T5NLUModel.from_pretrained(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Test examples
    test_examples = [
        "set a timer for 5 minutes",
        "what is the weather",
        "open YouTube",
        "volume up",
        "remind me to eat dinner at 6 pm",
        "search Google for cats",
        "turn brightness down",
        "play music",
        "what time is it",
        "tell me a joke"
    ]

    print("\n" + "=" * 80)
    print("TESTING PREDICTIONS")
    print("=" * 80)

    for i, text in enumerate(test_examples, 1):
        print(f"\n{i}. Input: \"{text}\"")

        try:
            # Get raw model output
            input_text = f"nlu: {text}"
            inputs = model.tokenizer(
                input_text,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            )

            import torch
            device = next(model.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                    num_beams=4
                )

            raw_output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"   Raw model output: {raw_output}")

            # Now get parsed result
            result = model.predict(text)

            intent = result.get('intent', 'N/A')
            params = result.get('params', {})

            print(f"   Parsed Intent: {intent}")
            if params:
                print(f"   Parsed Params: {json.dumps(params, indent=11)[11:]}")  # Indent alignment
            else:
                print(f"   Parsed Params: (none)")

            # Show raw output if parsing failed
            if 'raw_output' in result:
                print(f"   ⚠ Warning: Fallback parsing used")

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n>>> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_input:
                continue

            result = model.predict(user_input)

            intent = result.get('intent', 'N/A')
            params = result.get('params', {})

            print(f"Intent: {intent}")
            if params:
                print(f"Params: {json.dumps(params, indent=2)}")
            else:
                print(f"Params: (none)")

            if 'raw_output' in result:
                print(f"⚠ Raw output: {result['raw_output']}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
