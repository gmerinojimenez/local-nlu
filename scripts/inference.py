"""
Inference script for testing the NLU model with custom inputs.
"""
import sys
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel


def main():
    """Main inference function."""
    print("=" * 80)
    print("NLU MODEL INFERENCE")
    print("=" * 80)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']

    # Load best model
    model_path = paths['model_save_dir'] + '/best_model'
    print(f"\nLoading model from: {model_path}")

    model = T5NLUModel.from_pretrained(model_path)
    print("Model loaded successfully!")

    # Test examples
    test_examples = [
        "set a timer for 5 minutes",
        "what is the weather",
        "open YouTube",
        "volume up",
        "remind me to eat dinner at 6 pm",
        "search Google for artificial intelligence",
        "play music",
        "what time is it",
        "turn brightness down",
        "tell me a joke"
    ]

    print("\n" + "=" * 80)
    print("TESTING WITH SAMPLE UTTERANCES")
    print("=" * 80)

    for text in test_examples:
        print(f"\n{'─' * 80}")
        print(f"Input: \"{text}\"")

        result = model.predict(text)

        print(f"Intent: {result.get('intent', 'N/A')}")
        print(f"Parameters:")
        params = result.get('params', {})
        if params:
            print(json.dumps(params, indent=2))
        else:
            print("  (none)")

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Enter utterances to test (type 'quit' or 'exit' to stop):\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_input:
                continue

            result = model.predict(user_input)

            print(f"\nIntent: {result.get('intent', 'N/A')}")
            print(f"Parameters:")
            params = result.get('params', {})
            if params:
                print(json.dumps(params, indent=2))
            else:
                print("  (none)")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
