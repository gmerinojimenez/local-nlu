"""
Test ONNX exported model using Optimum's ORTModelForSeq2SeqLM.
This provides a simpler interface that handles ONNX inference automatically.
"""
import sys
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_with_optimum(model_path: str):
    """
    Test model using Optimum's ONNX Runtime integration.

    Args:
        model_path: Path to ONNX model directory
    """
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import T5Tokenizer
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nPlease install required packages:")
        print('  pip install "optimum[onnxruntime]"')
        return

    print("=" * 80)
    print("TEST ONNX MODEL WITH OPTIMUM")
    print("=" * 80)

    model_path = Path(model_path)

    if not model_path.exists():
        print(f"✗ Model path not found: {model_path}")
        print("\nPlease run export_to_onnx.py first:")
        print("  python scripts/export_to_onnx.py models/best_model")
        return

    # Load model and tokenizer
    print(f"\nLoading ONNX model from: {model_path}")
    try:
        model = ORTModelForSeq2SeqLM.from_pretrained(str(model_path))
        print("✓ ONNX model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Load tokenizer - try multiple locations
    tokenizer_paths = [
        model_path.parent / "tokenizer",  # models/onnx/tokenizer
        model_path,  # models/onnx/t5_nlu_full
        model_path.parent.parent / "best_model",  # models/best_model
    ]

    tokenizer = None
    for tok_path in tokenizer_paths:
        if tok_path.exists():
            print(f"Trying tokenizer from: {tok_path}")
            try:
                tokenizer = T5Tokenizer.from_pretrained(str(tok_path))
                print(f"✓ Tokenizer loaded from: {tok_path}")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                continue

    if tokenizer is None:
        print("✗ Could not load tokenizer from any location")
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
            start_time = time.time()

            # Format input
            input_text = f"nlu: {text}"

            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            )

            # Generate
            output_ids = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

            # Decode
            raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000

            print(f"   Raw model output: {raw_output}")
            print(f"   Inference time: {inference_time:.2f}ms")

            # Parse JSON
            try:
                result = json.loads(raw_output)
                intent = result.get('intent', 'N/A')
                params = result.get('params', {})
            except json.JSONDecodeError:
                # Try to extract intent
                import re
                intent_match = re.search(r'"intent":\s*"([^"]+)"', raw_output)
                intent = intent_match.group(1) if intent_match else 'PARSE_ERROR'
                params = {}

            print(f"   Parsed Intent: {intent}")
            if params:
                print(f"   Parsed Params: {json.dumps(params, indent=11)[11:]}")
            else:
                print(f"   Parsed Params: (none)")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

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

            start_time = time.time()

            # Format and tokenize
            input_text = f"nlu: {user_input}"
            inputs = tokenizer(
                input_text,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            )

            # Generate
            output_ids = model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )

            # Decode
            raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            end_time = time.time()
            inference_time = (end_time - start_time) * 1000

            # Parse
            try:
                result = json.loads(raw_output)
                intent = result.get('intent', 'N/A')
                params = result.get('params', {})
            except json.JSONDecodeError:
                import re
                intent_match = re.search(r'"intent":\s*"([^"]+)"', raw_output)
                intent = intent_match.group(1) if intent_match else 'PARSE_ERROR'
                params = {}

            print(f"Intent: {intent}")
            if params:
                print(f"Params: {json.dumps(params, indent=2)}")
            else:
                print(f"Params: (none)")
            print(f"Inference time: {inference_time:.2f}ms")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function."""
    print("Optimum ONNX Runtime Test Script\n")

    if len(sys.argv) < 2:
        print("Usage: python scripts/test_checkpoint_optimum.py <onnx_model_path>")
        print("\nExamples:")
        print("  python scripts/test_checkpoint_optimum.py models/onnx/t5_nlu_full")
        print("\nFirst export your model to ONNX:")
        print("  python scripts/export_to_onnx.py models/best_model")
        print("\nNote: This script uses Optimum which automatically handles ONNX inference.")
        print("For Snapdragon NPU support, you'll need onnxruntime-qnn installed.")
        return

    model_path = sys.argv[1]
    test_with_optimum(model_path)


if __name__ == "__main__":
    main()
