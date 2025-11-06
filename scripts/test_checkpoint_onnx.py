"""
Test ONNX exported model with support for Snapdragon NPU via ONNX Runtime.
This script provides an alternative to PyTorch for running inference on NPU hardware.
"""
import sys
from pathlib import Path
import json
import time
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def test_with_onnxruntime(model_path: str, use_qnn: bool = False):
    """
    Test model using ONNX Runtime.

    Args:
        model_path: Path to ONNX model directory
        use_qnn: Whether to use QNN execution provider (for Snapdragon NPU)
    """
    try:
        import onnxruntime as ort
        from transformers import T5Tokenizer
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nPlease install required packages:")
        print("  pip install onnxruntime transformers")
        if use_qnn:
            print("  pip install onnxruntime-qnn  # For Snapdragon NPU support")
        return

    print("=" * 80)
    print("TEST ONNX MODEL WITH ONNX RUNTIME")
    print("=" * 80)

    model_path = Path(model_path)

    # Try to load full optimized model first
    if (model_path / "encoder_model.onnx").exists():
        print(f"\nLoading optimized ONNX model from: {model_path}")
        encoder_path = str(model_path / "encoder_model.onnx")
        decoder_path = str(model_path / "decoder_model.onnx")
        decoder_with_past_path = str(model_path / "decoder_with_past_model.onnx")
        use_optimized = True
    elif (model_path / "t5_nlu_encoder.onnx").exists():
        print(f"\nLoading basic ONNX model from: {model_path}")
        encoder_path = str(model_path / "t5_nlu_encoder.onnx")
        decoder_path = str(model_path / "t5_nlu_decoder.onnx")
        decoder_with_past_path = None
        use_optimized = False
    else:
        print(f"✗ No ONNX models found in: {model_path}")
        print("\nPlease run export_to_onnx.py first:")
        print("  python scripts/export_to_onnx.py models/best_model")
        return

    # Set up execution providers
    if use_qnn:
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
        print("\n🚀 Using QNN Execution Provider for Snapdragon NPU")
    else:
        providers = ['CPUExecutionProvider']
        print("\n💻 Using CPU Execution Provider")

    print(f"Available providers: {ort.get_available_providers()}")

    # Create sessions
    print("\nLoading ONNX sessions...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        encoder_session = ort.InferenceSession(
            encoder_path,
            sess_options,
            providers=providers
        )
        decoder_session = ort.InferenceSession(
            decoder_path,
            sess_options,
            providers=providers
        )
        print("✓ ONNX sessions created successfully!")
        print(f"Encoder running on: {encoder_session.get_providers()}")
        print(f"Decoder running on: {decoder_session.get_providers()}")

    except Exception as e:
        print(f"✗ Error creating ONNX sessions: {e}")
        if use_qnn:
            print("\n⚠ QNN provider might not be available. Falling back to CPU...")
            providers = ['CPUExecutionProvider']
            encoder_session = ort.InferenceSession(encoder_path, sess_options, providers=providers)
            decoder_session = ort.InferenceSession(decoder_path, sess_options, providers=providers)

    # Load tokenizer
    tokenizer_path = model_path / "tokenizer" if (model_path / "tokenizer").exists() else model_path.parent.parent / "best_model"
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path))
    print("✓ Tokenizer loaded!")

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

            # Tokenize input
            input_text = f"nlu: {text}"
            inputs = tokenizer(
                input_text,
                return_tensors='np',
                max_length=128,
                truncation=True,
                padding='max_length'
            )

            # Run encoder
            encoder_outputs = encoder_session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )

            # Simple greedy decoding (not beam search for simplicity)
            encoder_hidden_states = encoder_outputs[0]
            decoder_input_ids = np.array([[0]], dtype=np.int64)  # Start token

            output_tokens = [0]
            max_length = 256

            for _ in range(max_length):
                decoder_outputs = decoder_session.run(
                    None,
                    {
                        'decoder_input_ids': decoder_input_ids,
                        'encoder_attention_mask': inputs['attention_mask'].astype(np.int64),
                        'encoder_hidden_states': encoder_hidden_states
                    }
                )

                # Get next token (greedy)
                logits = decoder_outputs[0]
                next_token = np.argmax(logits[0, -1, :])

                output_tokens.append(next_token)

                # Stop if EOS token
                if next_token == 1:  # T5 EOS token
                    break

                # Update decoder input
                decoder_input_ids = np.array([output_tokens], dtype=np.int64)

            # Decode output
            raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True)

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
                # Try to extract intent and params
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

            # Tokenize
            input_text = f"nlu: {user_input}"
            inputs = tokenizer(
                input_text,
                return_tensors='np',
                max_length=128,
                truncation=True,
                padding='max_length'
            )

            # Run encoder
            encoder_outputs = encoder_session.run(
                None,
                {
                    'input_ids': inputs['input_ids'].astype(np.int64),
                    'attention_mask': inputs['attention_mask'].astype(np.int64)
                }
            )

            # Greedy decoding
            encoder_hidden_states = encoder_outputs[0]
            decoder_input_ids = np.array([[0]], dtype=np.int64)
            output_tokens = [0]

            for _ in range(256):
                decoder_outputs = decoder_session.run(
                    None,
                    {
                        'decoder_input_ids': decoder_input_ids,
                        'encoder_attention_mask': inputs['attention_mask'].astype(np.int64),
                        'encoder_hidden_states': encoder_hidden_states
                    }
                )

                logits = decoder_outputs[0]
                next_token = np.argmax(logits[0, -1, :])
                output_tokens.append(next_token)

                if next_token == 1:
                    break

                decoder_input_ids = np.array([output_tokens], dtype=np.int64)

            raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True)
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
    print("ONNX Runtime Test Script for Snapdragon NPU\n")

    if len(sys.argv) < 2:
        print("Usage: python scripts/test_checkpoint_onnx.py <onnx_model_path> [--npu]")
        print("\nExamples:")
        print("  python scripts/test_checkpoint_onnx.py models/onnx")
        print("  python scripts/test_checkpoint_onnx.py models/onnx --npu  # Use Snapdragon NPU")
        print("\nFirst export your model to ONNX:")
        print("  python scripts/export_to_onnx.py models/best_model")
        return

    model_path = sys.argv[1]
    use_qnn = '--npu' in sys.argv or '--qnn' in sys.argv

    if use_qnn:
        print("🚀 NPU mode enabled (requires onnxruntime-qnn and Snapdragon hardware)")
    else:
        print("💻 CPU mode (add --npu flag to use Snapdragon NPU)")

    print()
    test_with_onnxruntime(model_path, use_qnn)


if __name__ == "__main__":
    main()
