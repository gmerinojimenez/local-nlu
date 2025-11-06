"""
Example gRPC client for NLU Service.
"""
import argparse
import json
import sys
from pathlib import Path

import grpc

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import nlu_service_pb2
import nlu_service_pb2_grpc


def predict_single(stub, text: str, max_length: int = 256, num_beams: int = 4):
    """
    Make a single prediction request.

    Args:
        stub: gRPC stub
        text: Input text
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
    """
    print(f"\n{'='*60}")
    print(f"Input: {text}")
    print(f"{'='*60}")

    request = nlu_service_pb2.PredictRequest(
        text=text,
        max_length=max_length,
        num_beams=num_beams
    )

    try:
        response = stub.Predict(request)

        print(f"Intent: {response.intent}")
        print(f"Parameters:")
        for key, value in response.params.items():
            # Try to parse JSON if it's a complex type
            try:
                parsed_value = json.loads(value)
                print(f"  {key}: {parsed_value}")
            except:
                print(f"  {key}: {value}")

        print(f"Inference time: {response.inference_time_ms:.2f}ms")

        if response.error:
            print(f"Error: {response.error}")
        if response.raw_output:
            print(f"Raw output: {response.raw_output}")

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()}: {e.details()}")


def predict_batch(stub, texts: list, max_length: int = 256, num_beams: int = 4):
    """
    Make a batch prediction request.

    Args:
        stub: gRPC stub
        texts: List of input texts
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
    """
    print(f"\n{'='*60}")
    print(f"Batch prediction for {len(texts)} texts")
    print(f"{'='*60}")

    request = nlu_service_pb2.PredictBatchRequest(
        texts=texts,
        max_length=max_length,
        num_beams=num_beams
    )

    try:
        response = stub.PredictBatch(request)

        print(f"Total inference time: {response.total_inference_time_ms:.2f}ms")
        print(f"Average per text: {response.total_inference_time_ms / len(texts):.2f}ms\n")

        for i, (text, prediction) in enumerate(zip(texts, response.predictions)):
            print(f"[{i+1}] Input: {text}")
            print(f"    Intent: {prediction.intent}")
            print(f"    Parameters:", end="")
            if prediction.params:
                print()
                for key, value in prediction.params.items():
                    try:
                        parsed_value = json.loads(value)
                        print(f"      {key}: {parsed_value}")
                    except:
                        print(f"      {key}: {value}")
            else:
                print(" (none)")
            if prediction.error:
                print(f"    Error: {prediction.error}")
            print()

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()}: {e.details()}")


def health_check(stub):
    """
    Perform a health check.

    Args:
        stub: gRPC stub
    """
    print(f"\n{'='*60}")
    print("Health Check")
    print(f"{'='*60}")

    request = nlu_service_pb2.HealthCheckRequest()

    try:
        response = stub.HealthCheck(request)

        print(f"Status: {'Healthy' if response.is_healthy else 'Unhealthy'}")
        print(f"Model: {response.model_name}")
        print(f"Device: {response.device}")
        print(f"Parameters: {response.model_parameters:,}")

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.code()}: {e.details()}")


def interactive_mode(stub):
    """
    Interactive mode for testing predictions.

    Args:
        stub: gRPC stub
    """
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Enter text to predict (or 'quit' to exit)")
    print()

    while True:
        try:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in ('quit', 'exit', 'q'):
                break

            predict_single(stub, text)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="gRPC Client for NLU Service")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Server port (default: 50051)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to predict"
    )
    parser.add_argument(
        "--batch",
        type=str,
        nargs="+",
        help="Multiple texts for batch prediction"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Perform health check"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum generation length (default: 256)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)"
    )

    args = parser.parse_args()

    # Connect to server
    address = f"{args.host}:{args.port}"
    print(f"Connecting to {address}...")

    with grpc.insecure_channel(
        address,
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    ) as channel:
        stub = nlu_service_pb2_grpc.NLUServiceStub(channel)

        try:
            # Perform health check first
            health_check(stub)

            # Handle different modes
            if args.health:
                # Health check already done
                pass
            elif args.text:
                predict_single(stub, args.text, args.max_length, args.num_beams)
            elif args.batch:
                predict_batch(stub, args.batch, args.max_length, args.num_beams)
            elif args.interactive:
                interactive_mode(stub)
            else:
                # Default: run some example predictions
                print("\nRunning example predictions...")
                examples = [
                    "set a timer for 5 minutes",
                    "what's the weather like today",
                    "play some music",
                    "turn on the lights in the living room"
                ]
                predict_batch(stub, examples, args.max_length, args.num_beams)

        except grpc.RpcError as e:
            print(f"\nConnection failed: {e.code()}: {e.details()}")
            sys.exit(1)


if __name__ == "__main__":
    main()
