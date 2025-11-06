"""
gRPC Server for NLU Model Inference.
"""
import argparse
import logging
import time
from concurrent import futures
from pathlib import Path
import sys
import json

import grpc
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.t5_nlu import T5NLUModel

# Import generated protobuf code
import nlu_service_pb2
import nlu_service_pb2_grpc


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NLUServicer(nlu_service_pb2_grpc.NLUServiceServicer):
    """Implementation of NLU Service."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the NLU servicer with a trained model.

        Args:
            model_path: Path to the saved model checkpoint
            device: Device to run inference on (cpu, cuda, npu)
        """
        logger.info(f"Loading model from {model_path}")
        self.model = T5NLUModel.from_pretrained(model_path)

        # Set device
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        elif device == "npu":
            try:
                import torch_npu
                if torch.npu.is_available():
                    logger.info("Using NPU for inference")
                else:
                    logger.warning("NPU not available, falling back to CPU")
                    self.device = "cpu"
            except ImportError:
                logger.warning("torch_npu not available, falling back to CPU")
                self.device = "cpu"

        self.model.model.to(self.device)
        self.model.model.eval()

        logger.info(f"Model loaded successfully on device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

    def Predict(self, request, context):
        """
        Handle single prediction request.

        Args:
            request: PredictRequest
            context: gRPC context

        Returns:
            PredictResponse
        """
        try:
            start_time = time.time()

            # Get parameters with defaults
            max_length = request.max_length if request.max_length > 0 else 256
            num_beams = request.num_beams if request.num_beams > 0 else 4

            # Debug: Log input
            logger.debug(f"Predict request - text: '{request.text}', max_length: {max_length}, num_beams: {num_beams}")

            # Run prediction
            result = self.model.predict(
                text=request.text,
                max_length=max_length,
                num_beams=num_beams
            )

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Debug: Log raw model output
            logger.debug(f"Raw model output: {result}")
            if "raw_output" in result:
                logger.debug(f"Raw output field: {result['raw_output']}")

            # Convert params dict to string map for protobuf
            params_map = {}
            if "params" in result and isinstance(result["params"], dict):
                for key, value in result["params"].items():
                    # Convert all values to strings for the protobuf map
                    params_map[key] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

            # Build response
            response = nlu_service_pb2.PredictResponse(
                intent=result.get("intent", "UNKNOWN"),
                params=params_map,
                inference_time_ms=inference_time
            )

            # Add optional fields if present
            if "raw_output" in result:
                response.raw_output = result["raw_output"]
            if "error" in result:
                response.error = result["error"]

            logger.info(f"Prediction completed in {inference_time:.2f}ms: {request.text[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction failed: {str(e)}")
            return nlu_service_pb2.PredictResponse(
                intent="ERROR",
                params={},
                error=str(e),
                inference_time_ms=0.0
            )

    def PredictBatch(self, request, context):
        """
        Handle batch prediction request.

        Args:
            request: PredictBatchRequest
            context: gRPC context

        Returns:
            PredictBatchResponse
        """
        try:
            start_time = time.time()

            # Get parameters with defaults
            max_length = request.max_length if request.max_length > 0 else 256
            num_beams = request.num_beams if request.num_beams > 0 else 4

            # Debug: Log input
            logger.debug(f"PredictBatch request - {len(request.texts)} texts, max_length: {max_length}, num_beams: {num_beams}")
            logger.debug(f"Batch texts: {list(request.texts)}")

            # Run batch prediction
            results = self.model.predict_batch(
                texts=list(request.texts),
                max_length=max_length,
                num_beams=num_beams
            )

            total_inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Debug: Log raw results
            logger.debug(f"Batch raw results: {results}")

            # Build responses
            predictions = []
            for i, result in enumerate(results):
                # Debug: Log individual result
                logger.debug(f"Batch result [{i}]: {result}")
                if "raw_output" in result:
                    logger.debug(f"Batch result [{i}] raw_output: {result['raw_output']}")
                # Convert params dict to string map
                params_map = {}
                if "params" in result and isinstance(result["params"], dict):
                    for key, value in result["params"].items():
                        params_map[key] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

                prediction = nlu_service_pb2.PredictResponse(
                    intent=result.get("intent", "UNKNOWN"),
                    params=params_map,
                    inference_time_ms=0.0  # Individual time not tracked in batch
                )

                if "raw_output" in result:
                    prediction.raw_output = result["raw_output"]
                if "error" in result:
                    prediction.error = result["error"]

                predictions.append(prediction)

            logger.info(f"Batch prediction completed in {total_inference_time:.2f}ms for {len(request.texts)} texts")

            return nlu_service_pb2.PredictBatchResponse(
                predictions=predictions,
                total_inference_time_ms=total_inference_time
            )

        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch prediction failed: {str(e)}")
            return nlu_service_pb2.PredictBatchResponse(
                predictions=[],
                total_inference_time_ms=0.0
            )

    def HealthCheck(self, request, context):
        """
        Handle health check request.

        Args:
            request: HealthCheckRequest
            context: gRPC context

        Returns:
            HealthCheckResponse
        """
        try:
            return nlu_service_pb2.HealthCheckResponse(
                is_healthy=True,
                model_name=self.model.model_name,
                device=self.device,
                model_parameters=self.model.count_parameters()
            )
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}", exc_info=True)
            return nlu_service_pb2.HealthCheckResponse(
                is_healthy=False,
                model_name="",
                device="",
                model_parameters=0
            )


def serve(model_path: str, host: str = "0.0.0.0", port: int = 50051,
          max_workers: int = 10, device: str = "cpu"):
    """
    Start the gRPC server.

    Args:
        model_path: Path to the saved model checkpoint
        host: Host address to bind to
        port: Port to listen on
        max_workers: Maximum number of worker threads
        device: Device to run inference on
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50 MB
        ]
    )

    # Add servicer to server
    nlu_service_pb2_grpc.add_NLUServiceServicer_to_server(
        NLUServicer(model_path, device),
        server
    )

    # Bind to address
    address = f"{host}:{port}"
    server.add_insecure_port(address)

    # Start server
    server.start()
    logger.info(f"NLU gRPC Server started on {address}")
    logger.info(f"Using device: {device}")
    logger.info("Press Ctrl+C to stop...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace=5)
        logger.info("Server stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="gRPC Server for NLU Model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen on (default: 50051)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of worker threads (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu"],
        help="Device to run inference on (default: cpu)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)

    # Verify model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)

    # Start server
    serve(
        model_path=str(model_path),
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        device=args.device
    )


if __name__ == "__main__":
    main()
