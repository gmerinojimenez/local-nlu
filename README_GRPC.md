# gRPC Service Quick Start

This guide will help you quickly set up and test the gRPC service for the NLU model.

## Prerequisites

1. A trained model checkpoint (e.g., from training)
2. Python 3.9+ with virtual environment activated

## Installation

### 1. Install gRPC dependencies

```bash
pip install grpcio grpcio-tools protobuf
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate gRPC code from proto files

```bash
./scripts/generate_grpc_code.sh
```

This creates:
- `nlu_service_pb2.py` - Message definitions
- `nlu_service_pb2_grpc.py` - Service definitions

## Quick Test

### 1. Start the server

Assuming you have a trained model at `checkpoints/best_model`:

```bash
python src/inference/grpc_server.py \
    --model-path checkpoints/best_model \
    --device cpu \
    --port 50051
```

For NPU (Ascend):
```bash
python src/inference/grpc_server.py \
    --model-path models/best_model \
    --device npu \
    --port 50051
```

With debug logging (to see raw model outputs):
```bash
python src/inference/grpc_server.py \
    --model-path models/best_model \
    --device cpu \
    --port 50051 \
    --log-level DEBUG
```

### 2. Test with the client

In another terminal, run:

```bash
# Health check
python examples/grpc_client.py --health

# Single prediction
python examples/grpc_client.py --text "set a timer for 5 minutes"

# Batch prediction
python examples/grpc_client.py --batch \
    "turn on the lights" \
    "play some music" \
    "what's the weather like"

# Interactive mode
python examples/grpc_client.py --interactive
```

## Example Output

### Health Check
```
============================================================
Health Check
============================================================
Status: Healthy
Model: checkpoints/best_model
Device: cpu
Parameters: 222,903,552
```

### Single Prediction
```
============================================================
Input: set a timer for 5 minutes
============================================================
Intent: SetTimer
Parameters:
  duration: 5
  unit: minutes
Inference time: 245.32ms
```

### Batch Prediction
```
============================================================
Batch prediction for 3 texts
============================================================
Total inference time: 582.15ms
Average per text: 194.05ms

[1] Input: turn on the lights
    Intent: TurnOnLights
    Parameters:
      device: lights

[2] Input: play some music
    Intent: PlayMusic
    Parameters: (none)

[3] Input: what's the weather like
    Intent: GetWeather
    Parameters: (none)
```

## Integration

### Python Client Example

```python
import grpc
import nlu_service_pb2
import nlu_service_pb2_grpc

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = nlu_service_pb2_grpc.NLUServiceStub(channel)

# Make prediction
request = nlu_service_pb2.PredictRequest(
    text="set a timer for 5 minutes"
)
response = stub.Predict(request)

print(f"Intent: {response.intent}")
print(f"Parameters: {dict(response.params)}")
print(f"Time: {response.inference_time_ms:.2f}ms")
```

### Other Languages

The `.proto` file can be used to generate clients in other languages:

- **Node.js**: Use `@grpc/grpc-js` and `@grpc/proto-loader`
- **Go**: Use `protoc-gen-go-grpc`
- **Java**: Use `protoc-gen-grpc-java`
- **C#**: Use `Grpc.Tools`

See [docs/GRPC_SERVICE.md](docs/GRPC_SERVICE.md) for detailed examples.

## Performance Tips

1. **Use batch predictions** for multiple utterances (more efficient)
2. **Reduce num_beams** for faster inference (default: 4)
3. **Use GPU/NPU** when available for better throughput
4. **Adjust max_workers** based on CPU cores

Example with optimized parameters:
```bash
python examples/grpc_client.py \
    --text "your text" \
    --num-beams 2 \
    --max-length 128
```

## Troubleshooting

### "Module 'nlu_service_pb2' not found"
Run the code generation script:
```bash
./scripts/generate_grpc_code.sh
```

### "Connection refused"
- Check that the server is running
- Verify the port matches (default: 50051)
- Check firewall settings

### "Model path does not exist"
- Ensure you have a trained model checkpoint
- Use the correct path to your saved model

### Slow inference
- Use `--device cuda` or `--device npu` for GPU/NPU acceleration
- Reduce `--num-beams` (e.g., 2 instead of 4)
- Use batch predictions for multiple texts

## Debugging

If you want to see the raw model outputs for debugging:

```bash
# Start server with debug logging
make server MODEL_PATH=models/best_model LOG_LEVEL=DEBUG

# Or directly
python src/inference/grpc_server.py \
    --model-path models/best_model \
    --log-level DEBUG
```

See [docs/DEBUG_LOGGING.md](docs/DEBUG_LOGGING.md) for detailed debugging guide.

## Next Steps

- Read the full documentation: [docs/GRPC_SERVICE.md](docs/GRPC_SERVICE.md)
- Learn about debug logging: [docs/DEBUG_LOGGING.md](docs/DEBUG_LOGGING.md)
- Deploy to production with Docker/Kubernetes
- Add TLS/SSL for secure communication
- Implement authentication and authorization
- Set up monitoring and logging

## Support

For issues or questions, please refer to:
- Full gRPC documentation: [docs/GRPC_SERVICE.md](docs/GRPC_SERVICE.md)
- Protocol Buffers definition: [proto/nlu_service.proto](proto/nlu_service.proto)
