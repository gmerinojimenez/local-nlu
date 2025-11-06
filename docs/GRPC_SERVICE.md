# gRPC Service for NLU Model

This document describes how to set up and use the gRPC service for serving the NLU model.

## Overview

The gRPC service provides a high-performance API for NLU inference with the following features:

- **Single predictions**: Predict intent and parameters for a single utterance
- **Batch predictions**: Process multiple utterances in a single request
- **Health checks**: Monitor service status and model information
- **Streaming support**: Binary protocol for efficient data transfer
- **Strong typing**: Protocol Buffers ensure type safety

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `grpcio`: gRPC runtime
- `grpcio-tools`: Tools for generating Python code from .proto files
- `protobuf`: Protocol Buffers library

### 2. Generate gRPC Code

Run the code generation script to create Python files from the Protocol Buffer definitions:

```bash
chmod +x scripts/generate_grpc_code.sh
./scripts/generate_grpc_code.sh
```

This will generate:
- `nlu_service_pb2.py`: Message definitions
- `nlu_service_pb2_grpc.py`: Service definitions

## Running the Server

### Basic Usage

```bash
python src/inference/grpc_server.py \
    --model-path /path/to/saved/model \
    --host 0.0.0.0 \
    --port 50051
```

### With NPU Support

```bash
python src/inference/grpc_server.py \
    --model-path /path/to/saved/model \
    --device npu \
    --port 50051
```

### With GPU Support

```bash
python src/inference/grpc_server.py \
    --model-path /path/to/saved/model \
    --device cuda \
    --port 50051
```

### Server Options

- `--model-path`: Path to the saved model checkpoint (required)
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 50051)
- `--max-workers`: Maximum number of worker threads (default: 10)
- `--device`: Device for inference - cpu, cuda, or npu (default: cpu)

## Using the Client

### Health Check

```bash
python examples/grpc_client.py --health
```

### Single Prediction

```bash
python examples/grpc_client.py --text "set a timer for 5 minutes"
```

### Batch Prediction

```bash
python examples/grpc_client.py --batch \
    "set a timer for 5 minutes" \
    "what's the weather like today" \
    "play some music"
```

### Interactive Mode

```bash
python examples/grpc_client.py --interactive
```

This will start an interactive session where you can enter text and get predictions in real-time.

### Client Options

- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 50051)
- `--text`: Single text to predict
- `--batch`: Multiple texts for batch prediction
- `--health`: Perform health check only
- `--interactive`: Run in interactive mode
- `--max-length`: Maximum generation length (default: 256)
- `--num-beams`: Number of beams for beam search (default: 4)

## API Reference

### Service Methods

#### Predict

Single prediction for one utterance.

**Request:**
```protobuf
message PredictRequest {
  string text = 1;                // Input utterance
  optional int32 max_length = 2;  // Maximum generation length (default: 256)
  optional int32 num_beams = 3;   // Number of beams (default: 4)
}
```

**Response:**
```protobuf
message PredictResponse {
  string intent = 1;                    // Predicted intent
  map<string, string> params = 2;       // Extracted parameters
  optional string raw_output = 3;       // Raw model output
  optional string error = 4;            // Error message if any
  float inference_time_ms = 5;          // Inference time
}
```

#### PredictBatch

Batch prediction for multiple utterances.

**Request:**
```protobuf
message PredictBatchRequest {
  repeated string texts = 1;      // List of input utterances
  optional int32 max_length = 2;  // Maximum generation length
  optional int32 num_beams = 3;   // Number of beams
}
```

**Response:**
```protobuf
message PredictBatchResponse {
  repeated PredictResponse predictions = 1;  // List of predictions
  float total_inference_time_ms = 2;         // Total inference time
}
```

#### HealthCheck

Check service health and get model information.

**Request:**
```protobuf
message HealthCheckRequest {}
```

**Response:**
```protobuf
message HealthCheckResponse {
  bool is_healthy = 1;         // Server health status
  string model_name = 2;       // Loaded model name
  string device = 3;           // Device (cpu, cuda, npu)
  int64 model_parameters = 4;  // Number of parameters
}
```

## Integration Examples

### Python Client

```python
import grpc
import nlu_service_pb2
import nlu_service_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = nlu_service_pb2_grpc.NLUServiceStub(channel)

# Single prediction
request = nlu_service_pb2.PredictRequest(text="set a timer for 5 minutes")
response = stub.Predict(request)
print(f"Intent: {response.intent}")
print(f"Params: {dict(response.params)}")

# Batch prediction
batch_request = nlu_service_pb2.PredictBatchRequest(
    texts=["turn on the lights", "play music"]
)
batch_response = stub.PredictBatch(batch_request)
for pred in batch_response.predictions:
    print(f"Intent: {pred.intent}, Params: {dict(pred.params)}")
```

### Node.js Client

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

// Load proto
const packageDefinition = protoLoader.loadSync('proto/nlu_service.proto');
const nlu = grpc.loadPackageDefinition(packageDefinition).nlu;

// Create client
const client = new nlu.NLUService(
  'localhost:50051',
  grpc.credentials.createInsecure()
);

// Make prediction
client.Predict({ text: 'set a timer for 5 minutes' }, (error, response) => {
  if (error) {
    console.error(error);
  } else {
    console.log('Intent:', response.intent);
    console.log('Params:', response.params);
  }
});
```

### Go Client

```go
package main

import (
    "context"
    "log"

    "google.golang.org/grpc"
    pb "path/to/nlu_service"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    client := pb.NewNLUServiceClient(conn)

    response, err := client.Predict(context.Background(), &pb.PredictRequest{
        Text: "set a timer for 5 minutes",
    })
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Intent: %s", response.Intent)
    log.Printf("Params: %v", response.Params)
}
```

## Performance Considerations

### Optimization Tips

1. **Use batch predictions** for multiple utterances to reduce overhead
2. **Adjust num_beams** based on accuracy vs. speed trade-off (lower is faster)
3. **Use GPU/NPU** when available for better throughput
4. **Tune max_workers** based on your CPU cores and memory
5. **Enable connection pooling** in production clients

### Benchmarking

You can benchmark the service with the client:

```bash
# Single prediction latency
time python examples/grpc_client.py --text "test utterance"

# Batch throughput
time python examples/grpc_client.py --batch \
    "utterance 1" "utterance 2" ... "utterance N"
```

## Production Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x scripts/generate_grpc_code.sh && \
    ./scripts/generate_grpc_code.sh

EXPOSE 50051

CMD ["python", "src/inference/grpc_server.py", \
     "--model-path", "/models/checkpoint", \
     "--host", "0.0.0.0", \
     "--port", "50051"]
```

Build and run:

```bash
docker build -t nlu-grpc-server .
docker run -p 50051:50051 -v /path/to/model:/models nlu-grpc-server
```

### Kubernetes Deployment

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlu-grpc-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nlu-grpc-server
  template:
    metadata:
      labels:
        app: nlu-grpc-server
    spec:
      containers:
      - name: nlu-server
        image: nlu-grpc-server:latest
        ports:
        - containerPort: 50051
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: nlu-grpc-service
spec:
  selector:
    app: nlu-grpc-server
  ports:
  - protocol: TCP
    port: 50051
    targetPort: 50051
  type: LoadBalancer
```

### Load Balancing

gRPC supports client-side and server-side load balancing:

```python
# Client-side load balancing
channel = grpc.insecure_channel(
    'dns:///nlu-service:50051',
    options=[('grpc.lb_policy_name', 'round_robin')]
)
```

## Security

### TLS/SSL

For production, use secure channels with TLS:

**Server:**
```python
import grpc
from grpc import ssl_server_credentials

# Read credentials
with open('server.key', 'rb') as f:
    private_key = f.read()
with open('server.crt', 'rb') as f:
    certificate_chain = f.read()

# Create secure server
server_credentials = ssl_server_credentials(
    [(private_key, certificate_chain)]
)
server.add_secure_port('[::]:50051', server_credentials)
```

**Client:**
```python
import grpc
from grpc import ssl_channel_credentials

# Read credentials
with open('ca.crt', 'rb') as f:
    trusted_certs = f.read()

# Create secure channel
credentials = ssl_channel_credentials(trusted_certs)
channel = grpc.secure_channel('server:50051', credentials)
```

### Authentication

Implement token-based authentication:

```python
class AuthInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        # Extract and verify token
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get('authorization')

        if not verify_token(token):
            return grpc.unary_unary_rpc_method_handler(
                lambda request, context: context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    'Invalid token'
                )
            )

        return continuation(handler_call_details)
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure server is running and firewall allows port 50051
2. **Module not found**: Run `./scripts/generate_grpc_code.sh` to generate Python code
3. **Out of memory**: Reduce batch size or max_workers
4. **Slow inference**: Use GPU/NPU or reduce num_beams

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check server logs for errors and performance metrics.

## Further Reading

- [gRPC Python Quickstart](https://grpc.io/docs/languages/python/quickstart/)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
- [gRPC Best Practices](https://grpc.io/docs/guides/performance/)
