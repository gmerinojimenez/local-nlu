#!/bin/bash
# Generate Python code from Protocol Buffer definitions

set -e

echo "Generating gRPC code from protobuf definitions..."

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Proto directory
PROTO_DIR="${PROJECT_ROOT}/proto"
OUTPUT_DIR="${PROJECT_ROOT}"

# Check if proto file exists
if [ ! -f "${PROTO_DIR}/nlu_service.proto" ]; then
    echo "Error: Proto file not found at ${PROTO_DIR}/nlu_service.proto"
    exit 1
fi

# Generate Python code
python3 -m grpc_tools.protoc \
    -I"${PROTO_DIR}" \
    --python_out="${OUTPUT_DIR}" \
    --grpc_python_out="${OUTPUT_DIR}" \
    "${PROTO_DIR}/nlu_service.proto"

echo "Generated files:"
echo "  - ${OUTPUT_DIR}/nlu_service_pb2.py"
echo "  - ${OUTPUT_DIR}/nlu_service_pb2_grpc.py"
echo "Done!"
