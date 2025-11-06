.PHONY: help install-grpc generate-grpc server client clean

help:
	@echo "Available commands:"
	@echo "  make install-grpc    - Install gRPC dependencies"
	@echo "  make generate-grpc   - Generate Python code from .proto files"
	@echo "  make server          - Start the gRPC server (requires MODEL_PATH)"
	@echo "  make client          - Run example client"
	@echo "  make clean           - Remove generated gRPC files"
	@echo ""
	@echo "Usage examples:"
	@echo "  make generate-grpc"
	@echo "  make server MODEL_PATH=models/best_model"
	@echo "  make server MODEL_PATH=models/best_model DEVICE=npu"
	@echo "  make server MODEL_PATH=models/best_model LOG_LEVEL=DEBUG"
	@echo "  make client TEXT='set a timer for 5 minutes'"

install-grpc:
	pip install grpcio grpcio-tools protobuf

generate-grpc:
	@echo "Generating gRPC code from proto files..."
	./scripts/generate_grpc_code.sh

server:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH is required"; \
		echo "Usage: make server MODEL_PATH=path/to/model"; \
		exit 1; \
	fi
	python src/inference/grpc_server.py \
		--model-path $(MODEL_PATH) \
		--device $(or $(DEVICE),cpu) \
		--port $(or $(PORT),50051) \
		--host $(or $(HOST),0.0.0.0) \
		--log-level $(or $(LOG_LEVEL),INFO)

client:
	@if [ -n "$(TEXT)" ]; then \
		python examples/grpc_client.py --text "$(TEXT)"; \
	elif [ -n "$(BATCH)" ]; then \
		python examples/grpc_client.py --batch $(BATCH); \
	elif [ "$(INTERACTIVE)" = "true" ]; then \
		python examples/grpc_client.py --interactive; \
	elif [ "$(HEALTH)" = "true" ]; then \
		python examples/grpc_client.py --health; \
	else \
		python examples/grpc_client.py; \
	fi

clean:
	rm -f nlu_service_pb2.py nlu_service_pb2_grpc.py
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
