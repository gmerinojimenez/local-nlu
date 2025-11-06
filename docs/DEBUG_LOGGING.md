# Debug Logging Guide

This guide explains how to enable and use debug logging in the gRPC server to troubleshoot model predictions.

## Enabling Debug Logging

### Option 1: Command Line Flag

Start the server with the `--log-level DEBUG` flag:

```bash
python src/inference/grpc_server.py \
    --model-path models/best_model \
    --device cpu \
    --log-level DEBUG
```

### Option 2: Using Makefile

```bash
make server MODEL_PATH=models/best_model LOG_LEVEL=DEBUG
```

### Option 3: Environment Variable

You can also set the log level in code or via environment:

```bash
export LOG_LEVEL=DEBUG
python src/inference/grpc_server.py --model-path models/best_model
```

## What Gets Logged

### At INFO Level (Default)

- Server startup information (model loaded, device used, etc.)
- Request completion logs with inference time
- Errors and exceptions

Example:
```
2025-11-06 10:44:38,331 - __main__ - INFO - Loading model from models/best_model
2025-11-06 10:44:42,156 - __main__ - INFO - Model loaded successfully on device: cpu
2025-11-06 10:44:42,156 - __main__ - INFO - Model parameters: 222,903,552
2025-11-06 10:44:42,156 - __main__ - INFO - NLU gRPC Server started on 0.0.0.0:50051
2025-11-06 10:44:50,234 - __main__ - INFO - Prediction completed in 245.32ms: set a timer for 5 minutes...
```

### At DEBUG Level

In addition to INFO logs, DEBUG level includes:

#### Single Predictions

1. **Input text and parameters**:
   ```
   DEBUG - Predict request - text: 'set a timer for 5 minutes', max_length: 256, num_beams: 4
   ```

2. **Raw model output** (complete dict returned by the model):
   ```
   DEBUG - Raw model output: {'intent': 'SetTimer', 'params': {'duration': 5, 'unit': 'minutes'}}
   ```

3. **Raw output field** (if model had parsing issues):
   ```
   DEBUG - Raw output field: {"intent": "SetTimer", "params": {"duration": 5, "unit": "minutes"}}
   ```

#### Batch Predictions

1. **Batch input information**:
   ```
   DEBUG - PredictBatch request - 3 texts, max_length: 256, num_beams: 4
   DEBUG - Batch texts: ['turn on the lights', 'play some music', 'what is the weather']
   ```

2. **Complete batch results**:
   ```
   DEBUG - Batch raw results: [{'intent': 'TurnOnLights', 'params': {...}}, ...]
   ```

3. **Individual result details**:
   ```
   DEBUG - Batch result [0]: {'intent': 'TurnOnLights', 'params': {'device': 'lights'}}
   DEBUG - Batch result [0] raw_output: {"intent": "TurnOnLights", "params": {"device": "lights"}}
   DEBUG - Batch result [1]: {'intent': 'PlayMusic', 'params': {}}
   ```

## Use Cases

### 1. Debugging Model Output Parsing

If you're getting `PARSE_ERROR` intents, enable debug logging to see the raw model output:

```bash
# Start server with debug logging
make server MODEL_PATH=models/best_model LOG_LEVEL=DEBUG

# In another terminal, test
python examples/grpc_client.py --text "your problematic text"
```

Look for the `Raw model output` and `Raw output field` logs to see what the model actually generated before parsing.

### 2. Verifying Input Processing

Check that your input text is being processed correctly:

```
DEBUG - Predict request - text: 'your input', max_length: 256, num_beams: 4
```

This helps verify:
- The text is being received correctly
- Parameters are being set as expected
- No text truncation or encoding issues

### 3. Analyzing Batch Prediction Issues

When batch predictions behave unexpectedly, debug logs show:

```
DEBUG - PredictBatch request - 5 texts, max_length: 256, num_beams: 4
DEBUG - Batch texts: ['text1', 'text2', 'text3', 'text4', 'text5']
DEBUG - Batch raw results: [{...}, {...}, {...}, {...}, {...}]
DEBUG - Batch result [0]: {...}
DEBUG - Batch result [1]: {...}
...
```

This helps identify:
- Which specific text in the batch is causing issues
- Whether the problem is in prediction or parsing
- Patterns across multiple inputs

### 4. Performance Troubleshooting

Debug logs include timing information that can help identify bottlenecks:

```
INFO - Prediction completed in 245.32ms: set a timer for 5 minutes...
INFO - Batch prediction completed in 582.15ms for 3 texts
```

Combined with debug logs, you can see:
- Model inference time
- Which parameters affect performance (num_beams, max_length)
- Batch vs. single prediction efficiency

## Example Debug Session

Here's a complete example of using debug logging to troubleshoot:

```bash
# 1. Start server with debug logging
python src/inference/grpc_server.py \
    --model-path models/best_model \
    --device cpu \
    --log-level DEBUG

# Server output shows:
# 2025-11-06 10:44:38,331 - __main__ - INFO - Loading model from models/best_model
# 2025-11-06 10:44:42,156 - __main__ - INFO - Model loaded successfully on device: cpu
# 2025-11-06 10:44:42,156 - __main__ - INFO - NLU gRPC Server started on 0.0.0.0:50051
```

```bash
# 2. In another terminal, make a prediction
python examples/grpc_client.py --text "set a timer for 5 minutes"

# Server logs show:
# 2025-11-06 10:45:10,234 - __main__ - DEBUG - Predict request - text: 'set a timer for 5 minutes', max_length: 256, num_beams: 4
# 2025-11-06 10:45:10,479 - __main__ - DEBUG - Raw model output: {'intent': 'SetTimer', 'params': {'duration': 5, 'unit': 'minutes'}}
# 2025-11-06 10:45:10,479 - __main__ - INFO - Prediction completed in 245.32ms: set a timer for 5 minutes...
```

If there's a parsing error:
```
# 2025-11-06 10:45:10,479 - __main__ - DEBUG - Raw model output: {'intent': 'PARSE_ERROR', 'params': {}, 'raw_output': '{"intent": "SetTimer", "params": {"duration": 5, "unit": "minu', 'error': 'Unterminated string starting at: line 1 column 45 (char 44)'}
# 2025-11-06 10:45:10,479 - __main__ - DEBUG - Raw output field: {"intent": "SetTimer", "params": {"duration": 5, "unit": "minu
```

This shows the model output was truncated, indicating you may need to increase `max_length`.

## Log Levels Reference

| Level | When to Use | What You See |
|-------|-------------|--------------|
| **DEBUG** | Development, troubleshooting model outputs | All logs including input/output details |
| **INFO** | Normal production use | Server status, request completion, timing |
| **WARNING** | Production with selective logging | Unusual conditions, fallbacks |
| **ERROR** | Production (minimal logging) | Only errors and failures |
| **CRITICAL** | Emergency debugging | Only critical failures |

## Filtering Logs

### View Only Your Logs

```bash
python src/inference/grpc_server.py ... 2>&1 | grep "__main__"
```

### Save Logs to File

```bash
python src/inference/grpc_server.py ... 2>&1 | tee server.log
```

### Filter for Specific Patterns

```bash
# Only show raw outputs
python src/inference/grpc_server.py ... 2>&1 | grep "Raw model output"

# Only show errors
python src/inference/grpc_server.py ... 2>&1 | grep "ERROR"

# Only show timing
python src/inference/grpc_server.py ... 2>&1 | grep "completed in"
```

## Tips

1. **Use DEBUG for development**: Always use DEBUG level when developing or training new models
2. **Use INFO for production**: Switch to INFO level for production deployments
3. **Log rotation**: For production, implement log rotation to manage disk space
4. **Structured logging**: Consider using structured logging (JSON) for easier parsing in production
5. **Monitor timing**: Watch the inference times to optimize batch sizes and parameters

## Advanced: Programmatic Logging

You can also add custom debug logs in your code:

```python
import logging
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Custom debug message: {variable}")
logger.info(f"Important event: {event}")
logger.warning(f"Warning condition: {condition}")
logger.error(f"Error occurred: {error}")
```

## Troubleshooting Common Issues

### Issue: No debug logs appear

**Solution**: Make sure you're setting the log level correctly:
```bash
--log-level DEBUG  # Not --log-level debug (case-sensitive)
```

### Issue: Too many logs from other libraries

**Solution**: Set the log level only for your module:
```python
# In grpc_server.py
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# But keep root logger at INFO
logging.basicConfig(level=logging.INFO)
```

### Issue: Can't see logs in real-time

**Solution**: Use unbuffered output:
```bash
python -u src/inference/grpc_server.py ... --log-level DEBUG
```

Or flush logs:
```bash
python src/inference/grpc_server.py ... --log-level DEBUG 2>&1 | tee /dev/tty
```
