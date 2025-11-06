# Deploying to Snapdragon X NPU

This guide explains how to deploy your T5 NLU model to Qualcomm Snapdragon X Elite NPU for efficient on-device inference.

## Overview

The Snapdragon X Elite features a powerful Hexagon NPU that can accelerate AI inference. To use it with your PyTorch model, you need to:

1. Export your model to ONNX format
2. Use ONNX Runtime with QNN Execution Provider
3. Run inference with NPU acceleration

## Prerequisites

### System Requirements
- Windows 11 PC with Snapdragon X Elite/Plus processor
- Python 3.8+ (AMD64 version, not ARM64)
- At least 8GB RAM

### Software Installation

```bash
# Install ONNX export tools
pip install optimum[onnxruntime]

# Install ONNX Runtime with QNN support for Snapdragon NPU
pip install onnxruntime-qnn

# Alternatively, for CPU-only testing:
pip install onnxruntime
```

## Step 1: Export Model to ONNX

Use the provided export script to convert your PyTorch model to ONNX format:

```bash
# Export best_model
python scripts/export_to_onnx.py models/best_model

# Or export final_model
python scripts/export_to_onnx.py models/final_model models/onnx_export
```

This will create:
- `models/onnx/t5_nlu_encoder.onnx` - Encoder model
- `models/onnx/t5_nlu_decoder.onnx` - Decoder model
- `models/onnx/t5_nlu_full/` - Optimized full model (if Optimum is installed)
- `models/onnx/tokenizer/` - Tokenizer configuration

### Export Output

```
Exported files in: models/onnx
├── t5_nlu_encoder.onnx          # Basic encoder
├── t5_nlu_decoder.onnx          # Basic decoder
├── t5_nlu_full/                 # Optimized version
│   ├── encoder_model.onnx
│   ├── decoder_model.onnx
│   └── decoder_with_past_model.onnx  # With KV cache optimization
└── tokenizer/                   # Tokenizer files
```

## Step 2: Test with ONNX Runtime

### CPU Testing (No NPU)

First, test on CPU to ensure the export worked correctly:

```bash
python scripts/test_checkpoint_onnx.py models/onnx
```

### NPU Testing (Snapdragon X Elite)

Enable NPU acceleration with the `--npu` flag:

```bash
python scripts/test_checkpoint_onnx.py models/onnx --npu
```

This will use the QNN (Qualcomm Neural Network) Execution Provider to run inference on the Hexagon NPU.

## Performance Comparison

Expected performance on Snapdragon X Elite:

| Device | Avg Inference Time | Notes |
|--------|-------------------|-------|
| CPU (PyTorch) | ~500-800ms | Using torch CPU backend |
| GPU (DirectML) | ~200-400ms | Using Adreno GPU |
| **NPU (QNN)** | ~100-250ms | **Using Hexagon NPU** |

Note: Actual performance depends on model size, input length, and system load.

## Alternative: Using DirectML for GPU

If you want to use the Adreno GPU instead of NPU:

```bash
# Install DirectML backend
pip install torch-directml

# Use regular PyTorch test script
python scripts/test_checkpoint.py models/best_model
```

DirectML will automatically use the GPU when available.

## Troubleshooting

### QNN Provider Not Available

**Error:** `QNNExecutionProvider is not available`

**Solutions:**
1. Make sure you're on a Snapdragon X Elite/Plus device
2. Install `onnxruntime-qnn` (not just `onnxruntime`)
3. Use AMD64 Python, not ARM64
4. The script will automatically fall back to CPU

### ONNX Export Errors

**Error:** Export fails with shape mismatches

**Solutions:**
1. Ensure model is in eval mode (handled automatically)
2. Update to latest transformers: `pip install -U transformers`
3. Try exporting with `opset_version=13` instead of 14

### Slow Inference

**Issue:** NPU inference is slower than expected

**Solutions:**
1. Use the optimized models in `t5_nlu_full/` (requires Optimum)
2. Reduce `max_length` in generation
3. Use smaller batch sizes
4. Enable KV cache with `decoder_with_past_model.onnx`

## Using Nexa SDK (Experimental)

Nexa SDK is another option for NPU deployment, but currently has limited support for custom T5 models:

```bash
# Install Nexa SDK
pip install nexaai

# This is experimental - may not work with custom models
nexa onnx gen-text -m models/onnx/t5_nlu_full
```

**Note:** As of 2025, Nexa SDK primarily supports GGUF models and pre-optimized models from their hub. Custom T5 ONNX models may not be fully supported.

## Production Deployment

For production apps on Snapdragon devices:

1. **Export optimized ONNX models** with Optimum
2. **Quantize to INT8** for better NPU performance:
   ```bash
   # Using Qualcomm AI Hub
   qai-hub compile --model t5_nlu_full --device "Snapdragon X Elite" --quantize int8
   ```
3. **Implement caching** for repeated queries
4. **Use async inference** for better throughput

## Qualcomm AI Hub (Advanced)

For maximum optimization:

1. Sign up at https://aihub.qualcomm.com
2. Install Qualcomm AI Hub CLI:
   ```bash
   pip install qai-hub-models
   ```
3. Compile for Snapdragon X Elite:
   ```bash
   qai-hub compile --model models/onnx/t5_nlu_full --device "Snapdragon X Elite CRD"
   ```

This will create optimized context binaries specifically tuned for your device's NPU.

## Summary

| Method | Ease of Use | Performance | Best For |
|--------|-------------|-------------|----------|
| PyTorch CPU | ⭐⭐⭐⭐⭐ | ⭐⭐ | Development/Testing |
| PyTorch + DirectML | ⭐⭐⭐⭐ | ⭐⭐⭐ | Quick GPU acceleration |
| **ONNX + QNN** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **NPU deployment** |
| Qualcomm AI Hub | ⭐⭐ | ⭐⭐⭐⭐⭐ | Production optimization |

## Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Qualcomm AI Hub](https://aihub.qualcomm.com)
- [Windows Copilot+ PC Dev Guide](https://learn.microsoft.com/en-us/windows/ai/npu-devices/)
- [Snapdragon NPU SDK](https://www.qualcomm.com/developer/software/neural-processing-sdk)
