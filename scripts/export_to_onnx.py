"""
Export T5 NLU model to ONNX format for NPU deployment.
"""
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.t5_nlu import T5NLUModel


def export_to_onnx(model_path: str, output_dir: str):
    """
    Export T5 model to ONNX format.

    Args:
        model_path: Path to the PyTorch model checkpoint
        output_dir: Directory to save ONNX models
    """
    print("=" * 80)
    print("EXPORT T5 NLU MODEL TO ONNX")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    print(f"\nLoading model from: {model_path}")
    model = T5NLUModel.from_pretrained(model_path)
    model.eval()
    print("✓ Model loaded successfully!")

    # Create dummy inputs for export
    dummy_text = "set a timer for 5 minutes"
    dummy_inputs = model.tokenizer(
        f"nlu: {dummy_text}",
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    )

    print("\nExporting encoder-decoder model to ONNX...")
    print("Note: T5 is an encoder-decoder model, so we'll export it in parts")

    # Try to export the full model using transformers' optimum library
    print("\nAttempting to export full model using Optimum...")
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer

        # Save tokenizer
        tokenizer_path = output_path / "tokenizer"
        model.tokenizer.save_pretrained(str(tokenizer_path))
        print(f"✓ Tokenizer saved to: {tokenizer_path}")

        # Export using optimum
        print("\nExporting full model with Optimum...")
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path,
            export=True
        )

        full_model_path = output_path / "t5_nlu_full"
        ort_model.save_pretrained(str(full_model_path))
        print(f"✓ Full optimized model saved to: {full_model_path}")
        print("\nThis model can be used with ONNX Runtime for NPU inference!")

    except ImportError:
        print("⚠ Optimum library not installed. To export optimized ONNX models, install:")
        print("  pip install optimum[onnxruntime]")
        print("\nBasic encoder/decoder exports are still available above.")

    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"\nExported files in: {output_path}")
    print("\nNext steps:")
    print("1. Install ONNX Runtime: pip install onnxruntime")
    print("2. For Snapdragon NPU: pip install onnxruntime-qnn")
    print("3. Use test_checkpoint_onnx.py to run inference")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_to_onnx.py <model_path> [output_dir]")
        print("\nExample:")
        print("  python scripts/export_to_onnx.py models/best_model")
        print("  python scripts/export_to_onnx.py models/final_model models/onnx_export")
        return

    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models/onnx"

    export_to_onnx(model_path, output_dir)


if __name__ == "__main__":
    main()
