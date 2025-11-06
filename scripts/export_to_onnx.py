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

    # Export encoder
    print("\n1. Exporting encoder...")
    encoder_path = output_path / "t5_nlu_encoder.onnx"

    torch.onnx.export(
        model.model.encoder,
        (dummy_inputs['input_ids'], dummy_inputs['attention_mask']),
        str(encoder_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✓ Encoder exported to: {encoder_path}")

    # For decoder, we need to create decoder inputs
    print("\n2. Exporting decoder...")
    decoder_path = output_path / "t5_nlu_decoder.onnx"

    # Get encoder outputs first
    with torch.no_grad():
        encoder_outputs = model.model.encoder(
            input_ids=dummy_inputs['input_ids'],
            attention_mask=dummy_inputs['attention_mask']
        )

    # Create decoder inputs
    decoder_input_ids = torch.tensor([[0]])  # Start token

    torch.onnx.export(
        model.model.decoder,
        (
            decoder_input_ids,
            dummy_inputs['attention_mask'],
            encoder_outputs.last_hidden_state
        ),
        str(decoder_path),
        input_names=['decoder_input_ids', 'encoder_attention_mask', 'encoder_hidden_states'],
        output_names=['logits'],
        dynamic_axes={
            'decoder_input_ids': {0: 'batch', 1: 'decoder_sequence'},
            'encoder_attention_mask': {0: 'batch', 1: 'encoder_sequence'},
            'encoder_hidden_states': {0: 'batch', 1: 'encoder_sequence'},
            'logits': {0: 'batch', 1: 'decoder_sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    print(f"✓ Decoder exported to: {decoder_path}")

    # Also try to export the full model using transformers' optimum library
    print("\n3. Attempting to export full model using Optimum (if available)...")
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
