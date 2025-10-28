"""
Script to preprocess the raw data and create train/val/test splits.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import NLUDataLoader


if __name__ == "__main__":
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    loader = NLUDataLoader()
    train_df, val_df, test_df = loader.process_and_save()

    print("\n✓ Data preprocessing complete!")
    print("\nNext step: Run training with:")
    print("  python scripts/train.py")
