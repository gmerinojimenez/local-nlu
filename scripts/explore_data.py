"""
Script to explore and analyze the dataset structure.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def explore_dataset(data_path):
    """Load and explore the dataset."""
    print("=" * 80)
    print("DATASET EXPLORATION")
    print("=" * 80)

    # Load the Excel file
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_excel(data_path)

    # Basic information
    print(f"\n{'Dataset Shape:':<30} {df.shape[0]} rows × {df.shape[1]} columns")

    # Column names
    print(f"\n{'Columns:':<30}")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # Data types
    print(f"\n{'Data Types:':<30}")
    print(df.dtypes.to_string())

    # Missing values
    print(f"\n{'Missing Values:':<30}")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    else:
        print("  No missing values")

    # Display first few rows
    print("\n" + "=" * 80)
    print("FIRST 5 ROWS:")
    print("=" * 80)
    print(df.head().to_string())

    # Check for potential intent column
    print("\n" + "=" * 80)
    print("ANALYZING POTENTIAL INTENT/LABEL COLUMNS:")
    print("=" * 80)

    for col in df.columns:
        unique_count = df[col].nunique()
        # Likely candidates for intent/label columns (categorical with reasonable number of unique values)
        if unique_count < 100 and df[col].dtype in ['object', 'str']:
            print(f"\nColumn: '{col}'")
            print(f"  Unique values: {unique_count}")
            print(f"  Sample values: {df[col].value_counts().head(10).to_dict()}")

    # Look for text/utterance columns (likely to be longer strings)
    print("\n" + "=" * 80)
    print("ANALYZING POTENTIAL TEXT/UTTERANCE COLUMNS:")
    print("=" * 80)

    for col in df.columns:
        if df[col].dtype in ['object', 'str']:
            # Calculate average length of strings in this column
            avg_length = df[col].dropna().astype(str).str.len().mean()
            if avg_length > 10:  # Text columns typically have longer strings
                print(f"\nColumn: '{col}'")
                print(f"  Average length: {avg_length:.1f} characters")
                print(f"  Sample texts:")
                for text in df[col].dropna().head(3):
                    text_str = str(text)[:100]  # First 100 chars
                    print(f"    - {text_str}...")

    # Statistical summary for numeric columns
    if df.select_dtypes(include=['number']).shape[1] > 0:
        print("\n" + "=" * 80)
        print("NUMERIC COLUMNS SUMMARY:")
        print("=" * 80)
        print(df.describe().to_string())

    # Save a sample to CSV for easier inspection
    sample_path = Path(data_path).parent.parent / "processed" / "sample_data.csv"
    df.head(20).to_csv(sample_path, index=False)
    print(f"\n{'Sample saved to:':<30} {sample_path}")

    return df

if __name__ == "__main__":
    # Path to the dataset
    data_path = "data/raw/all_origin_utterances_20240626_with_current_nli_response.xlsx"

    try:
        df = explore_dataset(data_path)

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        print("""
Next steps based on the dataset structure:
1. Identify the main text/utterance column
2. Identify the intent/label column
3. Check if there are parameter/entity columns
4. Determine if this is:
   - Intent classification only
   - Intent + parameter extraction (slot filling)
   - Intent + entity recognition
        """)

    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nPlease check:")
        print("  1. The file path is correct")
        print("  2. The Excel file is not corrupted")
        print("  3. You have the required permissions")
