#!/usr/bin/env python3
"""
Fix tokenizer.json to be compatible with DJL tokenizers library.

The DJL tokenizers library (versions < 0.30.0) doesn't support the 'prepend_scheme'
field in the Metaspace pre_tokenizer. This script removes that field to ensure compatibility.
"""

import json
import sys
from pathlib import Path


def fix_tokenizer_json(tokenizer_path: Path) -> bool:
    """
    Fix tokenizer.json by removing unsupported fields.

    Args:
        tokenizer_path: Path to tokenizer.json file

    Returns:
        True if changes were made, False otherwise
    """
    print(f"Loading tokenizer from: {tokenizer_path}")

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Check if pre_tokenizer exists and has the problematic field
    if 'pre_tokenizer' not in tokenizer_data:
        print("No pre_tokenizer found in tokenizer.json")
        return False

    pre_tokenizer = tokenizer_data['pre_tokenizer']

    # Check if prepend_scheme exists
    if 'prepend_scheme' not in pre_tokenizer:
        print("No prepend_scheme field found - tokenizer is already compatible")
        return False

    print(f"Found pre_tokenizer type: {pre_tokenizer.get('type')}")
    print(f"Removing incompatible field: prepend_scheme = {pre_tokenizer['prepend_scheme']}")

    # Remove the problematic field
    del pre_tokenizer['prepend_scheme']

    # For Metaspace tokenizer in older format, we may also need to adjust the structure
    # The old format expected: "add_prefix_space" instead of "prepend_scheme"
    if pre_tokenizer.get('type') == 'Metaspace':
        # Map prepend_scheme value to add_prefix_space (if needed)
        # prepend_scheme: "always" -> add_prefix_space: True
        # prepend_scheme: "never" -> add_prefix_space: False
        # prepend_scheme: "first" -> add_prefix_space: False
        pre_tokenizer['add_prefix_space'] = True  # T5 always prepends

    # Create backup
    backup_path = tokenizer_path.with_suffix('.json.backup')
    print(f"Creating backup at: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        with open(tokenizer_path, 'r', encoding='utf-8') as orig:
            f.write(orig.read())

    # Save fixed tokenizer
    print(f"Saving fixed tokenizer to: {tokenizer_path}")
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

    print("✓ Tokenizer fixed successfully!")
    print("\nChanges made:")
    print("  - Removed 'prepend_scheme' field")
    print("  - Added 'add_prefix_space' field (for backward compatibility)")

    return True


def main():
    """Main function."""
    if len(sys.argv) < 2:
        # Default path
        tokenizer_path = Path(__file__).parent.parent / "models" / "onnx" / "tokenizer" / "tokenizer.json"
    else:
        tokenizer_path = Path(sys.argv[1])

    if not tokenizer_path.exists():
        print(f"Error: Tokenizer file not found: {tokenizer_path}")
        print("\nUsage:")
        print("  python scripts/fix_tokenizer_for_djl.py [path/to/tokenizer.json]")
        print("\nExample:")
        print("  python scripts/fix_tokenizer_for_djl.py models/onnx/tokenizer/tokenizer.json")
        return 1

    try:
        fixed = fix_tokenizer_json(tokenizer_path)
        if fixed:
            print("\n" + "=" * 80)
            print("SUCCESS")
            print("=" * 80)
            print("\nYou can now run the Kotlin NLU application:")
            print("  cd kotlin-nlu")
            print("  ./gradlew run")
            print("\nIf you want to revert the changes:")
            print(f"  mv {tokenizer_path}.backup {tokenizer_path}")
        return 0
    except Exception as e:
        print(f"Error fixing tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
