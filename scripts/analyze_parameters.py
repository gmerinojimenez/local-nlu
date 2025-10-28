"""
Script to analyze the parameter structure in the NLI Output.
"""
import pandas as pd
import json
from collections import defaultdict, Counter
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def analyze_parameters(data_path):
    """Analyze the parameter structure across all intents."""
    print("=" * 80)
    print("PARAMETER STRUCTURE ANALYSIS")
    print("=" * 80)

    # Load the dataset
    df = pd.read_excel(data_path)

    # Parse NLI Output JSON
    print("\nParsing NLI Output JSON structures...")

    intent_params = defaultdict(lambda: defaultdict(set))
    intent_param_types = defaultdict(lambda: defaultdict(Counter))
    intent_samples = defaultdict(list)

    for idx, row in df.iterrows():
        try:
            nli_output = json.loads(row['NLI Output'])
            if isinstance(nli_output, list) and len(nli_output) > 0:
                nli_data = nli_output[0]
                intent = nli_data.get('id', row['intent'])
                params = nli_data.get('params', {})

                # Store sample
                if len(intent_samples[intent]) < 3:
                    intent_samples[intent].append({
                        'transcript': row['transcript'],
                        'params': params
                    })

                # Analyze parameter keys and value types
                for param_key, param_value in params.items():
                    intent_params[intent][param_key].add(type(param_value).__name__)
                    intent_param_types[intent][param_key][type(param_value).__name__] += 1

                    # Track unique values for categorical parameters
                    if isinstance(param_value, str):
                        intent_param_types[intent][f"{param_key}_values"].update([param_value])
                    elif isinstance(param_value, (int, float)):
                        intent_param_types[intent][f"{param_key}_range"].update([param_value])

        except Exception as e:
            if idx < 5:  # Only print first few errors
                print(f"  Warning: Error parsing row {idx}: {e}")

    print(f"\nTotal intents analyzed: {len(intent_params)}")

    # Display parameter schema by intent
    print("\n" + "=" * 80)
    print("PARAMETER SCHEMA BY INTENT (Top 20 intents with parameters)")
    print("=" * 80)

    # Sort intents by number of samples
    intent_counts = df['intent'].value_counts()

    displayed = 0
    for intent in intent_counts.index:
        if intent not in intent_params or len(intent_params[intent]) == 0:
            continue

        if displayed >= 20:
            break

        print(f"\n{'─' * 80}")
        print(f"Intent: {intent}")
        print(f"Samples: {intent_counts[intent]}")
        print(f"Parameters:")

        for param_key, types in sorted(intent_params[intent].items()):
            type_str = ', '.join(types)
            count = sum(intent_param_types[intent][param_key].values())
            print(f"  • {param_key:<30} [{type_str}]  ({count} occurrences)")

            # Show value examples for strings
            if f"{param_key}_values" in intent_param_types[intent]:
                values = list(intent_param_types[intent][f"{param_key}_values"])[:10]
                if values:
                    print(f"    → Examples: {values}")

            # Show range for numbers
            elif f"{param_key}_range" in intent_param_types[intent]:
                values = list(intent_param_types[intent][f"{param_key}_range"])
                if values:
                    nums = [v for v in values if isinstance(v, (int, float))]
                    if nums:
                        print(f"    → Range: {min(nums)} to {max(nums)} (showing {len(nums[:10])} values)")

        # Show sample utterances
        print(f"  Sample utterances:")
        for sample in intent_samples[intent][:3]:
            print(f"    - \"{sample['transcript']}\"")
            print(f"      Params: {sample['params']}")

        displayed += 1

    # Statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)

    total_with_params = sum(1 for intent in intent_params if len(intent_params[intent]) > 0)
    total_without_params = len(df['intent'].unique()) - total_with_params

    print(f"Intents with parameters:    {total_with_params}")
    print(f"Intents without parameters: {total_without_params}")
    print(f"Total unique intents:       {len(df['intent'].unique())}")

    # Count utterances with/without parameters
    has_params = 0
    no_params = 0
    for idx, row in df.iterrows():
        try:
            nli_output = json.loads(row['NLI Output'])
            if isinstance(nli_output, list) and len(nli_output) > 0:
                params = nli_output[0].get('params', {})
                if params:
                    has_params += 1
                else:
                    no_params += 1
        except:
            no_params += 1

    print(f"\nUtterances with parameters:    {has_params:>6} ({has_params/len(df)*100:.1f}%)")
    print(f"Utterances without parameters: {no_params:>6} ({no_params/len(df)*100:.1f}%)")

    # Most common parameter keys across all intents
    all_param_keys = Counter()
    for intent in intent_params:
        for param_key in intent_params[intent]:
            all_param_keys[param_key] += 1

    print(f"\nMost common parameter keys (across all intents):")
    for key, count in all_param_keys.most_common(15):
        print(f"  {key:<30} (used in {count} intents)")

    return intent_params, intent_param_types

if __name__ == "__main__":
    data_path = "data/raw/all_origin_utterances_20240626_with_current_nli_response.xlsx"

    try:
        intent_params, intent_param_types = analyze_parameters(data_path)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
Based on this analysis, we need to decide on the parameter extraction approach:

Option A: Sequence Labeling (Token Classification)
  - Treat parameter extraction as Named Entity Recognition (NER)
  - Tag each token with BIO/BILOU tags
  - Works well for extracting spans of text (e.g., "50%" → percent value)

Option B: Generative Approach (Seq2Seq)
  - Generate the complete parameter JSON as output
  - More flexible but requires more data and training time

Option C: Multi-Task Classification
  - Predict intent + predict each parameter separately
  - Each parameter becomes a classification/regression head
  - Works when parameters have fixed vocabularies

RECOMMENDATION: Start with Option A (Sequence Labeling) for extracting parameter
values from text, combined with intent classification. This is the most standard
approach for NLU tasks.
        """)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
