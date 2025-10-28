"""
Data loading and preprocessing for T5-based NLU model.
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import yaml


class NLUDataLoader:
    """Loads and preprocesses the NLU dataset for T5 training."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.random_seed = self.data_config['random_seed']

    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw Excel dataset."""
        data_path = self.data_config['raw_data_path']
        print(f"Loading data from {data_path}...")
        df = pd.read_excel(data_path)
        print(f"Loaded {len(df)} samples")
        return df

    def parse_nli_output(self, nli_output_str: str) -> Dict:
        """Parse the NLI Output JSON string."""
        try:
            nli_data = json.loads(nli_output_str)
            if isinstance(nli_data, list) and len(nli_data) > 0:
                return nli_data[0]
            return {}
        except Exception as e:
            print(f"Error parsing NLI output: {e}")
            return {}

    def format_for_t5(self, transcript: str, intent: str, params: Dict) -> Tuple[str, str]:
        """
        Format data for T5 text-to-text training.

        Input format: "nlu: {transcript}"
        Output format: JSON string with intent and params

        Args:
            transcript: User utterance
            intent: Intent label
            params: Parameter dictionary

        Returns:
            Tuple of (input_text, output_text)
        """
        # Input: prefix with "nlu:" to indicate the task
        input_text = f"nlu: {transcript}"

        # Output: JSON structure with intent and params
        output_data = {
            "intent": intent,
            "params": params
        }
        output_text = json.dumps(output_data, ensure_ascii=False)

        return input_text, output_text

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset and format for T5.

        Args:
            df: Raw dataframe

        Returns:
            Processed dataframe with input_text and output_text columns
        """
        print("Preprocessing dataset...")
        processed_data = []

        for idx, row in df.iterrows():
            transcript = row['transcript']
            intent = row['intent']

            # Parse NLI Output to extract parameters
            nli_data = self.parse_nli_output(row['NLI Output'])
            params = nli_data.get('params', {})

            # Format for T5
            input_text, output_text = self.format_for_t5(transcript, intent, params)

            processed_data.append({
                'transcript': transcript,
                'intent': intent,
                'params': params,
                'input_text': input_text,
                'output_text': output_text,
                'frequency': row['frequency']
            })

        processed_df = pd.DataFrame(processed_data)
        print(f"Preprocessed {len(processed_df)} samples")

        return processed_df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Processed dataframe

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_split = self.data_config['train_split']
        val_split = self.data_config['val_split']
        test_split = self.data_config['test_split']

        print(f"\nSplitting data: train={train_split}, val={val_split}, test={test_split}")

        # Check for intents with very few samples
        intent_counts = df['intent'].value_counts()
        rare_intents = intent_counts[intent_counts < 3].index.tolist()

        if rare_intents:
            print(f"Warning: Found {len(rare_intents)} intents with < 3 samples")
            print(f"Removing rare intents: {rare_intents[:5]}..." if len(rare_intents) > 5 else f"Removing rare intents: {rare_intents}")
            df = df[~df['intent'].isin(rare_intents)]
            print(f"Dataset size after filtering: {len(df)} samples")

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_split,
            random_state=self.random_seed,
            stratify=df['intent']  # Stratify by intent to maintain distribution
        )

        # Second split: separate train and validation
        val_ratio = val_split / (train_split + val_split)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_seed,
            stratify=train_val_df['intent']
        )

        print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val set:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

        return train_df, val_df, test_df

    def save_processed_data(self, train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame):
        """Save processed datasets to CSV files."""
        output_dir = self.data_config['processed_data_path']

        train_path = f"{output_dir}/train.csv"
        val_path = f"{output_dir}/val.csv"
        test_path = f"{output_dir}/test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nSaved processed data:")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        print(f"  Test:  {test_path}")

        # Also save intent statistics
        self._save_intent_stats(train_df, val_df, test_df, output_dir)

    def _save_intent_stats(self, train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          output_dir: str):
        """Save intent distribution statistics."""
        stats = {
            'train': train_df['intent'].value_counts().to_dict(),
            'val': val_df['intent'].value_counts().to_dict(),
            'test': test_df['intent'].value_counts().to_dict(),
            'total_intents': len(train_df['intent'].unique()),
            'intent_list': sorted(train_df['intent'].unique().tolist())
        }

        stats_path = f"{output_dir}/intent_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  Stats: {stats_path}")

    def process_and_save(self):
        """Complete pipeline: load, preprocess, split, and save data."""
        # Load raw data
        df = self.load_raw_data()

        # Preprocess
        processed_df = self.preprocess_dataset(df)

        # Split
        train_df, val_df, test_df = self.split_data(processed_df)

        # Save
        self.save_processed_data(train_df, val_df, test_df)

        print("\n" + "="*80)
        print("Data preprocessing complete!")
        print("="*80)

        # Show some examples
        print("\nSample training examples:")
        for i in range(min(3, len(train_df))):
            print(f"\n--- Example {i+1} ---")
            print(f"Input:  {train_df.iloc[i]['input_text']}")
            print(f"Output: {train_df.iloc[i]['output_text']}")

        return train_df, val_df, test_df


if __name__ == "__main__":
    loader = NLUDataLoader()
    loader.process_and_save()
