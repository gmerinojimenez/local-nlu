"""
PyTorch Dataset class for T5-based NLU training.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import T5Tokenizer
from typing import Dict, List


class NLUDataset(Dataset):
    """PyTorch Dataset for NLU data with T5 tokenization."""

    def __init__(self,
                 data_path: str,
                 tokenizer: T5Tokenizer,
                 max_input_length: int = 128,
                 max_output_length: int = 256):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the CSV file
            tokenizer: T5 tokenizer instance
            max_input_length: Maximum length for input sequences
            max_output_length: Maximum length for output sequences
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with tokenized inputs and labels
        """
        row = self.data.iloc[idx]

        input_text = row['input_text']
        output_text = row['output_text']

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize output (labels)
        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare labels (replace padding token id with -100 so it's ignored by loss)
        labels = output_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


def create_data_loaders(train_path: str,
                       val_path: str,
                       test_path: str,
                       tokenizer: T5Tokenizer,
                       batch_size: int = 16,
                       max_input_length: int = 128,
                       max_output_length: int = 256,
                       num_workers: int = 4):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        tokenizer: T5 tokenizer instance
        batch_size: Batch size for training
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = NLUDataset(
        train_path,
        tokenizer,
        max_input_length,
        max_output_length
    )

    val_dataset = NLUDataset(
        val_path,
        tokenizer,
        max_input_length,
        max_output_length
    )

    test_dataset = NLUDataset(
        test_path,
        tokenizer,
        max_input_length,
        max_output_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Created data loaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    print("Testing dataset loading...")
    dataset = NLUDataset(
        'data/processed/train.csv',
        tokenizer,
        max_input_length=128,
        max_output_length=256
    )

    print(f"Dataset size: {len(dataset)}")

    # Get first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"Input shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")

    # Decode to check
    input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    output_text = tokenizer.decode(
        sample['labels'][sample['labels'] != -100],
        skip_special_tokens=True
    )

    print(f"\nDecoded input: {input_text}")
    print(f"Decoded output: {output_text}")
