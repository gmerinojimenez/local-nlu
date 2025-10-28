# Local NLU - Intent Classification and Parameter Extraction

A Natural Language Understanding system built with DeBERTa-v3-base for intent classification and parameter extraction.

## Project Structure

```
local-nlu/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed data (train/val/test splits)
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architecture definitions
│   ├── training/               # Training loops and utilities
│   └── inference/              # Inference pipeline
├── configs/                    # Configuration files
├── models/                     # Saved model checkpoints
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/                    # Utility scripts
├── logs/                       # Training logs and tensorboard
└── requirements.txt            # Python dependencies
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration
```bash
python scripts/explore_data.py
```

### 2. Training
```bash
python scripts/train.py --config configs/config.yaml
```

### 3. Inference
```bash
python scripts/inference.py --model models/best_model.pt --text "Your input text here"
```

## Model

- **Base Model**: microsoft/deberta-v3-base
- **Task**: Intent classification + Parameter extraction
- **Framework**: PyTorch + HuggingFace Transformers

## Dataset

Dataset: `all_origin_utterances_20240626_with_current_nli_response.xlsx`
- Location: `data/raw/`
- Format: Excel (.xlsx)

## License

MIT
