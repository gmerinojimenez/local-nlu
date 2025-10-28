# NLU Project Guide - Intent Classification & Parameter Extraction

## Overview

This project implements a Natural Language Understanding (NLU) system using **T5-base** for joint intent classification and parameter extraction. The model takes natural language utterances and generates structured JSON output with intent and parameters.

## Dataset

- **Total samples**: 35,516 (after filtering rare intents)
- **Intents**: 72 unique intents
- **Format**: Text-to-text generation
  - Input: `"nlu: set a timer for 5 minutes"`
  - Output: `{"intent": "SET_TIMER", "params": {"timer_duration": {"amount": 5, "unit": "min"}}}`

## Project Structure

```
local-nlu/
├── data/
│   ├── raw/                              # Original Excel dataset
│   └── processed/                        # Train/val/test splits (CSV)
│       ├── train.csv (24,860 samples)
│       ├── val.csv   (5,328 samples)
│       ├── test.csv  (5,328 samples)
│       └── intent_stats.json
├── src/
│   ├── data/
│   │   ├── data_loader.py               # Data loading & preprocessing
│   │   └── dataset.py                   # PyTorch Dataset class
│   ├── models/
│   │   └── t5_nlu.py                    # T5 model wrapper
│   ├── training/
│   │   ├── trainer.py                   # Training loop
│   │   └── evaluator.py                 # Evaluation metrics
│   └── inference/
├── scripts/
│   ├── explore_data.py                  # Dataset exploration
│   ├── analyze_parameters.py            # Parameter analysis
│   ├── preprocess_data.py              # Data preprocessing
│   ├── train.py                         # Training script
│   ├── evaluate.py                      # Evaluation script
│   └── inference.py                     # Inference/testing script
├── configs/
│   └── config.yaml                      # Configuration file
├── models/                              # Saved model checkpoints
├── logs/                                # Training logs
└── requirements.txt                     # Dependencies
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Exploration (Already Done)

```bash
# Explore the dataset structure
python scripts/explore_data.py

# Analyze parameter schemas
python scripts/analyze_parameters.py
```

### 3. Data Preprocessing (Already Done)

```bash
# Preprocess and split data
python scripts/preprocess_data.py
```

This creates:
- `data/processed/train.csv` - Training set (70%)
- `data/processed/val.csv` - Validation set (15%)
- `data/processed/test.csv` - Test set (15%)

## Training

### Configuration

Edit `configs/config.yaml` to adjust hyperparameters:

```yaml
model:
  name: "t5-base"
  max_input_length: 128
  max_output_length: 256
  dropout: 0.1

training:
  batch_size: 16           # Adjust based on GPU memory
  learning_rate: 2.0e-5
  num_epochs: 5
  warmup_steps: 500
  weight_decay: 0.01
  max_grad_norm: 1.0
```

### Start Training

```bash
# Train the model
python scripts/train.py
```

**Training Details:**
- Model: T5-base (220M parameters)
- Training time: ~2-4 hours per epoch (depending on GPU)
- Checkpoints saved every epoch
- Best model saved based on validation loss

**Expected Output:**
- Checkpoints in `models/checkpoint_epoch_N/`
- Best model in `models/best_model/`
- Training history in `models/training_history.json`

### Monitor Training

Training progress is displayed with:
- Training loss per batch
- Validation loss per epoch
- Learning rate

## Evaluation

After training, evaluate the model:

```bash
python scripts/evaluate.py
```

**Metrics Reported:**
- **Exact Match**: Complete JSON match (intent + all params)
- **Intent Accuracy**: Intent prediction accuracy
- **Parse Success Rate**: Valid JSON generation rate
- **Parameter F1/Precision/Recall**: Parameter-level accuracy
- **Per-Intent Accuracy**: Accuracy breakdown by intent

Results are saved to:
- `models/val_metrics.json`
- `models/test_metrics.json`

## Inference

### Interactive Testing

```bash
python scripts/inference.py
```

This provides:
1. Pre-defined test examples
2. Interactive mode for custom utterances

### Programmatic Usage

```python
from src.models.t5_nlu import T5NLUModel

# Load trained model
model = T5NLUModel.from_pretrained('models/best_model')

# Single prediction
result = model.predict("set a timer for 5 minutes")
print(result)
# Output: {"intent": "SET_TIMER", "params": {"timer_duration": {"amount": 5, "unit": "min"}}}

# Batch prediction
results = model.predict_batch([
    "what is the weather",
    "open YouTube",
    "volume up"
])
```

## Model Architecture

**T5-base (Text-to-Text Transfer Transformer)**
- Encoder-decoder architecture
- 12 layers (encoder) + 12 layers (decoder)
- 220M parameters
- Pre-trained on C4 dataset
- Fine-tuned for NLU task

**Advantages:**
- Handles complex nested parameters
- Generates structured JSON output
- Flexible for various parameter types
- Strong zero-shot generalization

## Training Tips

### If GPU Memory is Limited:

1. Reduce batch size in `configs/config.yaml`:
   ```yaml
   batch_size: 8  # or 4
   ```

2. Enable gradient accumulation:
   ```yaml
   gradient_accumulation_steps: 2
   ```

3. Use mixed precision training (add to trainer.py if needed)

### If Training is Slow:

1. Reduce `num_workers` in data loaders (scripts/train.py):
   ```python
   num_workers=0  # Disable multiprocessing
   ```

2. Use a smaller model:
   ```yaml
   model:
     name: "t5-small"  # 60M parameters
   ```

### Improving Performance:

1. **More epochs**: Increase `num_epochs` to 10-15
2. **Learning rate tuning**: Try 1e-5 or 3e-5
3. **Data augmentation**: Add paraphrases of utterances
4. **Regularization**: Adjust dropout or weight_decay

## Next Steps After Training

1. **Evaluate Performance**
   ```bash
   python scripts/evaluate.py
   ```

2. **Test on Custom Inputs**
   ```bash
   python scripts/inference.py
   ```

3. **Deploy Model**
   - Create a REST API (Flask/FastAPI)
   - Package for production use
   - Add input validation and error handling

4. **Iterate and Improve**
   - Analyze errors from evaluation
   - Collect more training data for poorly performing intents
   - Fine-tune with domain-specific data

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size`
- Reduce `max_input_length` and `max_output_length`
- Use gradient checkpointing

### Poor Parse Success Rate
- Model may need more training epochs
- Check output examples during evaluation
- Consider adding more diverse training data

### Low Intent Accuracy
- Some intents may be too similar
- Need more samples for rare intents
- Consider intent grouping/hierarchical classification

## Files Generated During Training

```
models/
├── best_model/                    # Best model checkpoint
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   └── training_state.pt
├── checkpoint_epoch_N/            # Periodic checkpoints
├── training_history.json          # Loss curves
├── val_metrics.json              # Validation metrics
└── test_metrics.json             # Test metrics
```

## Citation

If you use this project, please cite:

```bibtex
@misc{t5-nlu-2025,
  title={T5-based NLU System for Intent Classification and Parameter Extraction},
  year={2025},
  note={Built with HuggingFace Transformers and PyTorch}
}
```

## Support

For issues or questions:
1. Check configuration in `configs/config.yaml`
2. Review logs in training output
3. Test with smaller dataset first
4. Ensure all dependencies are installed correctly

Good luck with your training! 🚀
