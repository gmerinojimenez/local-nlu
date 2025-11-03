"""
T5-based model for Intent Classification and Parameter Extraction.
"""
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
from typing import Dict, List, Optional


class T5NLUModel(nn.Module):
    """T5 model for NLU with intent and parameter extraction."""

    def __init__(self, model_name: str = "t5-base", dropout: float = 0.1):
        """
        Initialize the T5 NLU model.

        Args:
            model_name: Name of the pretrained T5 model
            dropout: Dropout rate
        """
        super().__init__()
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Apply dropout to the model
        self.model.config.dropout_rate = dropout

        print(f"Initialized T5NLUModel with {model_name}")
        print(f"Model parameters: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (for training)

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }

    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_length: int = 256,
                 num_beams: int = 4,
                 early_stopping: bool = True) -> torch.Tensor:
        """
        Generate output sequences.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )

    def predict(self,
                text: str,
                max_length: int = 256,
                num_beams: int = 4) -> Dict:
        """
        Predict intent and parameters from input text.

        Args:
            text: Input utterance
            max_length: Maximum generation length
            num_beams: Number of beams for beam search

        Returns:
            Dictionary with intent and params
        """
        # Format input
        input_text = f"nlu: {text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams
            )

        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Parse JSON with fallback for malformed output
        try:
            result = json.loads(output_text)
            return result
        except json.JSONDecodeError as e:
            # Try to fix common issues with early training outputs
            fixed_output = self._try_fix_json(output_text)
            if fixed_output:
                return fixed_output

            # If still can't parse, return raw output
            return {
                "intent": "PARSE_ERROR",
                "params": {},
                "raw_output": output_text,
                "error": str(e)
            }

    def _try_fix_json(self, text: str) -> Optional[Dict]:
        """
        Try to fix common JSON formatting issues from early training.

        Args:
            text: Raw output text

        Returns:
            Parsed dict if successful, None otherwise
        """
        # Common issue: Missing outer braces
        if not text.strip().startswith('{'):
            text = '{' + text
        if not text.strip().endswith('}'):
            text = text + '}'

        # Try parsing again
        try:
            result = json.loads(text)
            return result
        except:
            pass

        # Try more aggressive fixes
        # Issue: Missing braces around nested params
        import re

        # Pattern: "params": "key": value
        # Fix to: "params": {"key": value}
        text = re.sub(r'"params":\s*"', r'"params": {"', text)

        # Add closing brace before end if needed
        if '"params": {' in text and not re.search(r'"params":\s*\{[^}]*\}', text):
            # Find last occurrence and add closing brace
            text = re.sub(r'(\{[^}]*?)$', r'\1}', text)

        try:
            result = json.loads(text)
            return result
        except:
            return None

    def predict_batch(self,
                     texts: List[str],
                     max_length: int = 256,
                     num_beams: int = 4) -> List[Dict]:
        """
        Predict intent and parameters for a batch of texts.

        Args:
            texts: List of input utterances
            max_length: Maximum generation length
            num_beams: Number of beams for beam search

        Returns:
            List of dictionaries with intent and params
        """
        # Format inputs
        input_texts = [f"nlu: {text}" for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            input_texts,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams
            )

        # Decode and parse
        results = []
        for output_id in output_ids:
            output_text = self.tokenizer.decode(output_id, skip_special_tokens=True)
            try:
                result = json.loads(output_text)
                results.append(result)
            except json.JSONDecodeError:
                results.append({
                    "intent": "UNSUPPORTED",
                    "params": {},
                    "raw_output": output_text
                })

        return results

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model and tokenizer from directory."""
        # Create instance properly
        instance = object.__new__(cls)

        # Call parent class init
        nn.Module.__init__(instance)

        # Load model and tokenizer
        instance.model = T5ForConditionalGeneration.from_pretrained(load_directory)
        instance.tokenizer = T5Tokenizer.from_pretrained(load_directory)
        instance.model_name = load_directory

        print(f"Model loaded from {load_directory}")
        return instance


if __name__ == "__main__":
    # Test the model
    print("Testing T5NLUModel...")

    model = T5NLUModel("t5-base")

    # Test prediction
    test_text = "set a timer for 5 minutes"
    print(f"\nTest input: {test_text}")

    result = model.predict(test_text)
    print(f"Result: {result}")
