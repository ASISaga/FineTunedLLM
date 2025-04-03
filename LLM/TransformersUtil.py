"""
This module provides a utility class for loading the pre-trained model and tokenizer.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_DIR

class TransformersUtil:
    """
    A utility class for loading the pre-trained model and tokenizer.
    """

    @staticmethod
    def load_model_and_tokenizer():
        """
        Load the pre-trained model and tokenizer from the specified directory.

        Returns:
            tuple: A tuple containing the model and tokenizer objects.
        """
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        return model, tokenizer
