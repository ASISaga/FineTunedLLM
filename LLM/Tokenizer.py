"""
This module provides a utility class for loading the pre-trained model and tokenizer.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODEL_DIR

class Tokenizer(AutoTokenizer):  # Set base class as AutoTokenizer
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

    def preprocess_function(self, document):
        """
        Preprocess a document by tokenizing and truncating it to fit the model's input size.
        Args:
            document (str): The document text to preprocess.
        Returns:
            model_inputs (dict): A dictionary containing tokenized inputs.
        """
        model_inputs = self(document, max_length=1024, truncation=True, return_tensors="pt")
        return model_inputs
