from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Import the configuration file
from config import MODEL_DIR, MODEL_NAME

class Tokenizer(AutoTokenizer):
    """
    KnowledgeTokenizer is a custom tokenizer class that extends the functionality of AutoTokenizer.
    This class can be used to tokenize text data with additional features or customizations.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the KnowledgeTokenizer with the same parameters as AutoTokenizer.

        Args:
            *args: Positional arguments for the AutoTokenizer.
            **kwargs: Keyword arguments for the AutoTokenizer.
        """
        self.from_pretrained(MODEL_NAME)  # Load the tokenizer from the specified model directory
        super().__init__(*args, **kwargs)

    def custom_tokenize(self, text):
        """
        A custom method to tokenize text with additional processing.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens generated from the input text.
        """
        # Example of additional processing (can be customized as needed)
        processed_text = text.lower()  # Convert text to lowercase
        tokens = super().tokenize(processed_text)  # Use the base class's tokenize method
        return tokens

    @staticmethod
    def load_model_and_tokenizer():
        """
        Load the pre-trained model and tokenizer from the specified directory.

        Returns:
            tuple: A tuple containing the model and tokenizer objects.
        """
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = Tokenizer.from_pretrained(MODEL_DIR)
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