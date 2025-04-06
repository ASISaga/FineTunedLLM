from transformers import AutoTokenizer
# Import the configuration file
from config import MODEL_NAME

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
        # Initialize the model with the specified model name
        self.model_name = MODEL_NAME

        super().__init__(*args, **kwargs)

    def customTokenize(self, text):
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

    def preprocessFunction(self, document):
        """
        Preprocess a document by tokenizing and truncating it to fit the model's input size.
        Args:
            document (str): The document text to preprocess.
        Returns:
            model_inputs (dict): A dictionary containing tokenized inputs.
        """
        model_inputs = self(document, max_length=1024, truncation=True, return_tensors="pt")
        return model_inputs

    def load(self, tokenizer_path, local_files_only=False):
        """
        Load the tokenizer from the specified path.

        Args:
            tokenizer_path (str): Path to the tokenizer directory or name.
            local_files_only (bool): Whether to load only local files.

        Returns:
            Tokenizer: The loaded tokenizer instance.
        """
        try:
            tokenizer = self.from_pretrained(tokenizer_path, local_files_only=local_files_only)
            print(f"Tokenizer loaded successfully from {tokenizer_path}.")
            return tokenizer
        except Exception as e:
            print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise