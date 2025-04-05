import os
import torch
from LLM import DocumentDataset
from transformers import Trainer, TrainingArguments
from LLM import KnowledgeModel, KnowledgeTokenizer

# Import PEFT for LoRA (Parameter-Efficient Fine-Tuning)
from peft import LoraConfig, get_peft_model
peft_available = True


class ModelTrainer(Trainer):
    """
    Class to handle the fine tuning process for Model on domain-specific documents.
    Inherits from the Hugging Face `Trainer` class to leverage its training capabilities.
    
    The fine_tune_document method processes one document at a time,
    while fine_tune_documents iterates over a collection of documents.
    LoRA is applied to update only lightweight adapter weights.
    """
    def __init__(
        self,
        model_name: str,
        output_dir: str = "./fine_tuned_models",
        max_length: int = 2048,
        learning_rate: float = 1e-5,
        **kwargs
    ):
        # Initialize parent Trainer class
        super().__init__(**kwargs)

        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.learning_rate = learning_rate

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the tokenizer and KnowledgeModel
        print(f"Loading model: {model_name}")
        self.tokenizer = KnowledgeTokenizer.from_pretrained(model_name)
        self.model = KnowledgeModel.from_pretrained(model_name)

        # Apply LoRA unconditionally
        self._apply_lora_to_model()

    def _apply_lora_to_model(self):
        """
        Apply LoRA to the model by updating only selected modules.
        This reduces the number of parameters to be updated and can help
        mitigate overfitting or catastrophic forgetting.
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"  # Adjust based on model type if needed
        )
        self.model = get_peft_model(self.model, lora_config)
        print("LoRA has been applied to the model.")

    def fine_tune_document(self, document_text: str, doc_id: str, num_epochs: int = 1, batch_size: int = 1):
        """
        Fine tunes the model on a single document.
        
        Parameters:
          document_text (str): The text content of the document.
          doc_id (str): A unique identifier for the document.
          num_epochs (int): Number of epochs to train on this document.
          batch_size (int): Batch size per device during training.
        """
        # Prepare dataset from the document text.
        dataset = DocumentDataset(document_text, self.tokenizer, self.max_length)

        # Create an output directory for the fine tuning process
        output_path = os.path.join(self.output_dir, doc_id)
        os.makedirs(output_path, exist_ok=True)

        # Configure training arguments
        self.args.output_dir = output_path
        self.args.num_train_epochs = num_epochs
        self.args.per_device_train_batch_size = batch_size
        self.args.learning_rate = self.learning_rate

        print(f"Starting fine tuning on document: {doc_id}")
        self.train_dataset = dataset
        self.train()
        print(f"Fine tuning completed for document: {doc_id}")

        # Save the updated model and tokenizer after fine tuning on this document.
        model_save_path = os.path.join(output_path, "model")
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Model and tokenizer saved to: {model_save_path}")

    def fine_tune_documents(self, documents: dict, num_epochs: int = 1, batch_size: int = 1):
        """
        Fine tunes the model on multiple documents sequentially.
        
        Parameters:
          documents (dict): A dictionary mapping document IDs to document texts.
          num_epochs (int): Number of epochs for fine tuning each document.
          batch_size (int): Batch size per training step.
        """
        for doc_id, text in documents.items():
            self.fine_tune_document(document_text=text, doc_id=doc_id, num_epochs=num_epochs, batch_size=batch_size)

