from transformers import Trainer, TrainingArguments
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import KnowledgeDataset, KnowledgeModel, KnowledgeTokenizer

# Import PEFT for LoRA (Parameter-Efficient Fine-Tuning)
from peft import get_peft_model

from config import MODEL_NAME, OUTPUT_DIR, MAX_LENGTH, LEARNING_RATE, SEQ2SEQ_TRAINING_ARGS

# Iterative Fine-Tuning of LLM
# Gather domain-specific text relevant to your tasks (e.g., industry reports, technical documents).
# Clean and format the text. You can split your text into smaller chunks if needed.
# Fine tuning Model in a document-at-a-time or incremental fashion means that you update the model step by step—with each new document,
# you refine the model’s understanding of a specific domain. 
# This approach is especially useful when you want the model to gradually internalize domain-specific nuances 
# without overwhelming it with a massive dataset all at once.
#  
# Start with a Small Subset:** Select a small subset of your domain-specific text for the initial fine-tuning.
# Fine-Tune the Model:** Fine-tune the model on this subset and save the intermediate model.

class ModelTrainer(Seq2SeqTrainer):
    """
    Class to handle the fine tuning process for Model on domain-specific documents.
    Inherits from the Hugging Face `Seq2SeqTrainer` class to leverage its training capabilities for sequence-to-sequence tasks.
    
    The fine_tune_document method processes one document at a time,
    while fine_tune_documents iterates over a collection of documents.
    LoRA is applied to update only lightweight adapter weights.
    """
    def __init__(
        self,
        **kwargs
    ):
        # Initialize parent Seq2SeqTrainer class
        super().__init__(**kwargs)

        self.model_name = MODEL_NAME
        self.output_dir = OUTPUT_DIR
        self.max_length = MAX_LENGTH
        self.learning_rate = LEARNING_RATE

        # Load the tokenizer and KnowledgeModel
        print(f"Loading model: {self.model_name}")
        self.tokenizer = KnowledgeTokenizer()
        self.model = KnowledgeModel()

        # Initialize an empty dataset to store combined documents
        self.combined_dataset = None

        # Define training arguments for Seq2SeqTrainer
        self.seq2seq_training_args = Seq2SeqTrainingArguments(**SEQ2SEQ_TRAINING_ARGS)

    def fine_tune_document(self, document_text: str, doc_id: str, model_save_path: str, num_epochs: int = 1, batch_size: int = 1):
        """
        Fine tunes the model on a single document.
        
        Parameters:
          document_text (str): The text content of the document.
          doc_id (str): A unique identifier for the document.
          model_save_path (str): Path to save the fine-tuned model.
          num_epochs (int): Number of epochs to train on this document.
          batch_size (int): Batch size per device during training.
        """
        # Prepare dataset from the document text.
        dataset = KnowledgeDataset(document_text, self.tokenizer, self.max_length)

        # Configure training arguments
        self.args.output_dir = model_save_path
        self.args.num_train_epochs = num_epochs
        self.args.per_device_train_batch_size = batch_size
        self.args.learning_rate = self.learning_rate

        print(f"Starting fine tuning on document: {doc_id}")
        self.train_dataset = dataset
        self.train()
        print(f"Fine tuning completed for document: {doc_id}")

        # Save the updated model and tokenizer after fine tuning on this document.
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

    def train_combined_dataset(self):
        """
        Train the model on the combined dataset.
        """
        if self.combined_dataset is None:
            print("No documents added to the dataset for training.")
            return

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.seq2seq_training_args,
            train_dataset=self.combined_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()

    def generate_combined_document(self, documents):
        """
        Generate the final combined document from the list of documents.
        Args:
            documents (list): A list of document texts.
        Returns:
            combined_document (str): The combined document generated by the model.
        """
        inputs = self.tokenizer(
            "Combine knowledge from these documents: " + " ".join(documents),
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=1024,
            num_beams=5,
            early_stopping=True
        )

        combined_document = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return combined_document

    def progressive_fine_tuning(self, additional_texts):
        """
        Perform progressive fine-tuning by gradually including more domain-specific text.
        Args:
            additional_texts (list): List of additional domain-specific texts for fine-tuning.
        """
        # Load intermediate model and tokenizer
        intermediate_model_path = "./intermediate_model"
        tokenizer = KnowledgeTokenizer.from_pretrained(intermediate_model_path)
        model = KnowledgeModel.from_pretrained(intermediate_model_path)

        # Prepare additional domain-specific text
        inputs = tokenizer(
            additional_texts, return_tensors="pt", padding=True, truncation=True
        )

        # Define custom dataset
        dataset = KnowledgeDataset(inputs)

        # Update training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            num_train_epochs=2,  # Increase the number of epochs for further fine-tuning
            logging_dir="./logs",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        # Continue fine-tuning
        trainer.train()

        # Save progressively fine-tuned model
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        print("Progressive fine-tuning completed. Model and tokenizer saved.")

