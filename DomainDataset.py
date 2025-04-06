from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from config import MODEL_NAME
import KnowledgeModel

# Define custom dataset
class DomainDataset(Dataset):
    def __init__(self, inputs=None, document_text=None, max_length=2048):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = KnowledgeModel.from_pretrained(MODEL_NAME)

        if document_text:
            # Tokenize the document with truncation and padding to a fixed size.
            encodings = self.tokenizer(
                document_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
        elif inputs:
            self.input_ids = inputs['input_ids']
            self.attention_mask = inputs['attention_mask']
        else:
            raise ValueError("Either 'inputs' or 'document_text' must be provided.")

        # Prepare your initial domain-specific text.
        self.texts = ["Your initial domain-specific text goes here..."]

    def train(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        dataset = DomainDataset(inputs=inputs)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            num_train_epochs=1,  # Start with fewer epochs for initial fine-tuning
            logging_dir="./logs",
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        # Fine-tune the model
        trainer.train()

        # Save intermediate model
        self.model.save_pretrained("./intermediate_model")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
