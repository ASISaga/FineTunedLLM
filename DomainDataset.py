from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from config import MODEL_NAME

# Define custom dataset
class DomainDataset(Dataset):
    def __init__(self, inputs):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

        # Prepare your initial domain-specific text.
        self.texts = ["Your initial domain-specific text goes here..."]


    # Define training function. It takes a list of texts as input and fine-tunes the model.
    def train(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        dataset = DomainDataset(inputs)

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
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}
