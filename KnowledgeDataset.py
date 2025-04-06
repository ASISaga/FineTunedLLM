from datasets import Dataset

import torch
import KnowledgeTokenizer


# Define custom dataset
class KnowledgeDataset(Dataset):
    def __init__(self, inputs=None, document_text=None, max_length=2048):
        # Load tokenizer and model
        self.tokenizer = KnowledgeTokenizer()

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

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

    def add_document(self, document_text):
        """
        Add a new document to the dataset.
        Args:
            document_text (str): The document text to add.
        """
        tokenized_document = self.tokenizer(
            document_text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        new_input_ids = tokenized_document["input_ids"]
        new_attention_mask = tokenized_document["attention_mask"]

        # Concatenate the new data with the existing dataset
        self.input_ids = torch.cat((self.input_ids, new_input_ids), dim=0)
        self.attention_mask = torch.cat((self.attention_mask, new_attention_mask), dim=0)
