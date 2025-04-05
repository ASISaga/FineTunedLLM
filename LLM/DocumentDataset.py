import torch
from torch.utils.data import Dataset

class DocumentDataset(Dataset):
    """
    Dataset class to transform a single document (or a batch if needed)
    into tokenized inputs suitable for fine tuning.
    """
    def __init__(self, document_text: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Tokenize the document with truncation and padding to a fixed size.
        self.encodings = self.tokenizer(
            document_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Assume the entire document is one training example.
        self.input_ids = self.encodings["input_ids"]
        self.attention_mask = self.encodings["attention_mask"]

    def __len__(self):
        # We only have one example per document.
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }