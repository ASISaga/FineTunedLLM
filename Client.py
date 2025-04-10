import requests
import os
from KnowledgeModel.Trainer import Trainer

if __name__ == "__main__":
    # Example usage:

    # Initialize the Trainer
    trainer = Trainer()

def temp():
    trainer.load()

    # Define your domain-specific documents as a dictionary.
    # Each key represents a unique document ID, and the value is the document URL.
    document_urls = {
        "document1": "https://example.com/document1",
        # More document URLs can be added here.
    }

    MODEL_DIR = "fine_tuned_models"

    # Fetch the content of each document from the URLs
    documents = {}
    for doc_id, url in document_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for HTTP issues
            documents[doc_id] = response.text
        except requests.RequestException as e:
            print(f"Failed to fetch document {doc_id} from {url}: {e}")


    # Fine tune the model on the provided documents one at a time.
    for doc_id, text in documents.items():
        model_save_path = os.path.join(MODEL_DIR, doc_id, "model")
        trainer.fine_tune_document(document_text=text, doc_id=doc_id, model_save_path=model_save_path, num_epochs=1, batch_size=1)

    # Generate the final combined document
    combined_document = trainer.generate_combined_document(documents)
    print("Combined Document:")
    print(combined_document)