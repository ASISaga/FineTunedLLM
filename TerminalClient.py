from LLM.FineTuneLLM import DocumentTrainer

# Example usage of the DocumentTrainer class
if __name__ == "__main__":
    # List of documents to train on
    documents = [
        "First document text...",
        "Second document text...",
        "Third document text..."
    ]

    # Initialize the DocumentTrainer
    trainer = DocumentTrainer()

    # Iterate over documents and train incrementally
    for doc in documents:
        trainer.train(doc)

    # Generate the final combined document
    combined_document = trainer.generate_combined_document(documents)
    print("Combined Document:")
    print(combined_document)