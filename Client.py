from ModelTrainer import ModelTrainer
from FineTuneLLM import DocumentTrainer
from config import FINE_TUNE_MODEL_NAME, OUTPUT_DIR

if __name__ == "__main__":
    # Example usage:

    # Define your domain-specific documents as a dictionary.
    # Each key represents a unique document ID, and the value is the document text.
    documents = {
        "document1": "This is the text of a domain-specific document. It might include technical details, jargon, or subject-matter specifics that your model needs to learn.",
        "document2": "Another document with additional domain-specific information. Progressive fine tuning on individual documents helps the model gradually integrate all nuances.",
        # More documents can be added here.
    }

    # Specify the pre-trained model name (DeepSeek R1 model identifier)
    model_name = FINE_TUNE_MODEL_NAME  # Replace with your actual model name if needed

    # Initialize the Fine Tuner
    # Set use_lora=True if you want to apply LoRA (ensure you have the PEFT library installed)
    model_trainer = ModelTrainer(
        model_name=model_name,
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        use_lora=True
    )

    # Fine tune the model on the provided documents one at a time.
    model_trainer.fine_tune_documents(documents, num_epochs=1, batch_size=1)

    # Initialize the DocumentTrainer
    trainer = DocumentTrainer()

    # Iterate over documents and train incrementally
    for doc in documents:
        trainer.train(doc)

    # Generate the final combined document
    combined_document = trainer.generate_combined_document(documents)
    print("Combined Document:")
    print(combined_document)