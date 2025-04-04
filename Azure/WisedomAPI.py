# Description: This file contains the code for the Azure Function that serves as the API for the fine-tuned LLM model.
# The function takes a text input, tokenizes it, performs inference using the pre-trained model, and returns the prediction.

import azure.functions as func
import json
from FineTunedLLM.LLM.Tokenizer import TransformersUtil  # Import the class
from FineTunedLLM.config import MODEL_DIR

# Load the model and tokenizer

def main(req: func.HttpRequest) -> func.HttpResponse:
    # Load the model and tokenizer
    model, tokenizer = TransformersUtil.load_model_and_tokenizer()  # Use the static method

    # Extract input from the request
    text = req.params.get('text')
    if not text:
        try:
            req_body = req.get_json()
            text = req_body.get('text')
        except ValueError:
            pass

    if not text:
        return func.HttpResponse("Invalid input. Please provide a text input.", status_code=400)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform inference using the pre-trained model
    if inputs['input_ids'].size(1) == 0:
        return func.HttpResponse("Invalid input. Please provide a non-empty text input.", status_code=400)

    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1).item()

    # Return the prediction
    return func.HttpResponse(
        func.jsonify({"prediction": predictions}),
        status_code=200
    )

