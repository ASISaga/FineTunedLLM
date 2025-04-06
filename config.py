# Configuration file for FineTunedLLM

# Define the directory for the pre-trained model
MODEL_DIR = "path/to/your/model/directory"  # Replace with the actual path to your model directory

# Define the model name as a constant
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Model name for fine-tuning
FINE_TUNE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"

# Directory to store fine-tuned model outputs
OUTPUT_DIR = "./deepseek_r1_finetuned"

# Define the maximum token length for inputs
MAX_LENGTH = 2048

# Define the learning rate for training
LEARNING_RATE = 1e-5