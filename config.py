# Path to the pre-trained model directory (uploaded to Azure Storage or local app folder)
MODEL_DIR = "./final_model"
MODEL_NAME = "deepseek-ai/DeepSeek-R1" #DeepSeek R1 Distilled Model

# Define the model name as a constant
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Model name for fine-tuning
FINE_TUNE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"

# Directory to store fine-tuned model outputs
OUTPUT_DIR = "./deepseek_r1_finetuned"