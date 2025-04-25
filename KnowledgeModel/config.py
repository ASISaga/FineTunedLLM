# Configuration file for FineTunedLLM

# Directory to store fine-tuned model
MODEL_DIR = "model-directory"  # Replace with the actual path to your model directory

# Model name for fine-tuning
# MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Llama-8B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Define the maximum token length for inputs
MAX_LENGTH = 2048

# Define the learning rate for training
LEARNING_RATE = 1e-5

# Define LoraConfig as a constant
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Define Seq2SeqTrainingArguments configuration
SEQ2SEQ_TRAINING_ARGS = {
    "output_dir": MODEL_DIR,
    "evaluation_strategy": "epoch",
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "weight_decay": 0.01,
    "save_total_limit": 3,
    "num_train_epochs": 3,
    "predict_with_generate": True,
}