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
    "output_dir": OUTPUT_DIR,
    "evaluation_strategy": "epoch",
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "weight_decay": 0.01,
    "save_total_limit": 3,
    "num_train_epochs": 3,
    "predict_with_generate": True,
}