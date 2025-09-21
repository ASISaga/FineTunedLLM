# LoRA Adapter Trainer Class
# This file contains the LoRATrainer class for automating LoRA adapter training and saving.

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


class LoRATrainer:
    """
    LoRATrainer automates the process of training and saving LoRA adapters for a given base model.
    Now supports configurable adapter definitions.
    """
    def __init__(self, model_name, data_path, output_dir, adapters=None):
        """
        Initialize the trainer with model name, dataset path, output directory, and adapters.
        Loads the base model and tokenizer.
        Args:
            model_name (str): HuggingFace model identifier
            data_path (str): Path to training data (JSONL or HuggingFace dataset)
            output_dir (str): Directory to save trained adapters
            adapters (list): List of adapter config dicts (see below)
        """
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        # Default adapters if not provided
        self.adapters = adapters if adapters is not None else [
            {
                "adapter_name": "lora_qv",
                "task_type": TaskType.CAUSAL_LM,
                "inference_mode": False,
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"]
            },
            {
                "adapter_name": "lora_ko",
                "task_type": TaskType.CAUSAL_LM,
                "inference_mode": False,
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["k_proj", "o_proj"]
            }
        ]

    def get_adapters(self):
        """
        Build LoRA adapter configurations from self.adapters.
        Returns:
            List[LoraConfig]: List of LoRA adapter configurations
        """
        return [
            LoraConfig(
                task_type=cfg["task_type"],
                inference_mode=cfg.get("inference_mode", False),
                r=cfg["r"],
                lora_alpha=cfg["lora_alpha"],
                target_modules=cfg["target_modules"],
                adapter_name=cfg["adapter_name"]
            )
            for cfg in self.adapters
        ]

    def attach_adapters(self):
        """
        Attach all defined LoRA adapters to the base model sequentially.
        """
        adapters = self.get_adapters()
        for cfg in adapters:
            self.base_model = get_peft_model(self.base_model, cfg)

    def prepare_dataset(self):
        """
        Load and preprocess the training dataset.
        Tokenizes prompts for input to the model.
        Returns:
            Dataset: Tokenized HuggingFace dataset
        """
        dataset = load_dataset(self.data_path, split="train")
        def tokenize(batch):
            # Tokenize each prompt in the batch
            return self.tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=512)
        dataset = dataset.map(tokenize, batched=True)
        return dataset

    def get_training_args(self):
        """
        Set up HuggingFace TrainingArguments for model training.
        Returns:
            TrainingArguments: Configured training arguments
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=1,
            fp16=True,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="no"
        )

    def train(self):
        """
        Main method to train the model with LoRA adapters and save each adapter separately.
        Steps:
            1. Attach adapters
            2. Prepare dataset
            3. Set training arguments
            4. Train model
            5. Save each adapter to output directory
        """
        self.attach_adapters()
        dataset = self.prepare_dataset()
        training_args = self.get_training_args()
        trainer = Trainer(
            model=self.base_model,
            args=training_args,
            train_dataset=dataset
        )
        trainer.train()
        # Save each adapter separately
        for adapter in [cfg["adapter_name"] for cfg in self.adapters]:
            self.base_model.save_pretrained(f"{self.output_dir}/{adapter}")