"""
Configuration template for DomainAwareTrainer with Amazon Bedrock and Azure OpenAI.
Copy this file to config_local.py and fill in your credentials.
"""

import os

# Azure OpenAI Configuration (for fine-tuning with GPT-4.1)
AZURE_OPENAI_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_KEY", ""),
    "api_base": os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/"),
    "api_version": "2024-02-01",
    "organization": os.getenv("OPENAI_ORGANIZATION", ""),
}

# Alternative: Standard OpenAI Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "organization": os.getenv("OPENAI_ORGANIZATION", ""),
}

# Amazon Bedrock Configuration (for Claude Sonnet 4)
AWS_BEDROCK_CONFIG = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    "region_name": os.getenv("AWS_REGION", "us-east-1"),
    "claude_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

# Default Fine-tuning Configuration
DEFAULT_FINETUNING_CONFIG = {
    "model": "gpt-4-turbo-2024-04-09",  # GPT-4.1 equivalent
    "n_epochs": 3,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "validation_split": 0.1
}

# Domain-specific configurations
DOMAIN_CONFIGS = {
    "machine_learning": {
        "description": "Expert in machine learning algorithms, neural networks, and AI applications",
        "focus_areas": ["deep learning", "neural networks", "supervised learning", "unsupervised learning"],
        "example_prompts": [
            "Explain the difference between supervised and unsupervised learning",
            "How do convolutional neural networks work?",
            "What are the key considerations for feature engineering?"
        ]
    },
    "healthcare_ai": {
        "description": "Specialist in healthcare applications of artificial intelligence and medical data analysis",
        "focus_areas": ["medical imaging", "diagnostic AI", "clinical decision support", "health informatics"],
        "example_prompts": [
            "How can AI improve medical diagnosis?",
            "What are the challenges in medical image analysis?",
            "Explain the role of AI in drug discovery"
        ]
    },
    "financial_technology": {
        "description": "Expert in financial technology, algorithmic trading, and fintech applications",
        "focus_areas": ["algorithmic trading", "risk management", "blockchain", "digital payments"],
        "example_prompts": [
            "How do trading algorithms work?",
            "What are the key risks in algorithmic trading?",
            "Explain the role of AI in fraud detection"
        ]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "domain_trainer.log"
}

# File paths
OUTPUT_PATHS = {
    "training_data_dir": "./training_data",
    "model_artifacts_dir": "./model_artifacts",
    "logs_dir": "./logs"
}
