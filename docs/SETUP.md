# DomainAwareTrainer Setup Guide

## Overview

The DomainAwareTrainer now uses a hybrid approach:
- **Amazon Bedrock**: Claude Sonnet 4 for JSONL training data generation
- **Azure OpenAI**: GPT-4.1 for fine-tuning the final model

## Prerequisites

### 1. Python Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

### 2. AWS Setup for Bedrock

1. **Create AWS Account** and enable Amazon Bedrock
2. **Request Model Access** for Claude 3.5 Sonnet in Bedrock console
3. **Create IAM User** with Bedrock permissions:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "bedrock:InvokeModel",
                   "bedrock:ListFoundationModels"
               ],
               "Resource": "*"
           }
       ]
   }
   ```
4. **Note your credentials**:
   - AWS Access Key ID
   - AWS Secret Access Key
   - AWS Region (e.g., us-east-1)

### 3. Azure OpenAI Setup

1. **Create Azure OpenAI Resource** in Azure Portal
2. **Deploy GPT-4 Turbo** model in Azure AI Studio
3. **Note your credentials**:
   - API Key
   - Endpoint URL
   - API Version (2024-02-01)

## Configuration

### 1. Environment Variables

Set the following environment variables:

```powershell
# AWS Bedrock (for Claude Sonnet 4)
$env:AWS_ACCESS_KEY_ID = "your-aws-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-aws-secret-key"
$env:AWS_REGION = "us-east-1"

# Azure OpenAI (for GPT-4.1 fine-tuning)
$env:AZURE_OPENAI_KEY = "your-azure-openai-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"

# Alternative: Standard OpenAI
$env:OPENAI_API_KEY = "your-openai-api-key"
```

### 2. Configuration File

Copy `config_template.py` to `config_local.py` and update with your settings.

## Usage Examples

### Basic Usage

```python
from DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

# Initialize trainer
trainer = DomainAwareTrainer(
    api_key="your-openai-key",
    aws_access_key_id="your-aws-key",
    aws_secret_access_key="your-aws-secret",
    aws_region="us-east-1"
)

# Prepare your domain documents
documents = [
    "Your domain-specific text content here...",
    "More domain knowledge...",
    # ... add more documents
]

# Run complete pipeline
job_id = trainer.run_complete_training_pipeline(
    text_documents=documents,
    domain_name="your_domain",
    config=FineTuningConfig(
        model="gpt-4-turbo-2024-04-09",
        n_epochs=3,
        batch_size=4
    )
)

print(f"Fine-tuning job started: {job_id}")
```

### Bedrock-Only JSONL Generation

```python
# Generate training data without fine-tuning
training_file, validation_file = trainer.generate_training_data_with_bedrock(
    text_documents=documents,
    domain_name="your_domain",
    config=FineTuningConfig(domain_name="your_domain")
)
```

## Architecture Flow

1. **Input**: Raw domain-specific text documents
2. **Bedrock Processing**: Claude Sonnet 4 generates prompt-response pairs
3. **JSONL Creation**: Training data formatted for OpenAI fine-tuning
4. **Fine-tuning**: Azure OpenAI GPT-4.1 model customization
5. **Output**: Domain-specific fine-tuned model

## Cost Considerations

### Amazon Bedrock
- Claude 3.5 Sonnet pricing: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- For 100 documents generating 4 examples each: ~$10-20

### Azure OpenAI Fine-tuning
- GPT-4 Turbo fine-tuning: ~$8 per 1M training tokens
- For 400 training examples: ~$5-15

### Total estimated cost for small domain: $15-35

## Troubleshooting

### Common Issues

1. **Bedrock Access Denied**
   - Ensure model access is granted in Bedrock console
   - Check IAM permissions

2. **OpenAI Rate Limits**
   - Implement exponential backoff
   - Consider batch processing

3. **Large Document Processing**
   - Split documents into smaller chunks
   - Use streaming for memory efficiency

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Data Quality**: Ensure input documents are high-quality and domain-relevant
2. **Validation**: Always use validation data to monitor training progress
3. **Incremental Training**: Start with small datasets and scale up
4. **Cost Monitoring**: Track API usage to manage costs
5. **Model Evaluation**: Test fine-tuned models thoroughly before deployment

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify API credentials and permissions
3. Ensure all dependencies are installed correctly
4. Review the example usage scripts
