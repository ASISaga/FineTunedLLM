# 🛠️ Setup Guide

## 🎯 Overview

This guide walks you through setting up the FineTunedLLM system, which uses a hybrid cloud approach combining **Amazon Bedrock** (Claude Sonnet 4) for training data generation with **Azure OpenAI** (GPT-4) for model fine-tuning.

### ⏱️ Setup Time Estimate

| Setup Phase | Estimated Time | Difficulty |
|-------------|---------------|------------|
| 🐍 **Python Environment** | 5-10 minutes | 🟢 Easy |
| ☁️ **Cloud Services** | 15-30 minutes | 🟡 Moderate |
| 🔧 **Configuration** | 10-15 minutes | 🟢 Easy |
| 🧪 **Testing & Validation** | 10-15 minutes | 🟢 Easy |
| **Total Setup Time** | **40-70 minutes** | 🟡 **Moderate** |

### 🎯 What You'll Accomplish

By the end of this guide, you'll have:

- ✅ A fully configured development environment
- ✅ Active connections to both AWS Bedrock and Azure OpenAI
- ✅ Validated system functionality with test examples
- ✅ A working fine-tuning pipeline ready for your documents

### 🗺️ Documentation Journey

| Previous | **Current** | Next |
|----------|-------------|------|
| [📖 Documentation Hub](README.md) | **🛠️ Setup Guide** | [🚀 Deployment Guide](DEPLOYMENT.md) |

> **💡 First time here?** Start with the [Documentation Hub](README.md) for an overview of all available guides.

## 📋 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 5GB free space
- **Network**: Stable internet connection for cloud services

### Cloud Service Requirements

- **AWS Account** with Amazon Bedrock access
- **Azure Subscription** with Azure OpenAI Service
- **Active API Keys** for both services

## 🛠️ Installation Steps

### 1. Clone and Setup Repository

```powershell
# Clone the repository
git clone https://github.com/your-org/FineTunedLLM.git
cd FineTunedLLM

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install .
```

### 2. Python Dependencies

Install required packages:

```powershell
# Core dependencies
pip install openai>=1.0.0
pip install boto3>=1.26.0
pip install botocore>=1.29.0

# Azure dependencies (for Azure OpenAI)
pip install azure-openai>=1.0.0
pip install azure-identity>=1.12.0

# Data processing
pip install dataclasses-json>=0.5.0
pip install typing-extensions>=4.0.0

# Utilities
pip install python-dotenv>=1.0.0
pip install pydantic>=2.0.0

# Development tools (optional)
pip install pytest>=7.0.0
pip install black>=23.0.0
```

## ☁️ Cloud Service Setup

### 1. Amazon Bedrock Configuration

#### Step 1: Create AWS Account and Enable Bedrock

1. **Sign up** for AWS account at [aws.amazon.com](https://aws.amazon.com)
2. **Navigate** to Amazon Bedrock console
3. **Request model access** for Claude 3.5 Sonnet:
   - Go to "Model access" in Bedrock console
   - Click "Request model access"
   - Select "Anthropic Claude 3.5 Sonnet"
   - Submit request (approval usually takes a few minutes)

#### Step 2: Create IAM User with Bedrock Permissions

1. **Open IAM Console** in AWS
2. **Create new user** with programmatic access
3. **Attach policy** with following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            "Resource": "*"
        }
    ]
}
```

4. **Save credentials**:
   - Access Key ID
   - Secret Access Key
   - Preferred region (e.g., us-east-1)

#### Step 3: Test Bedrock Access

```powershell
# Test Bedrock connectivity
python -c "
import boto3
client = boto3.client('bedrock', region_name='us-east-1')
models = client.list_foundation_models()
print('Available models:', len(models['modelSummaries']))
"
```

### 2. Azure OpenAI Configuration

#### Step 1: Create Azure OpenAI Resource

1. **Sign in** to [Azure Portal](https://portal.azure.com)
2. **Create resource** → Search "OpenAI"
3. **Configure** Azure OpenAI:
   - **Subscription**: Select your subscription
   - **Resource Group**: Create new or use existing
   - **Region**: Choose available region (e.g., East US)
   - **Name**: Unique resource name
   - **Pricing Tier**: Standard S0

#### Step 2: Deploy GPT-4 Model

1. **Navigate** to Azure OpenAI Studio
2. **Go to Deployments** section
3. **Create new deployment**:
   - **Model**: GPT-4 Turbo (2024-04-09)
   - **Deployment name**: gpt-4-turbo
   - **Version**: Latest available
   - **Deployment type**: Standard

#### Step 3: Get API Credentials

1. **In Azure OpenAI resource** → Keys and Endpoint
2. **Copy the following**:
   - **API Key** (Key 1 or Key 2)
   - **Endpoint URL**
   - **API Version** (2024-02-01)

#### Step 4: Test Azure OpenAI Access

```powershell
# Test Azure OpenAI connectivity
python -c "
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key='your-api-key',
    api_version='2024-02-01',
    azure_endpoint='https://your-resource.openai.azure.com/'
)
models = client.models.list()
print('Available models:', [m.id for m in models.data])
"
```

## 🔧 Configuration Setup

### 1. Environment Variables

Create environment variables for secure credential storage:

```powershell
# AWS Bedrock Configuration
$env:AWS_ACCESS_KEY_ID = "your-aws-access-key"
$env:AWS_SECRET_ACCESS_KEY = "your-aws-secret-key"
$env:AWS_REGION = "us-east-1"

# Azure OpenAI Configuration
$env:AZURE_OPENAI_KEY = "your-azure-openai-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_VERSION = "2024-02-01"

# Alternative: Standard OpenAI (if not using Azure)
$env:OPENAI_API_KEY = "your-openai-api-key"
```

### 2. Configuration File Setup

1. **Copy configuration template**:

```powershell
Copy-Item KnowledgeModel\config_template.py KnowledgeModel\config_local.py
```

2. **Edit** `KnowledgeModel\config_local.py`:

```python
# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID = "your-aws-access-key"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-key"
AWS_REGION = "us-east-1"

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = "your-azure-openai-key"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-02-01"

# Model Configuration
DEFAULT_MODEL = "gpt-4-turbo-2024-04-09"
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Training Configuration
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1.0
```

### 3. Alternative: .env File Configuration

Create a `.env` file in the project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# Azure OpenAI Configuration
AZURE_OPENAI_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Training Configuration
DEFAULT_MODEL=gpt-4-turbo-2024-04-09
DEFAULT_EPOCHS=3
DEFAULT_BATCH_SIZE=4
```

## 🧪 Testing Setup

### 1. Run Basic Setup Test

```powershell
# Test system configuration
python KnowledgeModel\test_setup.py
```

Expected output:
```
✅ AWS Bedrock connection successful
✅ Azure OpenAI connection successful
✅ Domain context manager initialized
✅ All systems ready for training
```

### 2. Run Example Usage

```powershell
# Run basic example
python KnowledgeModel\example_usage.py
```

This will:
- Initialize the trainer with your credentials
- Process a sample document
- Generate training data via Bedrock
- Display the results

### 3. Test Individual Components

```powershell
# Test Bedrock summarization
python -c "
from KnowledgeModel.AbstractiveSummarizer import AbstractiveSummarizer
summarizer = AbstractiveSummarizer('aws-key', 'aws-secret', 'us-east-1')
result = summarizer.summarize_with_domain_context('Test document', 'technical')
print('Summary generated:', len(result) > 0)
"

# Test JSONL generation
python -c "
from KnowledgeModel.JsonlGenerator import JsonlGenerator
generator = JsonlGenerator('aws-key', 'aws-secret', 'us-east-1')
examples = generator.generate_training_examples('Test content', 'technical')
print('Examples generated:', len(examples))
"
```

## 🚀 Quick Start Example

Once setup is complete, try this basic workflow:

```python
from KnowledgeModel.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

# Initialize trainer
trainer = DomainAwareTrainer(
    api_key="your-openai-key",
    aws_access_key_id="your-aws-key",
    aws_secret_access_key="your-aws-secret",
    aws_region="us-east-1"
)

# Prepare sample documents
documents = [
    """
    API Authentication Best Practices
    
    When designing REST APIs, authentication is a critical security component.
    OAuth 2.0 with PKCE (Proof Key for Code Exchange) provides robust security
    for public clients. Always use HTTPS, implement rate limiting, and validate
    all input parameters to prevent injection attacks.
    """,
    """
    Database Optimization Strategies
    
    Query performance can be dramatically improved through proper indexing.
    Composite indexes should match your query patterns, and avoid over-indexing
    as it impacts write performance. Use query execution plans to identify
    bottlenecks and consider partitioning for large datasets.
    """
]

# Configure training parameters
config = FineTuningConfig(
    model="gpt-4-turbo-2024-04-09",
    domain_name="technical",
    n_epochs=3,
    batch_size=4,
    suffix="tech-docs-v1"
)

# Run complete pipeline
print("Starting training pipeline...")
job_id = trainer.run_complete_training_pipeline(
    text_documents=documents,
    domain_name="technical",
    config=config
)

print(f"Fine-tuning job started: {job_id}")

# Monitor progress
import time
while True:
    status = trainer.check_finetuning_status(job_id)
    print(f"Status: {status['status']}")
    
    if status['status'] in ['succeeded', 'failed']:
        break
    
    time.sleep(60)  # Check every minute

print("Training completed!")
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. AWS Bedrock Access Denied

**Error**: `AccessDeniedException: Could not access model`

**Solutions**:
- Verify model access is granted in Bedrock console
- Check IAM permissions for your user
- Ensure you're using the correct AWS region
- Wait a few minutes after requesting model access

#### 2. Azure OpenAI Authentication Failed

**Error**: `AuthenticationError: Invalid API key`

**Solutions**:
- Verify API key is correct in Azure portal
- Check endpoint URL format (must include https://)
- Ensure API version is supported (use 2024-02-01)
- Verify your subscription has quota available

#### 3. Model Deployment Issues

**Error**: `Model deployment not found`

**Solutions**:
- Ensure GPT-4 model is deployed in Azure OpenAI Studio
- Check deployment name matches configuration
- Verify model is in "Succeeded" state
- Try redeploying the model

#### 4. Rate Limit Exceeded

**Error**: `RateLimitError: Too many requests`

**Solutions**:
- Implement exponential backoff in your code
- Reduce batch size in configuration
- Check service quotas in both AWS and Azure
- Consider upgrading your service tier

#### 5. Large Document Processing

**Error**: Memory or timeout errors with large documents

**Solutions**:
- Split documents into smaller chunks
- Use streaming processing for memory efficiency
- Increase timeout settings
- Process documents in batches

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Your training code here
```

### Health Check Script

Create a comprehensive health check:

```python
# health_check.py
import os
from KnowledgeModel.DomainAwareTrainerBedrock import DomainAwareTrainer

def health_check():
    """Comprehensive system health check."""
    checks = []
    
    # Check environment variables
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    azure_key = os.getenv('AZURE_OPENAI_KEY')
    
    checks.append(("AWS credentials", bool(aws_key)))
    checks.append(("Azure credentials", bool(azure_key)))
    
    # Test connections
    try:
        trainer = DomainAwareTrainer(
            api_key=azure_key,
            aws_access_key_id=aws_key,
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1')
        )
        checks.append(("Trainer initialization", True))
    except Exception as e:
        checks.append(("Trainer initialization", False))
        print(f"Error: {e}")
    
    # Print results
    for check, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")

if __name__ == "__main__":
    health_check()
```

Run health check:

```powershell
python health_check.py
```

## 💰 Cost Considerations

### Amazon Bedrock Pricing

- **Claude 3.5 Sonnet**: ~$3 per 1M input tokens, ~$15 per 1M output tokens
- **Typical usage**: 100 documents generating 4 examples each ≈ $10-20

### Azure OpenAI Pricing

- **GPT-4 Turbo fine-tuning**: ~$8 per 1M training tokens
- **Typical usage**: 400 training examples ≈ $5-15

### Total Estimated Cost

- **Small domain (100 docs)**: $15-35
- **Medium domain (500 docs)**: $75-175
- **Large domain (1000+ docs)**: $150-350

### Cost Optimization Tips

1. **Batch processing**: Process multiple documents together
2. **Caching**: Store intermediate results to avoid reprocessing
3. **Validation split**: Use 80/20 split to reduce training data size
4. **Monitor usage**: Track API calls and costs regularly
5. **Start small**: Begin with a subset of documents and scale up

## 🎉 Setup Complete - Next Steps

Congratulations! Your FineTunedLLM system is now configured and ready for action. Here's your roadmap:

### 🏃‍♂️ Quick Start (Recommended)

1. **🧪 Test with sample documents** to verify functionality
2. **📊 Run the complete pipeline** with your data  
3. **📈 Monitor results** and iterate on your domain

### 🚀 Production Deployment

Ready for production? Follow our deployment guide:

| Deployment Option | Best For | Guide |
|------------------|----------|-------|
| **Azure Functions** | Serverless, auto-scaling | [🚀 Deployment Guide](DEPLOYMENT.md#azure-functions) |
| **Container Apps** | Microservices architecture | [🚀 Deployment Guide](DEPLOYMENT.md#containers) |
| **Local Development** | Testing and development | You're all set! |

### 📚 Deep Dive Resources

| Resource | When to Use | Link |
|----------|-------------|------|
| **🔧 Implementation Guide** | Understanding system architecture | [Implementation.md](Implementation.md) |
| **📖 API Reference** | Building custom integrations | [API.md](API.md) |
| **🏠 Documentation Hub** | Finding specific information | [README.md](README.md) |

### 💡 Pro Tips for Success

- **Start small**: Begin with 10-20 documents to test your domain
- **Monitor costs**: Track API usage with your first runs
- **Save configurations**: Keep your working config files for future projects
- **Join the community**: Share your results and get help from other users

## 🆘 Getting Help

| Issue Type | Resource | Link |
|------------|----------|------|
| **🐛 Setup Problems** | Troubleshooting section above | [Troubleshooting](#-troubleshooting) |
| **📖 Usage Questions** | API documentation | [API.md](API.md) |
| **🚀 Deployment Issues** | Deployment guide | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **💬 Community Support** | GitHub Issues | [Report Issue](../README.md#-support) |
| **📧 Direct Support** | Contact information | [README.md](../README.md#-support) |

---

**🎯 Ready to start fine-tuning?** Your environment is configured and tested. Time to process your first domain-specific documents!
