# FineTunedLLM Azure Deployment Guide

This guide provides instructions for deploying the enhanced FineTunedLLM serverless pipeline with **domain-specific context** support to Azure using Infrastructure as Code (IaC).

## Overview

The FineTunedLLM system now features:
- **Domain-Aware Summarization** - Claude Sonnet 4 with specialized domain prompts
- **Domain-Specific Fine-Tuning** - OpenAI GPT models optimized for specific domains
- **Automatic Domain Detection** - Intelligent domain classification from filenames
- **Multi-Domain Support** - Technical, Medical, Legal, and Financial domains
- **Serverless Architecture** - Auto-scaling Azure Functions for cost efficiency

## Prerequisites

1. **Azure CLI** - Install from [Azure CLI documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. **Azure Developer CLI (azd)** - Install from [azd documentation](https://docs.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
3. **Azure Subscription** - Active Azure subscription with appropriate permissions
4. **API Keys** - OpenAI API key and Anthropic API key

### Install Azure Developer CLI

**Windows (PowerShell):**
```powershell
# Using PowerShell
Invoke-RestMethod 'https://aka.ms/install-azd.ps1' | Invoke-Expression
```

**Windows (Winget):**
```bash
winget install microsoft.azd
```

**Verify Installation:**
```bash
azd version
```

## Architecture Overview

The deployment creates a comprehensive serverless pipeline:
- **Storage Account** - Blob storage for documents, summaries, and training data
- **Key Vault** - Secure storage for API keys and secrets
- **Function Apps** - Two serverless applications:
  - **Summarization Pipeline** - Domain-aware text processing with Claude Sonnet 4
  - **Fine-Tuning Pipeline** - Domain-specific model training with OpenAI
- **Application Insights** - Monitoring and logging with domain-specific metrics
- **Managed Identity** - Secure access between Azure services

### Domain Support

The system supports four predefined domains:
- **Technical** - Software, APIs, system architecture
- **Medical** - Healthcare, clinical, pharmaceutical content
- **Legal** - Contracts, compliance, regulatory content  
- **Financial** - Banking, investment, economic analysis

## Deployment Steps

### 1. Initialize Azure Developer CLI

```bash
# Clone and navigate to the project
cd c:\Development\RealmOfAgents\FineTunedLLM

# Initialize azd (if not already done)
azd init

# Login to Azure
azd auth login
```

### 2. Set Environment Variables

```bash
# Set your Azure location preference
azd env set AZURE_LOCATION "eastus"

# Set API keys (replace with your actual keys)
azd env set OPENAI_API_KEY "your-openai-api-key-here"
azd env set ANTHROPIC_API_KEY "your-anthropic-api-key-here"

# Optional: Set specific subscription
azd env set AZURE_SUBSCRIPTION_ID "your-subscription-id"
```

### 3. Preview Deployment

```bash
# Preview what will be deployed
azd provision --preview
```

### 4. Deploy Infrastructure and Applications

```bash
# Deploy everything (infrastructure + code)
azd up

# Or deploy in stages:
# azd provision  # Deploy infrastructure only
# azd deploy     # Deploy application code only
```

### 5. Verify Deployment

```bash
# Check deployment status
azd show

# View function logs
azd logs

# Get service endpoints
azd env get-values
```

## Post-Deployment Configuration

### Storage Containers

The deployment automatically creates these blob containers:
- `input-documents` - Upload documents for summarization
- `summaries` - Generated summaries storage
- `training-data` - Processed training data for fine-tuning
- `models` - Fine-tuned model artifacts

### Function Endpoints

After deployment, you'll have these endpoints:

#### Summarization Pipeline
- **Health Check**: `GET https://{summarization-function-url}/api/health`
- **Manual Processing**: `POST https://{summarization-function-url}/api/process-document`
- **Batch Processing**: `POST https://{summarization-function-url}/api/process-batch`

#### Fine-tuning Pipeline  
- **Health Check**: `GET https://{finetuning-function-url}/api/health`
- **Start Training**: `POST https://{finetuning-function-url}/api/start-training`
- **Training Status**: `GET https://{finetuning-function-url}/api/training-status/{job_id}`

## Usage Examples

### Domain-Aware Document Processing

#### Upload Documents with Domain Context

```bash
# Technical document
az storage blob upload \
  --account-name {storage-account-name} \
  --container-name input-documents \
  --name "technical_api_documentation.pdf" \
  --file "./api-docs.pdf"

# Medical document  
az storage blob upload \
  --account-name {storage-account-name} \
  --container-name input-documents \
  --name "medical_clinical_study.pdf" \
  --file "./clinical-study.pdf"
```

#### Manual Processing with Domain Specification

```bash
# Process with specific domain context
curl -X POST "https://{summarization-function-url}/api/process" \
  -H "Content-Type: application/json" \
  -d '{
    "blob_name": "document.pdf", 
    "domain_context": "technical",
    "focus_areas": ["API design", "performance optimization"]
  }'

# Batch processing with domain mapping
curl -X POST "https://{summarization-function-url}/api/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "files": ["tech_doc1.pdf", "medical_doc1.pdf"],
    "domain_mapping": {
      "tech_doc1.pdf": "technical",
      "medical_doc1.pdf": "medical"
    }
  }'
```

### Domain-Specific Fine-Tuning

#### Start Domain-Aware Fine-Tuning

```bash
# Fine-tune for technical domain
curl -X POST "https://{finetuning-function-url}/api/start-finetuning" \
  -H "Content-Type: application/json" \
  -d '{
    "training_file_blob": "technical_training_data.jsonl",
    "domain": "technical",
    "model": "gpt-3.5-turbo",
    "suffix": "tech-specialist"
  }'

# Batch fine-tuning for multiple domains
curl -X POST "https://{finetuning-function-url}/api/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "training_files": [
      "technical_training_data.jsonl",
      "medical_training_data.jsonl"
    ],
    "domain_mapping": {
      "technical_training_data.jsonl": "technical",
      "medical_training_data.jsonl": "medical"
    },
    "model": "gpt-3.5-turbo"
  }'
```

#### Monitor Domain-Specific Training

```bash
# Check training job status
curl "https://{finetuning-function-url}/api/check-status/{job_id}"

# Get domain training progress
curl "https://{finetuning-function-url}/api/domain-progress/technical"

# List available domains
curl "https://{finetuning-function-url}/api/domains"
```

### Health Checks

```bash
# Check summarization pipeline health
curl "https://{summarization-function-url}/api/health"

# Check fine-tuning pipeline health
curl "https://{finetuning-function-url}/api/health"
```

## Monitoring and Troubleshooting

### View Logs

```bash
# Function App logs
azd logs --service summarization-pipeline
azd logs --service finetuning-pipeline

# Or via Azure portal
# Navigate to Function App > Monitor > Logs
```

### Common Issues

1. **API Key Issues**: Ensure API keys are properly set using `azd env set`
2. **Permission Errors**: Verify managed identity has proper role assignments
3. **Storage Access**: Check that function apps can access storage containers
4. **Cold Start**: First function execution may take longer due to cold start

### Resource Management

```bash
# Update environment variables
azd env set KEY_NAME "new-value"
azd deploy  # Redeploy with new settings

# Clean up resources
azd down --purge
```

## Cost Optimization

- Functions use Consumption plan (pay-per-execution)
- Storage uses Standard LRS (locally redundant)
- Key Vault uses Standard tier
- Application Insights with 30-day retention

## Security Features

- Managed Identity for service-to-service authentication
- Key Vault for secret management
- HTTPS-only communication
- Role-based access control (RBAC)
- Network restrictions available for production deployments

## Next Steps

1. **Upload test documents** to the `input-documents` container
2. **Monitor processing** via Application Insights
3. **Review generated summaries** in the `summaries` container
4. **Configure fine-tuning** with your domain-specific data
5. **Set up CI/CD pipeline** for automated deployments

For detailed API documentation and advanced configuration, see the function app source code in `azure-functions/` directories.
