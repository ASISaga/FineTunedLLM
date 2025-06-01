# FineTunedLLM Documentation Hub

This directory contains comprehensive documentation for the FineTunedLLM system - a sophisticated domain-aware fine-tuning pipeline that combines Amazon Bedrock's Claude Sonnet 4 with Azure OpenAI's GPT-4.

## 📚 Documentation Structure

| Document | Purpose | Audience |
|----------|---------|----------|
| [🛠️ SETUP.md](SETUP.md) | Environment configuration and API setup | Developers, DevOps |
| [📖 Implementation.md](Implementation.md) | Technical architecture and code structure | Developers, Architects |
| [🚀 DEPLOYMENT.md](DEPLOYMENT.md) | Azure serverless deployment instructions | DevOps, Platform Engineers |
| [🔧 API.md](API.md) | Complete API documentation | Developers, Integrators |

## 🎯 Getting Started

### For New Users
1. **Start here**: [SETUP.md](SETUP.md) - Configure your environment and API credentials
2. **Understand the system**: [Implementation.md](Implementation.md) - Learn the architecture and design patterns
3. **Integration guide**: [API.md](API.md) - Explore the Python SDK and HTTP endpoints

### For Deployment Teams
1. **Prerequisites**: [SETUP.md](SETUP.md) - Ensure all requirements are met
2. **Deploy infrastructure**: [DEPLOYMENT.md](DEPLOYMENT.md) - Step-by-step Azure deployment
3. **Verify deployment**: [API.md](API.md) - Test endpoints and functionality

## 🏗️ System Overview

The FineTunedLLM system implements a hybrid cloud architecture that leverages the strengths of multiple AI services:

### 🔄 Hybrid Architecture
- **Amazon Bedrock**: Claude Sonnet 4 for intelligent training data generation
- **Azure OpenAI**: GPT-4 for domain-specific model fine-tuning  
- **Azure Functions**: Serverless orchestration and scaling
- **Azure Storage**: Document and model artifact management

### 🎯 Domain Specialization

The system supports four key domains with specialized processing:

| Domain | Focus Areas | Use Cases |
|--------|-------------|-----------|
| **Technical** | APIs, architecture, performance | Code docs, technical specs, system guides |
| **Medical** | Clinical, pharmaceutical, research | Medical protocols, research papers, clinical notes |
| **Legal** | Contracts, compliance, regulatory | Legal documents, compliance guides, contract analysis |
| **Financial** | Banking, investment, risk analysis | Financial reports, investment research, risk assessment |

## 🚀 Quick Start Examples

### Basic Python Usage

```python
from KnowledgeModel.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

# Initialize trainer with credentials
trainer = DomainAwareTrainer(
    api_key="your-openai-key",
    aws_access_key_id="your-aws-key",
    aws_secret_access_key="your-aws-secret",
    aws_region="us-east-1"
)

# Prepare domain documents
documents = [
    "Your technical documentation content...",
    "API reference materials...",
    "System architecture descriptions..."
]

# Configure fine-tuning parameters
config = FineTuningConfig(
    model="gpt-4-turbo-2024-04-09",
    n_epochs=3,
    batch_size=4,
    suffix="technical-v1"
)

# Execute complete pipeline
job_id = trainer.run_complete_training_pipeline(
    text_documents=documents,
    domain_name="technical",
    config=config
)

print(f"Fine-tuning job started: {job_id}")
```

### Azure Functions Deployment

```powershell
# Quick deployment to Azure
azd up

# Monitor deployment
azd logs --follow
```

## 🆘 Support & Resources

- **🐛 Issues**: Report bugs and feature requests via GitHub Issues
- **📚 Documentation**: Check the individual documentation files linked above
- **💡 Examples**: See [KnowledgeModel/example_usage.py](../KnowledgeModel/example_usage.py) for usage examples
- **🛠️ Troubleshooting**: Refer to [SETUP.md](SETUP.md#troubleshooting) for common issues

---

*For the most up-to-date information and examples, always refer to the individual documentation files linked above.*