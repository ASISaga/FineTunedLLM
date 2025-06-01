# FineTunedLLM

A sophisticated domain-aware fine-tuning system that combines **Claude Sonnet 4** via Amazon Bedrock for intelligent training data generation with **Azure OpenAI GPT-4** for specialized model fine-tuning.

## 🚀 Overview

FineTunedLLM is a comprehensive pipeline designed to create domain-specific language models through automated training data generation and fine-tuning. The system leverages the strengths of multiple cloud AI services to deliver high-quality, specialized models for technical, medical, legal, and financial domains.

## ✨ Key Features

- **🔄 Hybrid Cloud Architecture**: Amazon Bedrock (Claude Sonnet 4) + Azure OpenAI (GPT-4) integration
- **🎯 Domain-Aware Processing**: Specialized prompts and context for 4 key domains
- **⚡ Serverless Pipeline**: Azure Functions for automatic scaling and cost optimization
- **📊 Intelligent Data Generation**: Claude Sonnet 4 creates high-quality training pairs
- **🔧 Automated Fine-Tuning**: End-to-end model customization workflow
- **📈 Performance Monitoring**: Built-in metrics and evaluation tools

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Documents │───▶│  Amazon Bedrock  │───▶│ Training Data   │
│                 │    │  (Claude Sonnet) │    │    (JSONL)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Fine-Tuned Model│◀───│   Azure OpenAI   │◀───│   Fine-Tuning   │
│                 │    │    (GPT-4)       │    │    Process      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Supported Domains

| Domain | Description | Use Cases |
|--------|-------------|-----------|
| **Technical** | Software, APIs, system architecture | Code documentation, API guides, technical specs |
| **Medical** | Healthcare, clinical, pharmaceutical | Clinical notes, research papers, medical protocols |
| **Legal** | Contracts, compliance, regulatory | Legal documents, compliance guides, contracts |
| **Financial** | Banking, investment, economic | Financial reports, investment analysis, risk assessment |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Azure subscription with OpenAI service
- Active API keys for both services

### Installation

```powershell
# Clone the repository
git clone <repository-url>
cd FineTunedLLM

# Install dependencies
pip install -r requirements.txt

# Copy configuration template
Copy-Item KnowledgeModel\config_template.py KnowledgeModel\config_local.py
```

### Basic Usage

```python
from KnowledgeModel.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

# Initialize trainer
trainer = DomainAwareTrainer(
    api_key="your-openai-key",
    aws_access_key_id="your-aws-key",
    aws_secret_access_key="your-aws-secret",
    aws_region="us-east-1"
)

# Prepare domain documents
documents = [
    "Your domain-specific content here...",
    "Additional knowledge documents..."
]

# Run complete pipeline
job_id = trainer.run_complete_training_pipeline(
    text_documents=documents,
    domain_name="technical",
    config=FineTuningConfig(
        model="gpt-4-turbo-2024-04-09",
        n_epochs=3,
        batch_size=4
    )
)
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [🛠️ Setup Guide](docs/SETUP.md) | Environment configuration and API setup |
| [📖 Implementation Guide](docs/Implementation.md) | Technical architecture and code structure |
| [🚀 Deployment Guide](docs/DEPLOYMENT.md) | Azure serverless deployment instructions |
| [🔧 API Reference](docs/API.md) | Complete API documentation |

## 🛠️ Development

### Project Structure

```
FineTunedLLM/
├── KnowledgeModel/           # Core training pipeline
│   ├── DomainAwareTrainer.py # Main training orchestrator
│   ├── AbstractiveSummarizer.py # Claude Sonnet 4 integration
│   └── JsonlGenerator.py     # Training data generation
├── azure-functions/          # Serverless deployment
│   ├── summarization-pipeline/
│   └── finetuning-pipeline/
├── infra/                    # Infrastructure as Code
└── docs/                     # Documentation
```

### Running Tests

```powershell
# Run basic setup test
python KnowledgeModel\test_setup.py

# Run example usage
python KnowledgeModel\example_usage.py
```

## 💰 Cost Estimation

| Service | Usage | Estimated Cost |
|---------|-------|----------------|
| **Amazon Bedrock** | 100 docs → 400 training examples | $10-20 |
| **Azure OpenAI** | Fine-tuning 400 examples | $5-15 |
| **Total** | Small domain training | **$15-35** |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Check the [docs/](docs/) directory for detailed guides
- **Examples**: See [KnowledgeModel/example_usage.py](KnowledgeModel/example_usage.py) for usage examples
