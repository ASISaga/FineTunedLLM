# FineTunedLLM

A domain-aware fine-tuning system that uses **Claude Sonnet 4** via Amazon Bedrock for JSONL generation and **OpenAI GPT-4.1** for fine-tuning.

## Features

- **Hybrid Cloud Architecture**: Amazon Bedrock for Claude access + Azure OpenAI for fine-tuning
- **Domain-Aware Training**: Specialized prompts for Technical, Medical, Legal, and Financial domains
- **Automated Pipeline**: End-to-end training data generation and model fine-tuning
- **Serverless Deployment**: Azure Functions for scalable processing

## Quick Start

1. **Setup**: Follow the [Setup Guide](docs/SETUP.md) to configure AWS Bedrock and Azure OpenAI
2. **Implementation**: See [Implementation Guide](docs/Implementation.md) for technical details
3. **Deployment**: Use the [Deployment Guide](docs/DEPLOYMENT.md) for Azure deployment

## Documentation

- [📖 Implementation Guide](docs/Implementation.md) - Technical architecture and implementation details
- [🛠️ Setup Guide](docs/SETUP.md) - Step-by-step configuration instructions
- [🚀 Deployment Guide](docs/DEPLOYMENT.md) - Azure deployment instructions
- [📚 README](docs/README.md) - Comprehensive project overview

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE) file for details.
