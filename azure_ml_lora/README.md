# Azure ML LoRA Training Pipeline

This module contains all Azure ML and LoRA adapter functionality, refactored from the BusinessInfinity `ml_pipeline` directory.

## Overview

The azure_ml_lora module provides comprehensive Azure ML management and LoRA (Low-Rank Adaptation) training capabilities, including:

- **Unified ML Management** - Centralized Azure ML operations and endpoint management
- **LoRA Training** - Automated LoRA adapter training for fine-tuning large language models
- **Pipeline Automation** - End-to-end Azure ML pipeline orchestration
- **Model Registration** - Automated model artifact registration and deployment

## Components

### Core Classes

- **`UnifiedMLManager`** - Main interface for Azure ML operations, endpoint inference, and pipeline management
- **`MLManager`** - Azure ML workspace management, compute provisioning, and model lifecycle
- **`LoRATrainer`** - Local LoRA adapter training with configurable adapter definitions
- **`LoRAPipeline`** - Azure ML pipeline automation for distributed LoRA training

### Configuration

- **`endpoints.py`** - Azure ML endpoint configurations for different agents (CMO, CFO, CTO)
- **`configs/`** - YAML configuration files for Azure ML environments, compute clusters, and jobs

## Usage

### Basic Usage

```python
from azure_ml_lora import UnifiedMLManager, LoRATrainer, MLManager

# Initialize unified manager
ml_manager = UnifiedMLManager()

# Perform inference on an agent endpoint
result = await ml_manager.aml_infer("cmo", "Your marketing question")

# Train LoRA adapters locally
trainer = LoRATrainer(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    data_path="./data/training.jsonl",
    output_dir="./outputs"
)
trainer.train()

# Use Azure ML for training and deployment
azure_manager = MLManager(subscription_id, resource_group, workspace_name)
compute = azure_manager.get_or_create_compute()
env = azure_manager.get_or_create_environment()
```

### Running the Full Pipeline

```bash
cd azure_ml_lora
python run_pipeline.py
```

## Configuration Files

- **`environment.yml`** - Conda environment specification with required ML packages
- **`compute-cluster.yml`** - Azure ML compute cluster configuration
- **`job.yml`** - Azure ML job definition for training
- **`azure_commands.sh`** - Azure CLI commands for manual operations

## Environment Variables

Required environment variables:

- `AZURESUBSCRIPTION_ID` - Azure subscription ID
- `AZURERESOURCEGROUP` - Azure resource group name  
- `AZUREML_WORKSPACE` - Azure ML workspace name
- `PIPELINEENDPOINT_NAME` - Pipeline endpoint name
- `AML_CMO_SCORING_URI` / `AML_CMO_KEY` - CMO agent endpoint configuration
- `AML_CFO_SCORING_URI` / `AML_CFO_KEY` - CFO agent endpoint configuration
- `AML_CTO_SCORING_URI` / `AML_CTO_KEY` - CTO agent endpoint configuration

## Migration from BusinessInfinity ml_pipeline

This module represents a complete refactoring of the original `ml_pipeline` functionality from BusinessInfinity, with the following improvements:

- **Better Organization** - Clear separation of concerns between training, inference, and management
- **Enhanced Documentation** - Comprehensive docstrings and usage examples
- **Improved Imports** - Fixed relative import paths and dependencies
- **Configuration Management** - Centralized configuration files in `configs/` directory
- **Backwards Compatibility** - Maintained singleton pattern and key interfaces for existing integrations