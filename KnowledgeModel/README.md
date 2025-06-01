# FineTunedLLM KnowledgeModel - Refactored Structure

## Overview

The KnowledgeModel has been refactored into a modular structure based on the 6 core components identified in [`FineTunedLLM/docs/CORE_FUNCTIONALITY.md`](../docs/CORE_FUNCTIONALITY.md). This organization improves maintainability, scalability, and alignment with the system's architectural design.

## New Folder Structure

```
KnowledgeModel/
├── __init__.py                          # Main package initialization
├── multi_domain_manager/                # Component 1: Multi-Domain LLM Manager
│   ├── __init__.py
│   ├── MultiDomainLLMManager.py        # Central orchestrator for domain-specific LLMs
│   └── ModelManager.py                 # Model versioning and deployment tracking
├── domain_knowledge_system/             # Component 2: Domain Knowledge Base System
│   ├── __init__.py
│   ├── DomainKnowledgeBase.py          # Domain-specific knowledge storage and retrieval
│   ├── DomainContextManager.py         # Domain context management
│   └── EnhancedDomainContextManager.py # Enhanced domain context features
├── adaptive_summarization/              # Component 3: Adaptive Summarization Engine
│   ├── __init__.py
│   └── AbstractiveSummarizer.py        # Claude Sonnet 4 content generation
├── training_data_generation/            # Component 4: Training Data Generation Pipeline
│   ├── __init__.py
│   ├── Dataset.py                      # Dataset management and processing
│   └── Tokenizer.py                    # Text tokenization for training data
├── finetuning_pipeline/                 # Component 5: Fine-Tuning Pipeline
│   ├── __init__.py
│   ├── Trainer.py                      # Base training functionality
│   ├── DomainAwareTrainer.py           # Domain-specific training logic
│   └── DomainAwareTrainerBedrock.py    # Bedrock-integrated training
├── feedback_integration/                # Component 6: Feedback Integration System
│   ├── __init__.py
│   ├── FeedbackCollector.py            # Real-time feedback collection
│   └── ContinuousLearningPipeline.py   # Continuous improvement workflows
└── shared/                             # Shared utilities and configuration
    ├── __init__.py
    ├── config.py                       # Configuration management
    ├── config_template.py              # Configuration template
    ├── Model.py                        # Base model definitions
    ├── example_usage.py                # Usage examples
    └── test_setup.py                   # Testing utilities
```

## Component Mapping

### 1. Multi-Domain LLM Manager (`multi_domain_manager/`)
**Core Component 1** from CORE_FUNCTIONALITY.md
- **Purpose**: Central orchestrator for managing multiple domain-specific language models
- **Key Files**:
  - `MultiDomainLLMManager.py`: Main manager class handling lifecycle management
  - `ModelManager.py`: Model versioning and deployment tracking
- **Responsibilities**:
  - Domain registration and configuration management
  - Training pipeline orchestration across domains
  - Model versioning and deployment tracking
  - Performance monitoring and feedback integration
  - Adaptive learning based on real-world usage

### 2. Domain Knowledge Base System (`domain_knowledge_system/`)
**Core Component 2** from CORE_FUNCTIONALITY.md
- **Purpose**: Manages domain-specific text knowledge bases and contextual information
- **Key Files**:
  - `DomainKnowledgeBase.py`: Knowledge storage with semantic indexing
  - `DomainContextManager.py`: Domain context management
  - `EnhancedDomainContextManager.py`: Enhanced context features
- **Features**:
  - Text-based knowledge storage with semantic indexing
  - Domain-specific context paragraphs and knowledge sources
  - Similarity-based knowledge retrieval using TF-IDF vectorization
  - Adaptive content ranking based on performance feedback
  - Knowledge versioning and update tracking

### 3. Adaptive Summarization Engine (`adaptive_summarization/`)
**Core Component 3** from CORE_FUNCTIONALITY.md
- **Purpose**: Uses Claude Sonnet 4 via Amazon Bedrock for domain-aware content generation
- **Key Files**:
  - `AbstractiveSummarizer.py`: Claude Sonnet 4 integration for summarization
- **Capabilities**:
  - Domain-specific prompt engineering with contextual awareness
  - Multi-document synthesis for comprehensive knowledge representation
  - Structured output generation for consistent training data format
  - Focus area targeting based on domain requirements
  - Quality validation and content filtering

### 4. Training Data Generation Pipeline (`training_data_generation/`)
**Core Component 4** from CORE_FUNCTIONALITY.md
- **Purpose**: Creates OpenAI-compatible training data in JSONL format
- **Key Files**:
  - `Dataset.py`: Dataset management and processing
  - `Tokenizer.py`: Text tokenization for training data
- **Process Flow**:
  - Text chunking with overlap handling for comprehensive coverage
  - Domain-specific prompt-response pair generation using Claude Sonnet 4
  - Quality assurance through content validation and diversity checks
  - Format compliance ensuring adherence to OpenAI specifications
  - Iterative improvement based on model performance feedback

### 5. Fine-Tuning Pipeline (`finetuning_pipeline/`)
**Core Component 5** from CORE_FUNCTIONALITY.md
- **Purpose**: Manages the complete fine-tuning lifecycle with Azure OpenAI
- **Key Files**:
  - `Trainer.py`: Base training functionality
  - `DomainAwareTrainer.py`: Domain-specific training logic
  - `DomainAwareTrainerBedrock.py`: Bedrock-integrated training
- **Core Operations**:
  - Automated file upload and validation to Azure OpenAI
  - Training job orchestration with hyperparameter optimization
  - Progress monitoring and metrics collection during training
  - Model validation and performance testing
  - Deployment and endpoint creation for production use

### 6. Feedback Integration System (`feedback_integration/`)
**Core Component 6** from CORE_FUNCTIONALITY.md
- **Purpose**: Handles real-world deployment feedback for continuous improvement
- **Key Files**:
  - `FeedbackCollector.py`: Real-time feedback collection and analysis
  - `ContinuousLearningPipeline.py`: Continuous improvement workflows
- **Key Features**:
  - Real-time feedback collection from deployed models
  - Performance metrics analysis and trend identification
  - Weak area detection based on user ratings and corrections
  - Knowledge base updates incorporating user feedback
  - Adaptive training data generation targeting improvement areas

### Shared Components (`shared/`)
- **Purpose**: Common utilities, configuration, and base classes
- **Contents**:
  - Configuration management and templates
  - Base model definitions
  - Testing utilities and examples
  - Shared constants and utilities

## Usage

### Importing Components

```python
# Import main package
from KnowledgeModel import MultiDomainLLMManager, DomainKnowledgeBase, AbstractiveSummarizer, FeedbackCollector

# Import specific components
from KnowledgeModel.multi_domain_manager import MultiDomainLLMManager, ModelManager
from KnowledgeModel.domain_knowledge_system import DomainKnowledgeBase, DomainContextManager
from KnowledgeModel.adaptive_summarization import AbstractiveSummarizer
from KnowledgeModel.training_data_generation import Dataset, Tokenizer
from KnowledgeModel.finetuning_pipeline import DomainAwareTrainer, DomainAwareTrainerBedrock
from KnowledgeModel.feedback_integration import FeedbackCollector, ContinuousLearningPipeline
```

### Example Usage

```python
# Initialize the system
from KnowledgeModel import MultiDomainLLMManager
from KnowledgeModel.shared.config import load_config

config = load_config()
manager = MultiDomainLLMManager(config)

# Register a new domain
domain_config = {
    "domain_name": "technical",
    "context_paragraph": "Software development and technical documentation",
    "knowledge_sources": ["api_docs.txt", "technical_guides.txt"]
}
manager.register_domain(domain_config)

# Generate training data and fine-tune
manager.generate_training_data("technical")
manager.start_fine_tuning("technical")
```

## Benefits of New Structure

1. **Modular Architecture**: Each component is self-contained with clear responsibilities
2. **Improved Maintainability**: Related functionality is grouped together
3. **Better Documentation**: Each component has clear purpose and documentation
4. **Easier Testing**: Components can be tested independently
5. **Scalability**: New features can be added to specific components without affecting others
6. **Alignment with Documentation**: Structure directly matches the CORE_FUNCTIONALITY.md specification

## Migration Guide

If you have existing imports from the old structure, update them as follows:

```python
# Old imports
from KnowledgeModel.MultiDomainLLMManager import MultiDomainLLMManager
from KnowledgeModel.DomainKnowledgeBase import DomainKnowledgeBase

# New imports
from KnowledgeModel.multi_domain_manager import MultiDomainLLMManager
from KnowledgeModel.domain_knowledge_system import DomainKnowledgeBase
```

## Future Enhancements

The modular structure supports easy addition of new features:
- New domain types can be added to `domain_knowledge_system/`
- Additional training strategies can be added to `finetuning_pipeline/`
- New feedback mechanisms can be added to `feedback_integration/`
- Additional content generation engines can be added to `adaptive_summarization/`

This refactored structure provides a solid foundation for the continued development and scaling of the FineTunedLLM system.
