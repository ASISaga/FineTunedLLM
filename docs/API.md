# API Reference

This document provides comprehensive API documentation for the FineTunedLLM system, covering both the core Python classes and the Azure Functions endpoints.

## 📚 Table of Contents

- [Core Python API](#core-python-api)
  - [DomainAwareTrainer](#domainawaretrainer)
  - [FineTuningConfig](#finetuningconfig)
  - [DomainContextManager](#domaincontextmanager)
  - [AbstractiveSummarizer](#abstractivesummarizer)
  - [JsonlGenerator](#jsonlgenerator)
- [Azure Functions API](#azure-functions-api)
  - [Summarization Pipeline](#summarization-pipeline)
  - [Fine-tuning Pipeline](#fine-tuning-pipeline)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)

## 🐍 Core Python API

### DomainAwareTrainer

The main orchestrator class for the fine-tuning pipeline.

#### Constructor

```python
DomainAwareTrainer(
    api_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str = "us-east-1",
    azure_endpoint: Optional[str] = None,
    azure_api_version: str = "2024-02-01"
)
```

**Parameters:**
- `api_key` (str): OpenAI or Azure OpenAI API key
- `aws_access_key_id` (str): AWS access key for Bedrock
- `aws_secret_access_key` (str): AWS secret key for Bedrock
- `aws_region` (str): AWS region for Bedrock (default: "us-east-1")
- `azure_endpoint` (Optional[str]): Azure OpenAI endpoint URL
- `azure_api_version` (str): Azure OpenAI API version

#### Methods

##### `run_complete_training_pipeline()`

Executes the complete training pipeline from documents to fine-tuned model.

```python
run_complete_training_pipeline(
    text_documents: List[str],
    domain_name: str,
    config: FineTuningConfig,
    validation_split: float = 0.2
) -> str
```

**Parameters:**
- `text_documents` (List[str]): List of raw text documents
- `domain_name` (str): Domain identifier ("technical", "medical", "legal", "financial")
- `config` (FineTuningConfig): Configuration object for fine-tuning
- `validation_split` (float): Fraction of data for validation (default: 0.2)

**Returns:**
- `str`: Fine-tuning job ID

**Example:**
```python
trainer = DomainAwareTrainer(
    api_key="your-key",
    aws_access_key_id="aws-key",
    aws_secret_access_key="aws-secret"
)

job_id = trainer.run_complete_training_pipeline(
    text_documents=["Document content here..."],
    domain_name="technical",
    config=FineTuningConfig(model="gpt-4-turbo-2024-04-09")
)
```

##### `generate_training_data_with_bedrock()`

Generates JSONL training data using Claude Sonnet 4 via Bedrock.

```python
generate_training_data_with_bedrock(
    text_documents: List[str],
    domain_name: str,
    config: FineTuningConfig,
    validation_split: float = 0.2
) -> Tuple[str, str]
```

**Returns:**
- `Tuple[str, str]`: Paths to training and validation JSONL files

##### `start_finetuning()`

Initiates the fine-tuning process with OpenAI.

```python
start_finetuning(
    training_file_path: str,
    validation_file_path: str,
    config: FineTuningConfig
) -> str
```

**Returns:**
- `str`: Fine-tuning job ID

##### `check_finetuning_status()`

Checks the status of a fine-tuning job.

```python
check_finetuning_status(job_id: str) -> dict
```

**Returns:**
- `dict`: Job status information

### FineTuningConfig

Configuration class for fine-tuning parameters.

#### Constructor

```python
FineTuningConfig(
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 3,
    batch_size: int = 4,
    learning_rate_multiplier: float = 1.0,
    domain_name: str = "general",
    examples_per_chunk: int = 4,
    suffix: Optional[str] = None
)
```

**Parameters:**
- `model` (str): Base model for fine-tuning (default: "gpt-3.5-turbo")
- `n_epochs` (int): Number of training epochs (default: 3)
- `batch_size` (int): Training batch size (default: 4)
- `learning_rate_multiplier` (float): Learning rate modifier (default: 1.0)
- `domain_name` (str): Domain identifier (default: "general")
- `examples_per_chunk` (int): Training examples per document chunk (default: 4)
- `suffix` (Optional[str]): Model name suffix

**Example:**
```python
config = FineTuningConfig(
    model="gpt-4-turbo-2024-04-09",
    n_epochs=5,
    batch_size=8,
    domain_name="medical",
    suffix="medical-specialist"
)
```

### DomainContextManager

Manages domain-specific prompts and context.

#### Methods

##### `get_domain_context()`

```python
get_domain_context(domain: str) -> Dict[str, Any]
```

**Parameters:**
- `domain` (str): Domain identifier

**Returns:**
- `Dict[str, Any]`: Domain-specific context and prompts

##### `detect_domain_from_filename()`

```python
detect_domain_from_filename(filename: str) -> str
```

**Parameters:**
- `filename` (str): File name to analyze

**Returns:**
- `str`: Detected domain identifier

### AbstractiveSummarizer

Handles text summarization using Claude Sonnet 4 via Amazon Bedrock.

#### Constructor

```python
AbstractiveSummarizer(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str = "us-east-1"
)
```

#### Methods

##### `summarize_with_domain_context()`

```python
summarize_with_domain_context(
    text: str,
    domain: str,
    focus_areas: Optional[List[str]] = None
) -> str
```

**Parameters:**
- `text` (str): Text to summarize
- `domain` (str): Domain context
- `focus_areas` (Optional[List[str]]): Specific areas to focus on

**Returns:**
- `str`: Domain-aware summary

### JsonlGenerator

Generates training data in JSONL format.

#### Methods

##### `generate_training_examples()`

```python
generate_training_examples(
    text: str,
    domain: str,
    num_examples: int = 4
) -> List[Dict[str, str]]
```

**Parameters:**
- `text` (str): Source text
- `domain` (str): Domain context
- `num_examples` (int): Number of examples to generate

**Returns:**
- `List[Dict[str, str]]`: List of prompt-response pairs

## ☁️ Azure Functions API

### Summarization Pipeline

Base URL: `https://{summarization-function-url}/api`

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-01T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "bedrock": "available",
    "storage": "connected"
  }
}
```

#### Process Document

```http
POST /process-document
```

**Request Body:**
```json
{
  "blob_name": "document.pdf",
  "domain_context": "technical",
  "focus_areas": ["API design", "performance"]
}
```

**Response:**
```json
{
  "status": "success",
  "job_id": "proc-123456",
  "summary_blob": "summaries/technical_document_summary.json",
  "processing_time": "45.2s"
}
```

#### Batch Process

```http
POST /process-batch
```

**Request Body:**
```json
{
  "files": ["doc1.pdf", "doc2.pdf"],
  "domain_mapping": {
    "doc1.pdf": "technical",
    "doc2.pdf": "medical"
  },
  "batch_size": 5
}
```

**Response:**
```json
{
  "status": "accepted",
  "batch_id": "batch-789012",
  "total_files": 2,
  "estimated_completion": "2025-06-01T13:00:00Z"
}
```

### Fine-tuning Pipeline

Base URL: `https://{finetuning-function-url}/api`

#### Start Training

```http
POST /start-training
```

**Request Body:**
```json
{
  "training_file_blob": "training_data.jsonl",
  "validation_file_blob": "validation_data.jsonl",
  "domain": "technical",
  "model": "gpt-4-turbo-2024-04-09",
  "config": {
    "n_epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 1.0
  }
}
```

**Response:**
```json
{
  "status": "started",
  "job_id": "ft-job-345678",
  "estimated_completion": "2025-06-01T15:00:00Z",
  "cost_estimate": "$12.50"
}
```

#### Training Status

```http
GET /training-status/{job_id}
```

**Response:**
```json
{
  "job_id": "ft-job-345678",
  "status": "running",
  "progress": 0.65,
  "current_epoch": 2,
  "total_epochs": 3,
  "metrics": {
    "training_loss": 0.234,
    "validation_loss": 0.267
  },
  "estimated_completion": "2025-06-01T15:00:00Z"
}
```

#### List Models

```http
GET /models
```

**Query Parameters:**
- `domain` (optional): Filter by domain
- `status` (optional): Filter by status ("training", "completed", "failed")

**Response:**
```json
{
  "models": [
    {
      "model_id": "ft:gpt-4-turbo:org:technical-specialist:abc123",
      "domain": "technical",
      "status": "completed",
      "created_at": "2025-06-01T12:00:00Z",
      "performance_metrics": {
        "final_training_loss": 0.198,
        "final_validation_loss": 0.223
      }
    }
  ]
}
```

## 📊 Data Models

### Training Example

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a technical documentation specialist..."
    },
    {
      "role": "user", 
      "content": "Explain the API authentication process"
    },
    {
      "role": "assistant",
      "content": "The API uses OAuth 2.0 with PKCE..."
    }
  ]
}
```

### Domain Context

```json
{
  "domain": "technical",
  "system_prompt": "You are a technical documentation specialist...",
  "focus_areas": ["APIs", "architecture", "performance"],
  "keywords": ["API", "endpoint", "authentication", "rate limit"],
  "examples": {
    "api_documentation": "...",
    "technical_spec": "..."
  }
}
```

## ⚠️ Error Handling

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `AUTH_ERROR` | Invalid API credentials | Check API keys and permissions |
| `QUOTA_EXCEEDED` | API rate limit exceeded | Implement exponential backoff |
| `INVALID_DOMAIN` | Unsupported domain | Use: technical, medical, legal, financial |
| `FILE_NOT_FOUND` | Blob storage file missing | Verify file path and permissions |
| `MODEL_NOT_AVAILABLE` | Requested model unavailable | Check model deployment status |

### Error Response Format

```json
{
  "error": {
    "code": "AUTH_ERROR",
    "message": "Invalid API key provided",
    "details": {
      "timestamp": "2025-06-01T12:00:00Z",
      "request_id": "req-123456"
    }
  }
}
```

## 🚦 Rate Limits

### Amazon Bedrock
- **Claude Sonnet 4**: 10 requests/minute
- **Token Limit**: 200K tokens/request

### Azure OpenAI
- **Fine-tuning**: 3 concurrent jobs
- **API Calls**: 240 requests/minute

### Recommended Practices

1. **Exponential Backoff**: Implement retry logic with exponential delays
2. **Batch Processing**: Group multiple documents into single requests
3. **Monitoring**: Track usage via Application Insights
4. **Caching**: Store intermediate results to avoid re-processing

## 📝 Examples

### Complete Workflow Example

```python
import asyncio
from KnowledgeModel.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

async def train_technical_model():
    # Initialize trainer
    trainer = DomainAwareTrainer(
        api_key="your-openai-key",
        aws_access_key_id="your-aws-key", 
        aws_secret_access_key="your-aws-secret"
    )
    
    # Load documents
    documents = [
        "Technical documentation content...",
        "API specification details...",
        "System architecture overview..."
    ]
    
    # Configure training
    config = FineTuningConfig(
        model="gpt-4-turbo-2024-04-09",
        n_epochs=3,
        domain_name="technical",
        suffix="tech-docs-v1"
    )
    
    # Run pipeline
    job_id = trainer.run_complete_training_pipeline(
        text_documents=documents,
        domain_name="technical", 
        config=config
    )
    
    # Monitor progress
    while True:
        status = trainer.check_finetuning_status(job_id)
        print(f"Status: {status['status']}")
        
        if status['status'] in ['succeeded', 'failed']:
            break
            
        await asyncio.sleep(60)  # Check every minute
    
    return status

# Run the workflow
result = asyncio.run(train_technical_model())
print(f"Final result: {result}")
```
