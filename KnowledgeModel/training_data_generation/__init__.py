"""
Training Data Generation Pipeline Component

Core Component 4: Training Data Generation Pipeline
Creates OpenAI-compatible training data in JSONL format for fine-tuning.

Process flow:
- Text chunking with overlap handling for comprehensive coverage
- Domain-specific prompt-response pair generation using Claude Sonnet 4
- Quality assurance through content validation and diversity checks
- Format compliance ensuring adherence to OpenAI specifications
- Iterative improvement based on model performance feedback
"""

from .Dataset import Dataset
from .Tokenizer import Tokenizer

__all__ = ["Dataset", "Tokenizer"]
