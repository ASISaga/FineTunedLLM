"""
Adaptive Summarization Engine Component

Core Component 3: Adaptive Summarization Engine
Uses Claude Sonnet 4 via Amazon Bedrock to generate domain-aware summaries and training content.

Core capabilities:
- Domain-specific prompt engineering with contextual awareness
- Multi-document synthesis for comprehensive knowledge representation
- Structured output generation for consistent training data format
- Focus area targeting based on domain requirements
- Quality validation and content filtering
"""

from .AbstractiveSummarizer import AbstractiveSummarizer

__all__ = ["AbstractiveSummarizer"]
