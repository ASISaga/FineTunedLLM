"""
FineTunedLLM Knowledge Model Package
Multi-domain adaptive fine-tuning system for specialized language models.

This package contains the core components for:
- Multi-Domain LLM Management
- Domain Knowledge Base System  
- Adaptive Summarization Engine
- Training Data Generation Pipeline
- Fine-Tuning Pipeline
- Feedback Integration System
"""

__version__ = "1.0.0"

# Core component imports for easy access
try:
    from .multi_domain_manager.MultiDomainLLMManager import MultiDomainLLMManager
    from .domain_knowledge_system.DomainKnowledgeBase import DomainKnowledgeBase
    from .adaptive_summarization.AbstractiveSummarizer import AbstractiveSummarizer
    from .feedback_integration.FeedbackCollector import FeedbackCollector
except ImportError:
    # Fallback for development/testing when relative imports might not work
    pass

__all__ = [
    "MultiDomainLLMManager",
    "DomainKnowledgeBase", 
    "AbstractiveSummarizer",
    "FeedbackCollector"
]
