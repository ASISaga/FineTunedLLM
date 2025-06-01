"""
Multi-Domain LLM Manager Component

Core Component 1: Multi-Domain LLM Manager
The central orchestrator for managing multiple domain-specific language models.
Handles the complete lifecycle from training data generation to deployment and continuous improvement.

Key responsibilities:
- Domain registration and configuration management
- Training pipeline orchestration across domains
- Model versioning and deployment tracking
- Performance monitoring and feedback integration
- Adaptive learning based on real-world usage
"""

from .MultiDomainLLMManager import MultiDomainLLMManager, DomainLLMConfig, FeedbackEntry
from .ModelManager import ModelManager

__all__ = ["MultiDomainLLMManager", "ModelManager", "DomainLLMConfig", "FeedbackEntry"]
