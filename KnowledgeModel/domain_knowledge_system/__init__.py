"""
Domain Knowledge Base System Component

Core Component 2: Domain Knowledge Base System
Manages domain-specific text knowledge bases and contextual information for each domain.

Core features:
- Text-based knowledge storage with semantic indexing
- Domain-specific context paragraphs and knowledge sources
- Similarity-based knowledge retrieval using TF-IDF vectorization
- Adaptive content ranking based on performance feedback
- Knowledge versioning and update tracking
"""

from .DomainKnowledgeBase import DomainKnowledgeBase, DomainKnowledgeEntry, DomainMetrics
from .DomainContextManager import DomainContextManager, DomainContext
from .EnhancedDomainContextManager import EnhancedDomainContextManager

__all__ = [
    "DomainKnowledgeBase", 
    "DomainContextManager", 
    "EnhancedDomainContextManager",
    "DomainKnowledgeEntry",
    "DomainMetrics",
    "DomainContext"
]
