"""
Domain Knowledge Base Management System
Manages domain-specific text knowledge bases and context for adaptive fine-tuning.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class DomainKnowledgeEntry:
    """Individual knowledge entry in a domain knowledge base"""
    id: str
    content: str
    source: str
    domain: str
    keywords: List[str]
    concepts: List[str]
    importance_score: float
    created_at: str
    updated_at: str
    version: int
    metadata: Dict[str, Any]

@dataclass
class DomainMetrics:
    """Performance metrics for a domain"""
    domain_name: str
    total_documents: int
    total_tokens: int
    quality_score: float
    coverage_score: float
    last_updated: str
    active_models: List[str]
    feedback_count: int
    improvement_rate: float

class DomainKnowledgeBase:
    """
    Manages domain-specific knowledge bases with adaptive learning capabilities.
    
    Features:
    - Text-based knowledge storage and retrieval
    - Semantic similarity search
    - Adaptive content ranking
    - Knowledge base versioning
    - Performance tracking
    """
    
    def __init__(self, base_path: str = "./knowledge_bases"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Domain storage
        self.knowledge_entries: Dict[str, List[DomainKnowledgeEntry]] = {}
        self.domain_metrics: Dict[str, DomainMetrics] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.content_vectors: Dict[str, np.ndarray] = {}
        
        # Load existing knowledge bases
        self._load_knowledge_bases()
    
    def add_domain_knowledge(self, domain: str, content: str, source: str, 
                           keywords: List[str] = None, concepts: List[str] = None,
                           importance_score: float = 1.0, metadata: Dict[str, Any] = None) -> str:
        """
        Add new knowledge entry to a domain knowledge base.
        
        Args:
            domain: Domain name
            content: Text content
            source: Source of the content
            keywords: Associated keywords
            concepts: Domain concepts covered
            importance_score: Importance weight (0.0-1.0)
            metadata: Additional metadata
            
        Returns:
            Entry ID
        """
        # Generate unique ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        entry_id = f"{domain}_{content_hash}"
        
        # Create knowledge entry
        entry = DomainKnowledgeEntry(
            id=entry_id,
            content=content,
            source=source,
            domain=domain,
            keywords=keywords or [],
            concepts=concepts or [],
            importance_score=importance_score,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            version=1,
            metadata=metadata or {}
        )
        
        # Add to domain knowledge
        if domain not in self.knowledge_entries:
            self.knowledge_entries[domain] = []
        
        self.knowledge_entries[domain].append(entry)
        
        # Update domain metrics
        self._update_domain_metrics(domain)
        
        # Rebuild vectorizer for domain
        self._rebuild_domain_vectors(domain)
        
        # Save to persistent storage
        self._save_domain_knowledge(domain)
        
        logger.info(f"Added knowledge entry {entry_id} to domain {domain}")
        return entry_id
    
    def get_domain_context(self, domain: str, query: str = None, limit: int = 10) -> Dict[str, Any]:
        """
        Get domain-specific context, optionally filtered by query similarity.
        
        Args:
            domain: Domain name
            query: Optional query for similarity search
            limit: Maximum number of entries to return
            
        Returns:
            Domain context with relevant knowledge entries
        """
        if domain not in self.knowledge_entries:
            return {"domain": domain, "entries": [], "total_entries": 0}
        
        entries = self.knowledge_entries[domain]
        
        if query and domain in self.vectorizers:
            # Perform similarity search
            relevant_entries = self._similarity_search(domain, query, limit)
        else:
            # Return top entries by importance score
            relevant_entries = sorted(
                entries, 
                key=lambda x: x.importance_score, 
                reverse=True
            )[:limit]
        
        # Compile comprehensive context
        context = {
            "domain": domain,
            "total_entries": len(entries),
            "entries": [asdict(entry) for entry in relevant_entries],
            "domain_keywords": self._extract_domain_keywords(domain),
            "domain_concepts": self._extract_domain_concepts(domain),
            "metrics": asdict(self.domain_metrics.get(domain)) if domain in self.domain_metrics else None,
            "last_updated": max([entry.updated_at for entry in entries]) if entries else None
        }
        
        return context
    
    def update_knowledge_entry(self, entry_id: str, content: str = None, 
                             importance_score: float = None, 
                             keywords: List[str] = None,
                             concepts: List[str] = None,
                             metadata: Dict[str, Any] = None) -> bool:
        """
        Update an existing knowledge entry.
        
        Args:
            entry_id: Entry ID to update
            content: New content
            importance_score: New importance score
            keywords: New keywords
            concepts: New concepts
            metadata: New metadata
            
        Returns:
            Success status
        """
        # Find entry across all domains
        for domain, entries in self.knowledge_entries.items():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    # Update fields
                    if content is not None:
                        entry.content = content
                    if importance_score is not None:
                        entry.importance_score = importance_score
                    if keywords is not None:
                        entry.keywords = keywords
                    if concepts is not None:
                        entry.concepts = concepts
                    if metadata is not None:
                        entry.metadata.update(metadata)
                    
                    # Update version and timestamp
                    entry.version += 1
                    entry.updated_at = datetime.now(timezone.utc).isoformat()
                    
                    # Rebuild vectors and save
                    self._rebuild_domain_vectors(domain)
                    self._save_domain_knowledge(domain)
                    
                    logger.info(f"Updated knowledge entry {entry_id}")
                    return True
        
        logger.warning(f"Knowledge entry {entry_id} not found")
        return False
    
    def remove_knowledge_entry(self, entry_id: str) -> bool:
        """
        Remove a knowledge entry.
        
        Args:
            entry_id: Entry ID to remove
            
        Returns:
            Success status
        """
        for domain, entries in self.knowledge_entries.items():
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    # Remove entry
                    del entries[i]
                    
                    # Update metrics and vectors
                    self._update_domain_metrics(domain)
                    self._rebuild_domain_vectors(domain)
                    self._save_domain_knowledge(domain)
                    
                    logger.info(f"Removed knowledge entry {entry_id}")
                    return True
        
        logger.warning(f"Knowledge entry {entry_id} not found")
        return False
    
    def initialize_domain_knowledge(self, domain: str, knowledge_sources: List[str], 
                                   context_paragraph: str) -> bool:
        """
        Initialize knowledge base for a new domain.
        
        Args:
            domain: Domain name
            knowledge_sources: List of knowledge source files/paths
            context_paragraph: Domain-specific context paragraph
            
        Returns:
            Success status
        """
        try:
            # Add context paragraph as primary knowledge entry
            self.add_domain_knowledge(
                domain=domain,
                content=context_paragraph,
                source="domain_context",
                keywords=self._extract_keywords(context_paragraph),
                concepts=self._extract_concepts(context_paragraph, domain),
                importance_score=1.0,
                metadata={"type": "context_paragraph", "is_primary": True}
            )
            
            # Process knowledge sources
            for source in knowledge_sources:
                if os.path.isfile(source):
                    content = self._load_text_file(source)
                    self.add_domain_knowledge(
                        domain=domain,
                        content=content,
                        source=source,
                        keywords=self._extract_keywords(content),
                        concepts=self._extract_concepts(content, domain),
                        importance_score=0.8,
                        metadata={"type": "knowledge_source", "file_path": source}
                    )
                elif source.startswith("http"):
                    # Handle web sources
                    content = self._fetch_web_content(source)
                    if content:
                        self.add_domain_knowledge(
                            domain=domain,
                            content=content,
                            source=source,
                            keywords=self._extract_keywords(content),
                            concepts=self._extract_concepts(content, domain),
                            importance_score=0.7,
                            metadata={"type": "web_source", "url": source}
                        )
            
            logger.info(f"Initialized knowledge base for domain: {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize domain knowledge for {domain}: {str(e)}")
            return False

    def get_adaptive_context(self, domain: str, recent_feedback: List[Dict] = None,
                           performance_metrics: Dict[str, float] = None,
                           query: str = None) -> Dict[str, Any]:
        """
        Get adaptive context based on recent feedback and performance.
        
        Args:
            domain: Domain name
            recent_feedback: Recent user feedback
            performance_metrics: Current performance metrics
            query: Optional query for context filtering
            
        Returns:
            Adaptive domain context
        """
        base_context = self.get_domain_context(domain, query)
        
        if not recent_feedback and not performance_metrics:
            return base_context
        
        # Analyze feedback to identify improvement areas
        weak_areas = self._identify_weak_areas(domain, recent_feedback, performance_metrics)
        
        # Enhance context with targeted knowledge
        enhanced_entries = []
        for entry in base_context["entries"]:
            # Boost importance of entries related to weak areas
            boosted_score = entry.importance_score
            for weak_concept in weak_areas:
                if weak_concept.lower() in entry.content.lower() or weak_concept in entry.concepts:
                    boosted_score *= 1.5
            
            entry.importance_score = min(boosted_score, 1.0)
            enhanced_entries.append(entry)
        
        # Re-sort by boosted importance
        enhanced_entries.sort(key=lambda x: x.importance_score, reverse=True)
        
        adaptive_context = base_context.copy()
        adaptive_context["entries"] = enhanced_entries[:15]  # Increased limit for adaptive context
        adaptive_context["adaptation_applied"] = True
        adaptive_context["weak_areas"] = weak_areas
        
        return adaptive_context

    def update_knowledge_from_feedback(self, domain: str, feedback_data: List[Dict]) -> bool:
        """
        Update domain knowledge based on user feedback.
        
        Args:
            domain: Domain name
            feedback_data: List of feedback entries
            
        Returns:
            Success status
        """
        try:
            for feedback in feedback_data:
                if feedback.get("user_rating", 0) < 3.0 and feedback.get("expected_response"):
                    # Add corrective knowledge from low-rated responses
                    self.add_domain_knowledge(
                        domain=domain,
                        content=f"Query: {feedback['user_query']}\nExpected Response: {feedback['expected_response']}",
                        source="user_feedback",
                        keywords=self._extract_keywords(feedback["user_query"]),
                        concepts=self._extract_concepts(feedback["expected_response"], domain),
                        importance_score=0.9,  # High importance for corrective knowledge
                        metadata={
                            "type": "feedback_correction",
                            "original_rating": feedback["user_rating"],
                            "feedback_text": feedback.get("user_feedback", "")
                        }
                    )
                
                elif feedback.get("user_rating", 0) >= 4.0:
                    # Reinforce good responses
                    good_response_content = f"Query: {feedback['user_query']}\nGood Response: {feedback['model_response']}"
                    self.add_domain_knowledge(
                        domain=domain,
                        content=good_response_content,
                        source="positive_feedback",
                        keywords=self._extract_keywords(feedback["user_query"]),
                        concepts=self._extract_concepts(feedback["model_response"], domain),
                        importance_score=0.7,
                        metadata={
                            "type": "positive_reinforcement",
                            "rating": feedback["user_rating"]
                        }
                    )
            
            logger.info(f"Updated domain {domain} knowledge from {len(feedback_data)} feedback entries")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge from feedback for domain {domain}: {str(e)}")
            return False

    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get comprehensive summary of domain knowledge base."""
        if domain not in self.knowledge_entries:
            return {"domain": domain, "status": "not_found"}
        
        entries = self.knowledge_entries[domain]
        metrics = self.domain_metrics.get(domain)
        
        # Calculate statistics
        total_content_length = sum(len(entry.content) for entry in entries)
        avg_importance = sum(entry.importance_score for entry in entries) / len(entries)
        
        # Get unique sources and concepts
        sources = set(entry.source for entry in entries)
        concepts = set()
        for entry in entries:
            concepts.update(entry.concepts)
        
        # Recent updates
        recent_entries = [entry for entry in entries 
                         if datetime.fromisoformat(entry.updated_at) > 
                         datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)]
        
        summary = {
            "domain": domain,
            "total_entries": len(entries),
            "total_content_length": total_content_length,
            "average_importance": avg_importance,
            "unique_sources": list(sources),
            "unique_concepts": list(concepts),
            "recent_updates": len(recent_entries),
            "metrics": asdict(metrics) if metrics else None,
            "last_updated": max(entry.updated_at for entry in entries) if entries else None
        }
        
        return summary

    def _similarity_search(self, domain: str, query: str, limit: int) -> List[DomainKnowledgeEntry]:
        """Perform similarity search within a domain."""
        if domain not in self.vectorizers or domain not in self.content_vectors:
            return []
        
        vectorizer = self.vectorizers[domain]
        content_vectors = self.content_vectors[domain]
        
        # Vectorize query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, content_vectors)[0]
        
        # Get top similar entries
        entries = self.knowledge_entries[domain]
        indexed_similarities = [(i, sim) for i, sim in enumerate(similarities)]
        indexed_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [entries[i] for i, _ in indexed_similarities[:limit] if _ > 0.1]
    
    def _rebuild_domain_vectors(self, domain: str):
        """Rebuild TF-IDF vectors for a domain."""
        if domain not in self.knowledge_entries or not self.knowledge_entries[domain]:
            return
        
        entries = self.knowledge_entries[domain]
        contents = [entry.content for entry in entries]
        
        # Create or update vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        vectors = vectorizer.fit_transform(contents)
        
        self.vectorizers[domain] = vectorizer
        self.content_vectors[domain] = vectors
    
    def _extract_domain_keywords(self, domain: str) -> List[str]:
        """Extract top keywords for a domain."""
        if domain not in self.knowledge_entries:
            return []
        
        all_keywords = []
        for entry in self.knowledge_entries[domain]:
            all_keywords.extend(entry.keywords)
        
        # Count and return top keywords
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    def _extract_domain_concepts(self, domain: str) -> List[str]:
        """Extract top concepts for a domain."""
        if domain not in self.knowledge_entries:
            return []
        
        all_concepts = []
        for entry in self.knowledge_entries[domain]:
            all_concepts.extend(entry.concepts)
        
        # Count and return top concepts
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        return sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    def _update_domain_metrics(self, domain: str):
        """Update metrics for a domain."""
        if domain not in self.knowledge_entries:
            return
        
        entries = self.knowledge_entries[domain]
        total_tokens = sum(len(entry.content.split()) for entry in entries)
        avg_importance = np.mean([entry.importance_score for entry in entries]) if entries else 0.0
        
        # Calculate quality score based on various factors
        quality_score = self._calculate_quality_score(domain)
        
        metrics = DomainMetrics(
            domain_name=domain,
            total_documents=len(entries),
            total_tokens=total_tokens,
            quality_score=quality_score,
            coverage_score=min(total_tokens / 10000, 1.0),  # Simple coverage metric
            last_updated=datetime.now(timezone.utc).isoformat(),
            active_models=[],  # Will be populated by model management
            feedback_count=0,  # Will be populated by feedback system
            improvement_rate=0.0  # Will be calculated by feedback analysis
        )
        
        self.domain_metrics[domain] = metrics
    
    def _calculate_quality_score(self, domain: str) -> float:
        """Calculate quality score for a domain."""
        if domain not in self.knowledge_entries:
            return 0.0
        
        entries = self.knowledge_entries[domain]
        if not entries:
            return 0.0
        
        # Quality factors
        avg_length = np.mean([len(entry.content) for entry in entries])
        avg_importance = np.mean([entry.importance_score for entry in entries])
        keyword_diversity = len(set(kw for entry in entries for kw in entry.keywords))
        
        # Normalize and combine factors
        length_score = min(avg_length / 1000, 1.0)  # Normalize to 1000 chars
        importance_score = avg_importance
        diversity_score = min(keyword_diversity / 50, 1.0)  # Normalize to 50 keywords
        
        return (length_score + importance_score + diversity_score) / 3
    
    def _analyze_feedback(self, feedback_data: List[Dict]) -> List[str]:
        """Analyze feedback to identify improvement areas."""
        improvement_areas = []
        
        for feedback in feedback_data:
            if feedback.get("rating", 5) < 4:  # Poor ratings
                # Extract keywords from negative feedback
                if "keywords" in feedback:
                    improvement_areas.extend(feedback["keywords"])
                if "concepts" in feedback:
                    improvement_areas.extend(feedback["concepts"])
        
        # Return most common improvement areas
        area_counts = {}
        for area in improvement_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
        
        return [area for area, count in sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    def _calculate_adaptive_score(self, entry_dict: Dict, improvement_areas: List[str], 
                                 performance_metrics: Dict = None) -> float:
        """Calculate adaptive importance score for an entry."""
        base_score = entry_dict.get("importance_score", 1.0)
        
        # Boost score if entry relates to improvement areas
        relevance_boost = 0.0
        entry_keywords = entry_dict.get("keywords", [])
        entry_concepts = entry_dict.get("concepts", [])
        
        for area in improvement_areas:
            if area in entry_keywords or area in entry_concepts:
                relevance_boost += 0.2
        
        # Performance-based adjustments
        performance_adjustment = 0.0
        if performance_metrics:
            if performance_metrics.get("accuracy", 1.0) < 0.8:
                performance_adjustment += 0.1
        
        return base_score + relevance_boost + performance_adjustment
    
    def _identify_weak_areas(self, domain: str, feedback: List[Dict], 
                           metrics: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement based on feedback and metrics."""
        weak_areas = []
        
        if not feedback:
            return weak_areas
        
        # Analyze low-rated feedback for common patterns
        low_rated_feedback = [f for f in feedback if f.get("user_rating", 0) < 3.0]
        
        if low_rated_feedback:
            # Extract common terms from low-rated queries
            low_rated_content = " ".join([f.get("user_query", "") + " " + f.get("user_feedback", "") 
                                        for f in low_rated_feedback])
            
            # Simple keyword extraction (could be enhanced with NLP)
            words = low_rated_content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent terms as weak areas
            weak_areas = [word for word, freq in sorted(word_freq.items(), 
                                                      key=lambda x: x[1], reverse=True)[:5]]
        
        return weak_areas

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        # Simple keyword extraction - could be enhanced with NLP libraries
        words = text.lower().split()
        
        # Filter out common words and short words
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word.strip(".,!?;:") for word in words 
                   if len(word) > 4 and word.lower() not in stopwords]
        
        # Return most frequent unique keywords
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        return [k for k, v in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]]

    def _extract_concepts(self, text: str, domain: str) -> List[str]:
        """Extract domain-specific concepts from text."""
        # Domain-specific concept patterns (simplified)
        concept_patterns = {
            "technical": ["api", "database", "algorithm", "framework", "architecture", "security"],
            "medical": ["diagnosis", "treatment", "patient", "clinical", "therapeutic", "syndrome"],
            "legal": ["contract", "liability", "regulation", "compliance", "statute", "jurisdiction"],
            "financial": ["investment", "portfolio", "risk", "return", "valuation", "market"]
        }
        
        domain_patterns = concept_patterns.get(domain.lower(), [])
        text_lower = text.lower()
        
        found_concepts = []
        for pattern in domain_patterns:
            if pattern in text_lower:
                found_concepts.append(pattern)
        
        return found_concepts

    def _load_text_file(self, file_path: str) -> str:
        """Load content from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {str(e)}")
            return ""

    def _fetch_web_content(self, url: str) -> str:
        """Fetch content from a web URL."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            return text
            
        except Exception as e:
            logger.error(f"Failed to fetch web content from {url}: {str(e)}")
            return ""
    
    def _load_knowledge_bases(self):
        """Load existing knowledge bases from persistent storage."""
        for domain_path in self.base_path.glob("*.json"):
            domain = domain_path.stem
            try:
                with open(domain_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                entries = []
                for entry_data in data.get("entries", []):
                    entries.append(DomainKnowledgeEntry(**entry_data))
                
                self.knowledge_entries[domain] = entries
                
                if data.get("metrics"):
                    self.domain_metrics[domain] = DomainMetrics(**data["metrics"])
                
                # Rebuild vectors
                self._rebuild_domain_vectors(domain)
                
                logger.info(f"Loaded {len(entries)} entries for domain {domain}")
                
            except Exception as e:
                logger.error(f"Failed to load knowledge base for {domain}: {str(e)}")
    
    def _save_domain_knowledge(self, domain: str):
        """Save domain knowledge base to persistent storage."""
        if domain not in self.knowledge_entries:
            return
        
        data = {
            "domain": domain,
            "entries": [asdict(entry) for entry in self.knowledge_entries[domain]],
            "metrics": asdict(self.domain_metrics.get(domain)) if domain in self.domain_metrics else None,
            "last_saved": datetime.now(timezone.utc).isoformat()
        }
        
        domain_path = self.base_path / f"{domain}.json"
        try:
            with open(domain_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save knowledge base for {domain}: {str(e)}")
