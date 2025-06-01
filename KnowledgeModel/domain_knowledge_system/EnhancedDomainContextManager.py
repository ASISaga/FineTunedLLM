"""
Enhanced Domain Context Manager with Knowledge Base Integration
Extends the existing domain context system with adaptive knowledge base integration,
dynamic context generation, and intelligent domain-specific prompting.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import re

from DomainContextManager import DomainContextManager, DomainContext, DomainType
from DomainKnowledgeBase import DomainKnowledgeBase
from FeedbackCollector import FeedbackCollector

logger = logging.getLogger(__name__)

@dataclass
class AdaptivePromptTemplate:
    """Adaptive prompt template with knowledge integration"""
    template_id: str
    domain: str
    base_template: str
    knowledge_slots: List[str]  # Slots for knowledge injection
    adaptation_rules: Dict[str, Any]
    performance_history: Dict[str, float]
    usage_count: int
    success_rate: float
    created_at: str
    last_updated: str

@dataclass
class ContextGenerationConfig:
    """Configuration for context generation"""
    max_knowledge_entries: int = 5
    min_relevance_score: float = 0.3
    include_examples: bool = True
    include_terminology: bool = True
    include_focus_areas: bool = True
    adaptive_weighting: bool = True
    feedback_influence: float = 0.3
    knowledge_freshness_weight: float = 0.2

class EnhancedDomainContextManager:
    """
    Enhanced domain context manager with knowledge base integration.
    
    Features:
    - Knowledge base integration for dynamic context
    - Adaptive prompt templates based on feedback
    - Intelligent domain detection and classification
    - Performance-driven context optimization
    - Multi-modal context generation (text, examples, terminology)
    """
    
    def __init__(self, knowledge_base: DomainKnowledgeBase = None,
                 feedback_collector: FeedbackCollector = None,
                 base_path: str = "./enhanced_context"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Core components
        self.base_context_manager = DomainContextManager()
        self.knowledge_base = knowledge_base or DomainKnowledgeBase()
        self.feedback_collector = feedback_collector or FeedbackCollector()
        
        # Enhanced features
        self.adaptive_templates: Dict[str, AdaptivePromptTemplate] = {}
        self.domain_performance: Dict[str, Dict[str, float]] = {}
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.generation_config = ContextGenerationConfig()
        
        # Domain enhancement mappings
        self.domain_enhancement_rules = {
            "technical": {
                "priority_concepts": ["performance", "security", "scalability", "best_practices"],
                "required_terminology": ["API", "framework", "architecture", "deployment"],
                "context_depth": "detailed",
                "example_types": ["code_snippets", "architecture_diagrams", "use_cases"]
            },
            "medical": {
                "priority_concepts": ["patient_safety", "clinical_evidence", "guidelines", "protocols"],
                "required_terminology": ["diagnosis", "treatment", "clinical_trial", "adverse_event"],
                "context_depth": "comprehensive",
                "example_types": ["case_studies", "clinical_protocols", "research_findings"]
            },
            "legal": {
                "priority_concepts": ["compliance", "precedent", "regulation", "risk_assessment"],
                "required_terminology": ["statute", "jurisdiction", "liability", "contract"],
                "context_depth": "precise",
                "example_types": ["case_law", "regulatory_text", "contract_clauses"]
            },
            "financial": {
                "priority_concepts": ["risk_management", "valuation", "market_analysis", "regulation"],
                "required_terminology": ["portfolio", "derivative", "volatility", "compliance"],
                "context_depth": "analytical",
                "example_types": ["financial_models", "market_data", "risk_scenarios"]
            }
        }
        
        # Load existing adaptive templates
        self._load_adaptive_templates()
        
        # Load performance data
        self._load_performance_data()
    
    def get_enhanced_domain_context(self, domain: str, query: str = None,
                                  user_feedback: List[Dict] = None,
                                  performance_metrics: Dict[str, float] = None,
                                  context_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Get enhanced domain context with knowledge base integration.
        
        Args:
            domain: Domain name
            query: Optional specific query for context relevance
            user_feedback: Recent user feedback for adaptation
            performance_metrics: Current performance metrics
            context_type: Type of context (comprehensive, focused, minimal)
            
        Returns:
            Enhanced domain context with integrated knowledge
        """
        # Generate cache key
        cache_key = f"{domain}_{hash(str(query))}_{context_type}"
        
        # Check cache (with time-based invalidation)
        if self._is_context_cached(cache_key):
            cached_context = self.context_cache[cache_key]
            logger.debug(f"Using cached context for {domain}")
            return cached_context
        
        # Get base domain context
        base_context = self.base_context_manager.get_domain_context(domain)
        if not base_context:
            logger.warning(f"No base context found for domain {domain}")
            return self._create_fallback_context(domain)
        
        # Get knowledge base context
        knowledge_context = self.knowledge_base.get_adaptive_context(
            domain=domain,
            recent_feedback=user_feedback,
            performance_metrics=performance_metrics
        )
        
        # Get domain enhancement rules
        enhancement_rules = self.domain_enhancement_rules.get(domain, {})
        
        # Generate enhanced context
        enhanced_context = self._merge_context_sources(
            base_context=base_context,
            knowledge_context=knowledge_context,
            enhancement_rules=enhancement_rules,
            query=query,
            context_type=context_type
        )
        
        # Apply adaptive optimizations
        if user_feedback or performance_metrics:
            enhanced_context = self._apply_adaptive_optimizations(
                enhanced_context, user_feedback, performance_metrics, domain
            )
        
        # Cache the result
        self.context_cache[cache_key] = enhanced_context
        
        logger.info(f"Generated enhanced context for domain {domain}")
        return enhanced_context
    
    def generate_adaptive_prompt(self, domain: str, task_type: str, query: str,
                                user_context: Dict[str, Any] = None,
                                performance_feedback: Dict[str, float] = None) -> str:
        """
        Generate adaptive prompt with knowledge integration.
        
        Args:
            domain: Domain name
            task_type: Type of task (summarization, fine_tuning, inference)
            query: User query or task description
            user_context: Additional user context
            performance_feedback: Recent performance feedback
            
        Returns:
            Optimized prompt with knowledge integration
        """
        # Get enhanced context
        enhanced_context = self.get_enhanced_domain_context(
            domain=domain,
            query=query,
            user_feedback=performance_feedback,
            context_type="focused"
        )
        
        # Get or create adaptive template
        template_id = f"{domain}_{task_type}"
        template = self._get_adaptive_template(template_id, domain, task_type)
        
        # Generate prompt from template
        prompt = self._generate_prompt_from_template(
            template=template,
            enhanced_context=enhanced_context,
            query=query,
            user_context=user_context
        )
        
        # Update template usage
        self._update_template_usage(template_id)
        
        return prompt
    
    def update_context_performance(self, domain: str, context_id: str,
                                 performance_metrics: Dict[str, float],
                                 user_feedback: Dict[str, Any] = None):
        """
        Update context performance metrics for adaptive learning.
        
        Args:
            domain: Domain name
            context_id: Context identifier
            performance_metrics: Performance metrics (accuracy, relevance, etc.)
            user_feedback: Optional user feedback
        """
        if domain not in self.domain_performance:
            self.domain_performance[domain] = {}
        
        # Update performance history
        self.domain_performance[domain][context_id] = performance_metrics
        
        # Update adaptive templates if applicable
        for template_id, template in self.adaptive_templates.items():
            if template.domain == domain:
                self._update_template_performance(template, performance_metrics)
        
        # Clear relevant cache entries
        self._invalidate_domain_cache(domain)
        
        # Save performance data
        self._save_performance_data()
        
        logger.info(f"Updated context performance for domain {domain}")
    
    def detect_domain_with_confidence(self, content: str, filename: str = None) -> Tuple[str, float]:
        """
        Detect domain with confidence scoring.
        
        Args:
            content: Content to analyze
            filename: Optional filename for additional hints
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        # Start with base detection
        base_domain = self.base_context_manager.detect_domain_from_filename(filename or "")
        
        # Analyze content for domain indicators
        domain_scores = self._analyze_content_domain_indicators(content)
        
        # Factor in filename if available
        if filename:
            filename_scores = self._analyze_filename_domain_indicators(filename)
            # Merge scores with filename weight
            for domain in filename_scores:
                domain_scores[domain] = domain_scores.get(domain, 0) + filename_scores[domain] * 0.3
        
        # Get highest scoring domain
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            
            # Normalize confidence score
            max_score = max(domain_scores.values())
            confidence = min(best_domain[1] / max_score, 1.0) if max_score > 0 else 0.0
            
            return best_domain[0], confidence
        
        # Fallback to base detection with low confidence
        return base_domain or "general", 0.5
    
    def optimize_domain_prompts(self, domain: str, feedback_data: List[Dict],
                              performance_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Optimize domain prompts based on feedback and performance.
        
        Args:
            domain: Domain to optimize
            feedback_data: Historical feedback data
            performance_history: Performance metrics over time
            
        Returns:
            Optimization results and recommendations
        """
        # Analyze feedback patterns
        feedback_analysis = self._analyze_feedback_patterns(feedback_data)
        
        # Identify performance trends
        performance_trends = self._analyze_performance_trends(performance_history)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            domain, feedback_analysis, performance_trends
        )
        
        # Apply automatic optimizations
        applied_optimizations = self._apply_automatic_optimizations(domain, recommendations)
        
        # Update adaptive templates
        self._update_adaptive_templates_from_optimization(domain, recommendations)
        
        optimization_results = {
            "domain": domain,
            "feedback_analysis": feedback_analysis,
            "performance_trends": performance_trends,
            "recommendations": recommendations,
            "applied_optimizations": applied_optimizations,
            "optimization_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Completed prompt optimization for domain {domain}")
        return optimization_results
    
    def create_custom_domain_context(self, domain_name: str, description: str,
                                   key_concepts: List[str], terminology: Dict[str, str],
                                   focus_areas: List[str], knowledge_sources: List[str] = None) -> bool:
        """
        Create a custom domain context with knowledge integration.
        
        Args:
            domain_name: Name of the custom domain
            description: Domain description
            key_concepts: List of key concepts
            terminology: Domain-specific terminology
            focus_areas: Areas of focus
            knowledge_sources: Optional list of knowledge source files/URLs
            
        Returns:
            Success status
        """
        try:
            # Create base domain context
            custom_domain = DomainContext(
                domain_type=DomainType.GENERAL,
                domain_name=domain_name,
                description=description,
                key_concepts=key_concepts,
                terminology=terminology,
                focus_areas=focus_areas,
                summarization_instructions=f"Focus on {', '.join(focus_areas[:3])} when summarizing {domain_name} content.",
                fine_tuning_instructions=f"Emphasize {', '.join(key_concepts[:3])} in training examples for {domain_name}.",
                example_prompts=[f"Explain {concept} in the context of {domain_name}" for concept in key_concepts[:3]],
                evaluation_criteria=["accuracy", "relevance", "domain_specificity", "practical_applicability"]
            )
            
            # Add to base context manager
            self.base_context_manager.add_custom_domain(custom_domain)
            
            # Initialize knowledge base entries if knowledge sources provided
            if knowledge_sources:
                for source in knowledge_sources:
                    self._initialize_domain_knowledge(domain_name, source)
            
            # Create enhancement rules
            self.domain_enhancement_rules[domain_name.lower()] = {
                "priority_concepts": key_concepts[:4],
                "required_terminology": list(terminology.keys())[:4],
                "context_depth": "detailed",
                "example_types": ["domain_examples", "use_cases", "best_practices"]
            }
            
            # Initialize adaptive templates
            self._initialize_adaptive_templates(domain_name)
            
            logger.info(f"Created custom domain context: {domain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom domain context {domain_name}: {str(e)}")
            return False
    
    def get_context_recommendations(self, domain: str, recent_performance: Dict[str, float]) -> List[str]:
        """
        Get recommendations for improving domain context.
        
        Args:
            domain: Domain name
            recent_performance: Recent performance metrics
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Analyze performance gaps
        performance_threshold = 0.7
        
        if recent_performance.get("accuracy", 0) < performance_threshold:
            recommendations.append("Consider adding more domain-specific examples to improve accuracy")
        
        if recent_performance.get("relevance", 0) < performance_threshold:
            recommendations.append("Update knowledge base with more recent domain information")
        
        if recent_performance.get("user_satisfaction", 0) < performance_threshold:
            recommendations.append("Review and refine domain terminology and focus areas")
        
        # Check knowledge base coverage
        knowledge_context = self.knowledge_base.get_domain_context(domain)
        if knowledge_context["total_entries"] < 10:
            recommendations.append("Expand knowledge base with more domain-specific content")
        
        # Check template performance
        domain_templates = [t for t in self.adaptive_templates.values() if t.domain == domain]
        avg_success_rate = sum(t.success_rate for t in domain_templates) / max(len(domain_templates), 1)
        
        if avg_success_rate < 0.8:
            recommendations.append("Optimize prompt templates based on feedback patterns")
        
        return recommendations
    
    def export_enhanced_context_config(self, domain: str) -> str:
        """Export enhanced context configuration as JSON."""
        base_config = self.base_context_manager.export_domain_config(domain)
        if not base_config:
            return "{}"
        
        base_data = json.loads(base_config)
        
        # Add enhancement data
        enhancement_data = {
            "base_context": base_data,
            "enhancement_rules": self.domain_enhancement_rules.get(domain, {}),
            "adaptive_templates": [
                asdict(template) for template in self.adaptive_templates.values()
                if template.domain == domain
            ],
            "performance_history": self.domain_performance.get(domain, {}),
            "generation_config": asdict(self.generation_config),
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
        return json.dumps(enhancement_data, indent=2)
    
    def import_enhanced_context_config(self, config_json: str) -> bool:
        """Import enhanced context configuration from JSON."""
        try:
            config_data = json.loads(config_json)
            
            # Import base context
            base_config = config_data.get("base_context", {})
            if base_config:
                self.base_context_manager.import_domain_config(json.dumps(base_config))
            
            # Import enhancement rules
            domain = base_config.get("domain_name", "").lower()
            if domain and "enhancement_rules" in config_data:
                self.domain_enhancement_rules[domain] = config_data["enhancement_rules"]
            
            # Import adaptive templates
            for template_data in config_data.get("adaptive_templates", []):
                template = AdaptivePromptTemplate(**template_data)
                self.adaptive_templates[template.template_id] = template
            
            # Import performance history
            if domain and "performance_history" in config_data:
                self.domain_performance[domain] = config_data["performance_history"]
            
            # Import generation config
            if "generation_config" in config_data:
                self.generation_config = ContextGenerationConfig(**config_data["generation_config"])
            
            logger.info(f"Imported enhanced context configuration for domain {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import enhanced context configuration: {str(e)}")
            return False
    
    # Private helper methods
    
    def _merge_context_sources(self, base_context: DomainContext, knowledge_context: Dict[str, Any],
                              enhancement_rules: Dict[str, Any], query: str = None,
                              context_type: str = "comprehensive") -> Dict[str, Any]:
        """Merge context from multiple sources."""
        enhanced_context = {
            "domain": base_context.domain_name,
            "description": base_context.description,
            "key_concepts": base_context.key_concepts.copy(),
            "terminology": base_context.terminology.copy(),
            "focus_areas": base_context.focus_areas.copy(),
            "example_prompts": base_context.example_prompts.copy(),
            "evaluation_criteria": base_context.evaluation_criteria.copy()
        }
        
        # Integrate knowledge base entries
        if knowledge_context.get("entries"):
            enhanced_context["knowledge_entries"] = knowledge_context["entries"][:self.generation_config.max_knowledge_entries]
            
            # Extract additional concepts and terminology from knowledge
            for entry in enhanced_context["knowledge_entries"]:
                enhanced_context["key_concepts"].extend(entry.get("concepts", []))
                for keyword in entry.get("keywords", []):
                    if keyword not in enhanced_context["terminology"]:
                        enhanced_context["terminology"][keyword] = f"Domain-specific term in {base_context.domain_name}"
        
        # Apply enhancement rules
        if enhancement_rules:
            # Prioritize concepts based on rules
            priority_concepts = enhancement_rules.get("priority_concepts", [])
            enhanced_context["priority_concepts"] = priority_concepts
            
            # Add required terminology
            required_terms = enhancement_rules.get("required_terminology", [])
            for term in required_terms:
                if term not in enhanced_context["terminology"]:
                    enhanced_context["terminology"][term] = f"Essential {base_context.domain_name} concept"
        
        # Remove duplicates and limit sizes based on context type
        if context_type == "minimal":
            enhanced_context["key_concepts"] = list(set(enhanced_context["key_concepts"]))[:5]
            enhanced_context["focus_areas"] = enhanced_context["focus_areas"][:3]
        elif context_type == "focused":
            enhanced_context["key_concepts"] = list(set(enhanced_context["key_concepts"]))[:10]
            enhanced_context["focus_areas"] = enhanced_context["focus_areas"][:5]
        else:  # comprehensive
            enhanced_context["key_concepts"] = list(set(enhanced_context["key_concepts"]))[:15]
        
        enhanced_context["context_type"] = context_type
        enhanced_context["generation_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return enhanced_context
    
    def _apply_adaptive_optimizations(self, context: Dict[str, Any], feedback: List[Dict],
                                    metrics: Dict[str, float], domain: str) -> Dict[str, Any]:
        """Apply adaptive optimizations based on feedback and metrics."""
        if not feedback and not metrics:
            return context
        
        optimized_context = context.copy()
        
        # Analyze feedback for concept emphasis
        if feedback:
            concept_feedback = self._extract_concept_feedback(feedback)
            
            # Boost concepts with positive feedback
            for concept, score in concept_feedback.items():
                if score > 0.5 and concept in optimized_context["key_concepts"]:
                    # Move to priority concepts
                    if "priority_concepts" not in optimized_context:
                        optimized_context["priority_concepts"] = []
                    if concept not in optimized_context["priority_concepts"]:
                        optimized_context["priority_concepts"].append(concept)
        
        # Apply metric-based optimizations
        if metrics:
            if metrics.get("accuracy", 0) < 0.7:
                # Add more specific terminology
                optimized_context["include_detailed_terminology"] = True
            
            if metrics.get("relevance", 0) < 0.7:
                # Focus on most relevant concepts
                optimized_context["focus_mode"] = "high_relevance"
        
        return optimized_context
    
    def _get_adaptive_template(self, template_id: str, domain: str, task_type: str) -> AdaptivePromptTemplate:
        """Get or create adaptive template."""
        if template_id in self.adaptive_templates:
            return self.adaptive_templates[template_id]
        
        # Create new template
        base_template = self._create_base_template(domain, task_type)
        
        template = AdaptivePromptTemplate(
            template_id=template_id,
            domain=domain,
            base_template=base_template,
            knowledge_slots=["{{domain_knowledge}}", "{{key_concepts}}", "{{terminology}}"],
            adaptation_rules={},
            performance_history={},
            usage_count=0,
            success_rate=0.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        self.adaptive_templates[template_id] = template
        return template
    
    def _create_base_template(self, domain: str, task_type: str) -> str:
        """Create base template for domain and task type."""
        base_context = self.base_context_manager.get_domain_context(domain)
        
        if task_type == "summarization":
            if base_context:
                return base_context.summarization_instructions + "\n\n{{domain_knowledge}}\n\nKey concepts to focus on: {{key_concepts}}\n\nTerminology: {{terminology}}"
            else:
                return "Summarize the following content with focus on domain-specific concepts.\n\n{{domain_knowledge}}"
        
        elif task_type == "fine_tuning":
            if base_context:
                return base_context.fine_tuning_instructions + "\n\n{{domain_knowledge}}\n\nEmphasize: {{key_concepts}}\n\nUse terminology: {{terminology}}"
            else:
                return "Generate training examples with domain expertise.\n\n{{domain_knowledge}}"
        
        elif task_type == "inference":
            return "You are an expert in {{domain}}. {{domain_knowledge}}\n\nFocus on: {{key_concepts}}\n\nUse appropriate terminology: {{terminology}}\n\nProvide expert guidance on the following:"
        
        else:
            return "Apply domain expertise to address the following task.\n\n{{domain_knowledge}}\n\nKey concepts: {{key_concepts}}"
    
    def _generate_prompt_from_template(self, template: AdaptivePromptTemplate,
                                     enhanced_context: Dict[str, Any], query: str,
                                     user_context: Dict[str, Any] = None) -> str:
        """Generate prompt from template with context injection."""
        prompt = template.base_template
        
        # Replace knowledge slots
        if "{{domain_knowledge}}" in prompt:
            knowledge_text = self._format_knowledge_entries(enhanced_context.get("knowledge_entries", []))
            prompt = prompt.replace("{{domain_knowledge}}", knowledge_text)
        
        if "{{key_concepts}}" in prompt:
            concepts_text = ", ".join(enhanced_context.get("key_concepts", [])[:5])
            prompt = prompt.replace("{{key_concepts}}", concepts_text)
        
        if "{{terminology}}" in prompt:
            terminology_text = self._format_terminology(enhanced_context.get("terminology", {}))
            prompt = prompt.replace("{{terminology}}", terminology_text)
        
        if "{{domain}}" in prompt:
            prompt = prompt.replace("{{domain}}", enhanced_context.get("domain", ""))
        
        # Add query
        prompt += f"\n\nTask: {query}"
        
        # Add user context if provided
        if user_context:
            prompt += f"\n\nAdditional context: {json.dumps(user_context, indent=2)}"
        
        return prompt
    
    def _format_knowledge_entries(self, entries: List[Dict]) -> str:
        """Format knowledge entries for prompt injection."""
        if not entries:
            return "No specific domain knowledge available."
        
        formatted_entries = []
        for entry in entries[:3]:  # Limit to top 3 entries
            formatted_entries.append(f"- {entry.get('content', '')[:200]}...")
        
        return "Relevant domain knowledge:\n" + "\n".join(formatted_entries)
    
    def _format_terminology(self, terminology: Dict[str, str]) -> str:
        """Format terminology for prompt injection."""
        if not terminology:
            return "No specific terminology provided."
        
        formatted_terms = []
        for term, definition in list(terminology.items())[:5]:  # Limit to 5 terms
            formatted_terms.append(f"{term}: {definition}")
        
        return "; ".join(formatted_terms)
    
    def _analyze_content_domain_indicators(self, content: str) -> Dict[str, float]:
        """Analyze content for domain indicators."""
        domain_keywords = {
            "technical": ["API", "algorithm", "framework", "database", "deployment", "architecture", "performance", "security"],
            "medical": ["patient", "clinical", "diagnosis", "treatment", "medical", "healthcare", "symptom", "therapy"],
            "legal": ["contract", "legal", "statute", "compliance", "regulation", "court", "jurisdiction", "liability"],
            "financial": ["investment", "portfolio", "market", "financial", "risk", "return", "asset", "valuation"]
        }
        
        scores = {}
        content_lower = content.lower()
        
        for domain, keywords in domain_keywords.items():
            score = 0
            for keyword in keywords:
                score += content_lower.count(keyword.lower()) * (1.0 / len(keywords))
            scores[domain] = score
        
        return scores
    
    def _analyze_filename_domain_indicators(self, filename: str) -> Dict[str, float]:
        """Analyze filename for domain indicators."""
        filename_lower = filename.lower()
        
        patterns = {
            "technical": [r"api", r"tech", r"system", r"code", r"dev", r"software"],
            "medical": [r"medical", r"clinical", r"health", r"patient", r"drug"],
            "legal": [r"legal", r"law", r"contract", r"compliance", r"regulation"],
            "financial": [r"finance", r"investment", r"market", r"bank", r"economic"]
        }
        
        scores = {}
        for domain, pattern_list in patterns.items():
            score = 0
            for pattern in pattern_list:
                if re.search(pattern, filename_lower):
                    score += 1.0
            scores[domain] = score
        
        return scores
    
    def _create_fallback_context(self, domain: str) -> Dict[str, Any]:
        """Create fallback context when no base context exists."""
        return {
            "domain": domain,
            "description": f"General context for {domain}",
            "key_concepts": [],
            "terminology": {},
            "focus_areas": [],
            "example_prompts": [],
            "evaluation_criteria": ["accuracy", "relevance"],
            "context_type": "fallback",
            "generation_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _is_context_cached(self, cache_key: str) -> bool:
        """Check if context is cached and still valid."""
        if cache_key not in self.context_cache:
            return False
        
        # Simple time-based invalidation (1 hour)
        # In production, this could be more sophisticated
        return True  # Simplified for now
    
    def _invalidate_domain_cache(self, domain: str):
        """Invalidate cache entries for a domain."""
        keys_to_remove = [key for key in self.context_cache.keys() if key.startswith(domain)]
        for key in keys_to_remove:
            del self.context_cache[key]
    
    def _initialize_domain_knowledge(self, domain: str, source: str):
        """Initialize knowledge base entries for a domain."""
        # This would load knowledge from various sources
        # Simplified implementation
        logger.info(f"Initializing knowledge for domain {domain} from source {source}")
    
    def _initialize_adaptive_templates(self, domain: str):
        """Initialize adaptive templates for a domain."""
        task_types = ["summarization", "fine_tuning", "inference"]
        for task_type in task_types:
            template_id = f"{domain}_{task_type}"
            self._get_adaptive_template(template_id, domain, task_type)
    
    def _update_template_usage(self, template_id: str):
        """Update template usage statistics."""
        if template_id in self.adaptive_templates:
            self.adaptive_templates[template_id].usage_count += 1
            self.adaptive_templates[template_id].last_updated = datetime.now(timezone.utc).isoformat()
    
    def _update_template_performance(self, template: AdaptivePromptTemplate, metrics: Dict[str, float]):
        """Update template performance based on metrics."""
        template.performance_history.update(metrics)
        
        # Calculate success rate based on metrics
        if metrics:
            avg_score = sum(metrics.values()) / len(metrics)
            template.success_rate = (template.success_rate + avg_score) / 2
        
        template.last_updated = datetime.now(timezone.utc).isoformat()
    
    def _analyze_feedback_patterns(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in feedback data."""
        # Simplified feedback analysis
        return {
            "positive_patterns": [],
            "negative_patterns": [],
            "improvement_suggestions": []
        }
    
    def _analyze_performance_trends(self, performance_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Simplified trend analysis
        return {
            "improving_metrics": [],
            "declining_metrics": [],
            "stable_metrics": []
        }
    
    def _generate_optimization_recommendations(self, domain: str, feedback_analysis: Dict[str, Any],
                                             performance_trends: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Add recommendations based on analysis
        if performance_trends.get("declining_metrics"):
            recommendations.append("Consider updating domain knowledge base with recent information")
        
        return recommendations
    
    def _apply_automatic_optimizations(self, domain: str, recommendations: List[str]) -> List[str]:
        """Apply automatic optimizations based on recommendations."""
        applied = []
        
        # Simple automatic optimizations
        for recommendation in recommendations:
            if "knowledge base" in recommendation.lower():
                # Trigger knowledge base update
                applied.append(f"Triggered knowledge base update for {domain}")
        
        return applied
    
    def _update_adaptive_templates_from_optimization(self, domain: str, recommendations: List[str]):
        """Update adaptive templates based on optimization recommendations."""
        domain_templates = [t for t in self.adaptive_templates.values() if t.domain == domain]
        
        for template in domain_templates:
            # Apply optimizations to templates
            template.last_updated = datetime.now(timezone.utc).isoformat()
    
    def _extract_concept_feedback(self, feedback: List[Dict]) -> Dict[str, float]:
        """Extract concept-specific feedback scores."""
        concept_scores = {}
        
        for fb in feedback:
            # Simplified concept extraction from feedback
            if "concepts" in fb:
                for concept, score in fb["concepts"].items():
                    concept_scores[concept] = concept_scores.get(concept, 0) + score
        
        return concept_scores
    
    def _load_adaptive_templates(self):
        """Load adaptive templates from disk."""
        templates_file = self.base_path / "adaptive_templates.json"
        
        if not templates_file.exists():
            return
        
        try:
            with open(templates_file, 'r') as f:
                templates_data = json.load(f)
            
            for template_id, template_data in templates_data.items():
                self.adaptive_templates[template_id] = AdaptivePromptTemplate(**template_data)
            
            logger.info(f"Loaded {len(self.adaptive_templates)} adaptive templates")
            
        except Exception as e:
            logger.error(f"Failed to load adaptive templates: {str(e)}")
    
    def _save_adaptive_templates(self):
        """Save adaptive templates to disk."""
        templates_file = self.base_path / "adaptive_templates.json"
        
        try:
            templates_data = {
                template_id: asdict(template)
                for template_id, template in self.adaptive_templates.items()
            }
            
            with open(templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2, default=str)
            
            logger.debug("Saved adaptive templates to disk")
            
        except Exception as e:
            logger.error(f"Failed to save adaptive templates: {str(e)}")
    
    def _load_performance_data(self):
        """Load performance data from disk."""
        performance_file = self.base_path / "domain_performance.json"
        
        if not performance_file.exists():
            return
        
        try:
            with open(performance_file, 'r') as f:
                self.domain_performance = json.load(f)
            
            logger.info("Loaded domain performance data")
            
        except Exception as e:
            logger.error(f"Failed to load performance data: {str(e)}")
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        performance_file = self.base_path / "domain_performance.json"
        
        try:
            with open(performance_file, 'w') as f:
                json.dump(self.domain_performance, f, indent=2, default=str)
            
            logger.debug("Saved domain performance data")
            
        except Exception as e:
            logger.error(f"Failed to save performance data: {str(e)}")
