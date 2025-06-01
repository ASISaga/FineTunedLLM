"""
Feedback Collection and Processing System
Manages real-world feedback for continuous model improvement.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    RATING = "rating"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    COMPLAINT = "complaint"
    PRAISE = "praise"
    USAGE_ANALYTICS = "usage_analytics"

class FeedbackSeverity(Enum):
    """Severity levels for feedback"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    id: str
    domain: str
    model_id: str
    feedback_type: FeedbackType
    severity: FeedbackSeverity
    rating: Optional[float]  # 1-5 scale
    user_input: str
    model_output: str
    expected_output: Optional[str]
    correction: Optional[str]
    user_comment: Optional[str]
    keywords: List[str]
    concepts: List[str]
    session_id: str
    user_id: Optional[str]
    timestamp: str
    processed: bool
    metadata: Dict[str, Any]

@dataclass
class FeedbackAnalysis:
    """Analysis results from feedback processing"""
    domain: str
    analysis_period: str
    total_feedback: int
    avg_rating: float
    feedback_distribution: Dict[str, int]
    top_issues: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    performance_trends: Dict[str, float]
    critical_areas: List[str]
    confidence_score: float

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model"""
    model_id: str
    domain: str
    accuracy_score: float
    user_satisfaction: float
    response_quality: float
    consistency_score: float
    relevance_score: float
    feedback_volume: int
    last_updated: str

class FeedbackCollector:
    """
    Collects and processes feedback for continuous model improvement.
    
    Features:
    - Multi-channel feedback collection
    - Real-time feedback processing
    - Performance analytics
    - Improvement recommendation generation
    - Trend analysis and alerting
    """
    
    def __init__(self, storage_path: str = "./feedback_data"):
        self.storage_path = storage_path
        self.feedback_entries: Dict[str, List[FeedbackEntry]] = defaultdict(list)
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.analysis_cache: Dict[str, FeedbackAnalysis] = {}
        
        # Initialize storage
        import os
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing feedback
        self._load_feedback_data()
    
    def collect_feedback(self, domain: str, model_id: str, user_input: str,
                        model_output: str, feedback_type: FeedbackType,
                        rating: Optional[float] = None,
                        expected_output: Optional[str] = None,
                        correction: Optional[str] = None,
                        user_comment: Optional[str] = None,
                        user_id: Optional[str] = None,
                        session_id: str = None,
                        metadata: Dict[str, Any] = None) -> str:
        """
        Collect feedback from users about model performance.
        
        Args:
            domain: Domain name
            model_id: Model identifier
            user_input: Original user input
            model_output: Model's output
            feedback_type: Type of feedback
            rating: User rating (1-5)
            expected_output: What the user expected
            correction: User's correction
            user_comment: Additional comments
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Feedback entry ID
        """
        # Generate feedback ID
        timestamp = datetime.now(timezone.utc)
        feedback_id = f"{domain}_{model_id}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Determine severity
        severity = self._determine_severity(rating, feedback_type, correction)
        
        # Extract keywords and concepts
        keywords, concepts = self._extract_feedback_features(
            user_input, model_output, user_comment, correction
        )
        
        # Create feedback entry
        feedback = FeedbackEntry(
            id=feedback_id,
            domain=domain,
            model_id=model_id,
            feedback_type=feedback_type,
            severity=severity,
            rating=rating,
            user_input=user_input,
            model_output=model_output,
            expected_output=expected_output,
            correction=correction,
            user_comment=user_comment,
            keywords=keywords,
            concepts=concepts,
            session_id=session_id or feedback_id,
            user_id=user_id,
            timestamp=timestamp.isoformat(),
            processed=False,
            metadata=metadata or {}
        )
        
        # Store feedback
        self.feedback_entries[domain].append(feedback)
        
        # Process feedback immediately for critical issues
        if severity == FeedbackSeverity.CRITICAL:
            self._process_critical_feedback(feedback)
        
        # Update model metrics
        self._update_model_metrics(domain, model_id)
        
        # Save to persistent storage
        self._save_feedback_data(domain)
        
        logger.info(f"Collected {feedback_type.value} feedback for {domain} model {model_id}")
        return feedback_id
    
    def analyze_feedback(self, domain: str, days: int = 30, 
                        model_id: Optional[str] = None) -> FeedbackAnalysis:
        """
        Analyze feedback for a domain over a specified period.
        
        Args:
            domain: Domain to analyze
            days: Number of days to analyze
            model_id: Optional specific model to analyze
            
        Returns:
            Feedback analysis results
        """
        # Get feedback within time period
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        relevant_feedback = []
        
        for feedback in self.feedback_entries.get(domain, []):
            feedback_date = datetime.fromisoformat(feedback.timestamp.replace('Z', '+00:00'))
            if feedback_date >= cutoff_date:
                if model_id is None or feedback.model_id == model_id:
                    relevant_feedback.append(feedback)
        
        if not relevant_feedback:
            return FeedbackAnalysis(
                domain=domain,
                analysis_period=f"{days} days",
                total_feedback=0,
                avg_rating=0.0,
                feedback_distribution={},
                top_issues=[],
                improvement_suggestions=[],
                performance_trends={},
                critical_areas=[],
                confidence_score=0.0
            )
        
        # Analyze feedback
        analysis = self._perform_feedback_analysis(relevant_feedback, domain, days)
        
        # Cache analysis
        cache_key = f"{domain}_{days}_{model_id or 'all'}"
        self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    def get_improvement_recommendations(self, domain: str, 
                                      analysis: FeedbackAnalysis = None) -> List[Dict[str, Any]]:
        """
        Generate specific improvement recommendations based on feedback analysis.
        
        Args:
            domain: Domain name
            analysis: Optional pre-computed analysis
            
        Returns:
            List of improvement recommendations
        """
        if analysis is None:
            analysis = self.analyze_feedback(domain)
        
        recommendations = []
        
        # Rating-based recommendations
        if analysis.avg_rating < 3.5:
            recommendations.append({
                "type": "training_data",
                "priority": "high",
                "description": "Overall rating is low. Consider expanding training data with high-quality examples.",
                "action": "increase_training_examples",
                "target_improvement": "overall_quality"
            })
        
        # Issue-based recommendations
        for issue in analysis.top_issues[:3]:
            if issue["frequency"] > 5:  # Frequent issues
                recommendations.append({
                    "type": "targeted_training",
                    "priority": "medium",
                    "description": f"Address frequent issue: {issue['description']}",
                    "action": "add_specific_examples",
                    "keywords": issue["keywords"],
                    "target_improvement": issue["category"]
                })
        
        # Critical area recommendations
        for area in analysis.critical_areas:
            recommendations.append({
                "type": "knowledge_base",
                "priority": "high",
                "description": f"Critical knowledge gap identified in: {area}",
                "action": "expand_knowledge_base",
                "focus_area": area,
                "target_improvement": "domain_coverage"
            })
        
        # Trend-based recommendations
        if "accuracy_trend" in analysis.performance_trends:
            if analysis.performance_trends["accuracy_trend"] < -0.1:  # Declining accuracy
                recommendations.append({
                    "type": "model_refresh",
                    "priority": "high",
                    "description": "Model accuracy is declining. Consider retraining with recent data.",
                    "action": "retrain_model",
                    "target_improvement": "accuracy_recovery"
                })
        
        return recommendations
    
    def get_adaptive_training_data(self, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Generate adaptive training examples based on feedback analysis.
        
        Args:
            domain: Domain name
            limit: Maximum number of examples to generate
            
        Returns:
            List of training examples
        """
        feedback_list = self.feedback_entries.get(domain, [])
        if not feedback_list:
            return []
        
        training_examples = []
        
        # Process corrections and suggestions
        for feedback in feedback_list:
            if feedback.correction and feedback.user_input:
                training_examples.append({
                    "messages": [
                        {"role": "user", "content": feedback.user_input},
                        {"role": "assistant", "content": feedback.correction}
                    ],
                    "feedback_id": feedback.id,
                    "quality_score": self._calculate_example_quality(feedback),
                    "keywords": feedback.keywords,
                    "concepts": feedback.concepts,
                    "improvement_type": "correction"
                })
            
            elif feedback.rating and feedback.rating >= 4.0 and feedback.model_output:
                # High-quality examples
                training_examples.append({
                    "messages": [
                        {"role": "user", "content": feedback.user_input},
                        {"role": "assistant", "content": feedback.model_output}
                    ],
                    "feedback_id": feedback.id,
                    "quality_score": feedback.rating,
                    "keywords": feedback.keywords,
                    "concepts": feedback.concepts,
                    "improvement_type": "reinforcement"
                })
        
        # Sort by quality and return top examples
        training_examples.sort(key=lambda x: x["quality_score"], reverse=True)
        return training_examples[:limit]
    
    def track_model_performance(self, domain: str, model_id: str) -> ModelPerformanceMetrics:
        """
        Track performance metrics for a specific model.
        
        Args:
            domain: Domain name
            model_id: Model identifier
            
        Returns:
            Performance metrics
        """
        feedback_list = [f for f in self.feedback_entries.get(domain, []) if f.model_id == model_id]
        
        if not feedback_list:
            return ModelPerformanceMetrics(
                model_id=model_id,
                domain=domain,
                accuracy_score=0.0,
                user_satisfaction=0.0,
                response_quality=0.0,
                consistency_score=0.0,
                relevance_score=0.0,
                feedback_volume=0,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        
        # Calculate metrics
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        user_satisfaction = statistics.mean(ratings) if ratings else 0.0
        
        # Quality assessment based on feedback types
        positive_feedback = len([f for f in feedback_list if f.feedback_type in [FeedbackType.PRAISE, FeedbackType.RATING] and (f.rating or 0) >= 4])
        total_feedback = len(feedback_list)
        response_quality = positive_feedback / total_feedback if total_feedback > 0 else 0.0
        
        # Accuracy based on corrections needed
        corrections = len([f for f in feedback_list if f.feedback_type == FeedbackType.CORRECTION])
        accuracy_score = max(0.0, 1.0 - (corrections / total_feedback)) if total_feedback > 0 else 0.0
        
        metrics = ModelPerformanceMetrics(
            model_id=model_id,
            domain=domain,
            accuracy_score=accuracy_score,
            user_satisfaction=user_satisfaction / 5.0,  # Normalize to 0-1
            response_quality=response_quality,
            consistency_score=self._calculate_consistency_score(feedback_list),
            relevance_score=self._calculate_relevance_score(feedback_list),
            feedback_volume=total_feedback,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        self.model_metrics[f"{domain}_{model_id}"] = metrics
        return metrics
    
    def get_real_time_alerts(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get real-time alerts for critical issues.
        
        Args:
            domain: Domain name
            
        Returns:
            List of alerts
        """
        alerts = []
        recent_feedback = self._get_recent_feedback(domain, hours=1)
        
        # Critical feedback alert
        critical_feedback = [f for f in recent_feedback if f.severity == FeedbackSeverity.CRITICAL]
        if critical_feedback:
            alerts.append({
                "type": "critical_feedback",
                "severity": "high",
                "count": len(critical_feedback),
                "description": f"{len(critical_feedback)} critical feedback entries in the last hour",
                "action_required": "immediate_review"
            })
        
        # Rating drop alert
        recent_ratings = [f.rating for f in recent_feedback if f.rating is not None]
        if recent_ratings:
            avg_recent_rating = statistics.mean(recent_ratings)
            if avg_recent_rating < 2.0:
                alerts.append({
                    "type": "rating_drop",
                    "severity": "medium",
                    "value": avg_recent_rating,
                    "description": f"Average rating dropped to {avg_recent_rating:.1f}",
                    "action_required": "investigate_issues"
                })
        
        # High correction volume alert
        corrections = [f for f in recent_feedback if f.feedback_type == FeedbackType.CORRECTION]
        if len(corrections) > 10:  # Threshold for alerts
            alerts.append({
                "type": "high_correction_volume",
                "severity": "medium",
                "count": len(corrections),
                "description": f"{len(corrections)} corrections needed in the last hour",
                "action_required": "review_model_performance"
            })
        
        return alerts
    
    def _determine_severity(self, rating: Optional[float], feedback_type: FeedbackType, 
                          correction: Optional[str]) -> FeedbackSeverity:
        """Determine severity of feedback."""
        if rating is not None and rating <= 1.0:
            return FeedbackSeverity.CRITICAL
        elif feedback_type == FeedbackType.COMPLAINT:
            return FeedbackSeverity.HIGH
        elif correction is not None:
            return FeedbackSeverity.MEDIUM
        elif rating is not None and rating <= 2.0:
            return FeedbackSeverity.HIGH
        else:
            return FeedbackSeverity.LOW
    
    def _extract_feedback_features(self, user_input: str, model_output: str, 
                                  user_comment: str = None, correction: str = None) -> Tuple[List[str], List[str]]:
        """Extract keywords and concepts from feedback."""
        text_content = " ".join(filter(None, [user_input, model_output, user_comment, correction]))
        
        # Simple keyword extraction (in production, use more sophisticated NLP)
        words = text_content.lower().split()
        keywords = [word for word in words if len(word) > 4 and word.isalpha()][:10]
        
        # Simple concept extraction (would be enhanced with domain knowledge)
        concepts = []
        domain_terms = {
            "technical": ["api", "database", "algorithm", "architecture", "performance"],
            "medical": ["diagnosis", "treatment", "symptoms", "medication", "therapy"],
            "legal": ["contract", "compliance", "regulation", "liability", "jurisdiction"],
            "financial": ["investment", "portfolio", "risk", "return", "market"]
        }
        
        for concept_list in domain_terms.values():
            for concept in concept_list:
                if concept in text_content.lower():
                    concepts.append(concept)
        
        return keywords[:10], concepts[:5]
    
    def _process_critical_feedback(self, feedback: FeedbackEntry):
        """Process critical feedback immediately."""
        logger.warning(f"Critical feedback received: {feedback.id}")
        
        # In a production system, this might:
        # - Send alerts to the team
        # - Trigger immediate model review
        # - Update model weights temporarily
        # - Log to monitoring systems
        
        # For now, just mark as processed and log
        feedback.processed = True
        
        # Could trigger automated responses
        if feedback.rating and feedback.rating <= 1.0:
            logger.critical(f"Extremely low rating ({feedback.rating}) for model {feedback.model_id}")
    
    def _update_model_metrics(self, domain: str, model_id: str):
        """Update cached model metrics."""
        metrics = self.track_model_performance(domain, model_id)
        self.model_metrics[f"{domain}_{model_id}"] = metrics
    
    def _perform_feedback_analysis(self, feedback_list: List[FeedbackEntry], 
                                  domain: str, days: int) -> FeedbackAnalysis:
        """Perform comprehensive feedback analysis."""
        total_feedback = len(feedback_list)
        
        # Calculate average rating
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        avg_rating = statistics.mean(ratings) if ratings else 0.0
        
        # Feedback distribution
        type_counts = Counter([f.feedback_type.value for f in feedback_list])
        feedback_distribution = dict(type_counts)
        
        # Top issues analysis
        issue_keywords = []
        for feedback in feedback_list:
            if feedback.severity in [FeedbackSeverity.HIGH, FeedbackSeverity.CRITICAL]:
                issue_keywords.extend(feedback.keywords)
        
        issue_counts = Counter(issue_keywords)
        top_issues = [
            {
                "keyword": keyword,
                "frequency": count,
                "description": f"Issue related to {keyword}",
                "keywords": [keyword],
                "category": "quality"
            }
            for keyword, count in issue_counts.most_common(10)
        ]
        
        # Performance trends (simplified)
        performance_trends = {
            "rating_trend": self._calculate_trend([f.rating for f in feedback_list if f.rating]),
            "feedback_volume_trend": 0.0  # Would calculate based on historical data
        }
        
        # Critical areas
        critical_areas = [keyword for keyword, count in issue_counts.most_common(5)]
        
        # Confidence score
        confidence_score = min(total_feedback / 100, 1.0)  # Simple confidence based on volume
        
        return FeedbackAnalysis(
            domain=domain,
            analysis_period=f"{days} days",
            total_feedback=total_feedback,
            avg_rating=avg_rating,
            feedback_distribution=feedback_distribution,
            top_issues=top_issues,
            improvement_suggestions=self._generate_improvement_suggestions(feedback_list),
            performance_trends=performance_trends,
            critical_areas=critical_areas,
            confidence_score=confidence_score
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate simple trend for a list of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        x = list(range(len(values)))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except:
            return 0.0
    
    def _generate_improvement_suggestions(self, feedback_list: List[FeedbackEntry]) -> List[str]:
        """Generate improvement suggestions from feedback."""
        suggestions = []
        
        # Analyze common patterns
        low_ratings = [f for f in feedback_list if f.rating and f.rating <= 2.0]
        if len(low_ratings) > len(feedback_list) * 0.3:  # > 30% low ratings
            suggestions.append("Consider retraining the model with higher quality examples")
        
        corrections = [f for f in feedback_list if f.correction]
        if len(corrections) > 10:
            suggestions.append("Multiple corrections indicate need for targeted training data")
        
        # Domain-specific suggestions based on feedback patterns
        concepts = [concept for f in feedback_list for concept in f.concepts]
        if concepts:
            concept_counts = Counter(concepts)
            top_concept = concept_counts.most_common(1)[0][0]
            suggestions.append(f"Focus on improving responses related to {top_concept}")
        
        return suggestions[:5]
    
    def _calculate_example_quality(self, feedback: FeedbackEntry) -> float:
        """Calculate quality score for a training example."""
        score = 3.0  # Base score
        
        if feedback.rating:
            score = feedback.rating
        
        # Boost for corrections (assume they're high quality)
        if feedback.correction:
            score = min(5.0, score + 1.0)
        
        # Reduce for very short content
        if len(feedback.user_input) < 20:
            score = max(1.0, score - 0.5)
        
        return score
    
    def _calculate_consistency_score(self, feedback_list: List[FeedbackEntry]) -> float:
        """Calculate consistency score based on feedback variance."""
        ratings = [f.rating for f in feedback_list if f.rating is not None]
        if len(ratings) < 2:
            return 0.5
        
        # Lower variance = higher consistency
        variance = statistics.variance(ratings)
        consistency = max(0.0, 1.0 - (variance / 4.0))  # Normalize variance to 0-1
        return consistency
    
    def _calculate_relevance_score(self, feedback_list: List[FeedbackEntry]) -> float:
        """Calculate relevance score based on feedback content."""
        # Simplified relevance calculation
        relevant_feedback = [f for f in feedback_list if f.keywords or f.concepts]
        if not feedback_list:
            return 0.0
        
        return len(relevant_feedback) / len(feedback_list)
    
    def _get_recent_feedback(self, domain: str, hours: int = 24) -> List[FeedbackEntry]:
        """Get feedback from recent hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_feedback = []
        for feedback in self.feedback_entries.get(domain, []):
            feedback_time = datetime.fromisoformat(feedback.timestamp.replace('Z', '+00:00'))
            if feedback_time >= cutoff_time:
                recent_feedback.append(feedback)
        
        return recent_feedback
    
    def _load_feedback_data(self):
        """Load feedback data from persistent storage."""
        import os
        feedback_file = os.path.join(self.storage_path, "feedback.json")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for domain, entries_data in data.items():
                    entries = []
                    for entry_data in entries_data:
                        # Convert string enums back to enum objects
                        entry_data["feedback_type"] = FeedbackType(entry_data["feedback_type"])
                        entry_data["severity"] = FeedbackSeverity(entry_data["severity"])
                        entries.append(FeedbackEntry(**entry_data))
                    
                    self.feedback_entries[domain] = entries
                
                logger.info(f"Loaded feedback data for {len(data)} domains")
            except Exception as e:
                logger.error(f"Failed to load feedback data: {str(e)}")
    
    def _save_feedback_data(self, domain: str):
        """Save feedback data to persistent storage."""
        import os
        feedback_file = os.path.join(self.storage_path, "feedback.json")
        
        # Load existing data
        all_data = {}
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
            except:
                pass
        
        # Update domain data
        if domain in self.feedback_entries:
            domain_data = []
            for entry in self.feedback_entries[domain]:
                entry_dict = asdict(entry)
                # Convert enums to strings for JSON serialization
                entry_dict["feedback_type"] = entry.feedback_type.value
                entry_dict["severity"] = entry.severity.value
                domain_data.append(entry_dict)
            
            all_data[domain] = domain_data
        
        # Save updated data
        try:
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save feedback data: {str(e)}")
