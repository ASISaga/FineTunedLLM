"""
Multi-Domain LLM Manager
Orchestrates multiple domain-specific fine-tuned LLMs with adaptive learning and feedback integration.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import openai
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..domain_knowledge_system.DomainContextManager import DomainContextManager, DomainContext
from ..domain_knowledge_system.EnhancedDomainContextManager import EnhancedDomainContextManager
from ..domain_knowledge_system.DomainKnowledgeBase import DomainKnowledgeBase
from ..feedback_integration.FeedbackCollector import FeedbackCollector
from ..feedback_integration.ContinuousLearningPipeline import ContinuousLearningPipeline

logger = logging.getLogger(__name__)

@dataclass
class DomainLLMConfig:
    """Configuration for a domain-specific LLM"""
    domain_name: str
    model_id: Optional[str] = None
    base_model: str = "gpt-4-turbo-2024-04-09"
    knowledge_base_path: str = ""
    context_paragraph: str = ""
    performance_threshold: float = 0.85
    retraining_interval_days: int = 30
    last_training_date: Optional[datetime] = None
    deployment_status: str = "pending"  # pending, training, deployed, failed
    feedback_count: int = 0
    average_rating: float = 0.0
    version: str = "1.0"

@dataclass
class FeedbackEntry:
    """Represents user feedback for a domain LLM"""
    domain_name: str
    model_id: str
    user_query: str
    model_response: str
    user_rating: float  # 1-5 scale
    user_feedback: str
    expected_response: Optional[str] = None
    timestamp: datetime = None
    processed: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class MultiDomainLLMManager:
    """
    Manages multiple domain-specific fine-tuned LLMs with adaptive learning capabilities.
    
    Features:
    - Domain-specific LLM creation and management
    - Adaptive summarization with domain context
    - Iterative fine-tuning based on performance
    - Real-world feedback integration
    - Continuous learning pipeline
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 azure_openai_config: Dict[str, str],
                 aws_bedrock_config: Dict[str, str],
                 base_path: str = "./multi_domain_llms"):
        """
        Initialize the Multi-Domain LLM Manager.
        
        Args:
            azure_openai_config: Azure OpenAI configuration
            aws_bedrock_config: AWS Bedrock configuration
            base_path: Base directory for storing domain LLM data
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize clients
        self.openai_client = openai.OpenAI(
            api_key=azure_openai_config["api_key"],
            api_base=azure_openai_config.get("api_base", ""),
            organization=azure_openai_config.get("organization", "")
        )
        
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=aws_bedrock_config["aws_region"],
            aws_access_key_id=aws_bedrock_config["aws_access_key_id"],
            aws_secret_access_key=aws_bedrock_config["aws_secret_access_key"]
        )
        
        # Initialize core components
        self.enhanced_context_manager = EnhancedDomainContextManager()
        self.knowledge_base = DomainKnowledgeBase()
        self.feedback_collector = FeedbackCollector()
        self.continuous_learning = ContinuousLearningPipeline()
        
        # Domain LLM configurations and state
        self.domain_configs: Dict[str, DomainLLMConfig] = {}
        self.active_models: Dict[str, str] = {}  # domain -> model_id
        self.feedback_queue: List[FeedbackEntry] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Load existing configurations
        self._load_domain_configurations()
        
        logger.info("Initialized MultiDomainLLMManager")

    def create_domain_llm(self, 
                         domain_name: str,
                         context_paragraph: str,
                         knowledge_sources: List[str],
                         training_documents: List[str],
                         config_overrides: Dict[str, Any] = None) -> str:
        """
        Create a new domain-specific LLM.
        
        Args:
            domain_name: Name of the domain
            context_paragraph: Domain-specific context paragraph
            knowledge_sources: List of knowledge base sources
            training_documents: Documents for initial training
            config_overrides: Optional configuration overrides
            
        Returns:
            Job ID for the training process
        """
        logger.info(f"Creating domain-specific LLM for: {domain_name}")
        
        # Create domain configuration
        domain_config = DomainLLMConfig(
            domain_name=domain_name,
            context_paragraph=context_paragraph,
            knowledge_base_path=str(self.base_path / f"{domain_name}_knowledge_base.json"),
            **(config_overrides or {})
        )
        
        # Initialize domain context in enhanced context manager
        success = self.enhanced_context_manager.create_custom_domain_context(
            domain_name=domain_name,
            description=context_paragraph,
            key_concepts=[],  # Will be extracted from knowledge sources
            terminology={},   # Will be extracted from knowledge sources
            focus_areas=[],   # Will be extracted from knowledge sources
            knowledge_sources=knowledge_sources
        )
        
        if not success:
            raise ValueError(f"Failed to create domain context for {domain_name}")
        
        # Initialize knowledge base for the domain
        self.knowledge_base.initialize_domain_knowledge(
            domain=domain_name,
            knowledge_sources=knowledge_sources,
            context_paragraph=context_paragraph
        )
        
        # Start initial training
        job_id = self._start_domain_training(domain_config, training_documents)
        
        # Save configuration
        self.domain_configs[domain_name] = domain_config
        self._save_domain_configurations()
        
        logger.info(f"Started training for domain {domain_name}, job ID: {job_id}")
        return job_id

    def get_domain_response(self, 
                           domain_name: str, 
                           query: str,
                           include_context: bool = True) -> Dict[str, Any]:
        """
        Get a response from a domain-specific LLM.
        
        Args:
            domain_name: Target domain
            query: User query
            include_context: Whether to include domain context
            
        Returns:
            Response with metadata
        """
        if domain_name not in self.active_models:
            raise ValueError(f"No active model found for domain: {domain_name}")
        
        model_id = self.active_models[domain_name]
        domain_config = self.domain_configs[domain_name]
        
        # Get enhanced domain context
        enhanced_context = self.enhanced_context_manager.get_enhanced_domain_context(
            domain=domain_name,
            query=query,
            context_type="inference"
        )
        
        # Prepare system message with domain context
        system_message = enhanced_context.get("system_prompt", "You are a helpful assistant.")
        if include_context and domain_config.context_paragraph:
            system_message += f"\n\nDomain Context: {domain_config.context_paragraph}"
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            result = {
                "domain": domain_name,
                "model_id": model_id,
                "query": query,
                "response": response.choices[0].message.content,
                "usage": response.usage.__dict__ if response.usage else None,
                "timestamp": datetime.utcnow().isoformat(),
                "version": domain_config.version
            }
            
            # Track usage for performance monitoring
            self._track_model_usage(domain_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting response from domain {domain_name}: {str(e)}")
            raise

    def submit_feedback(self, 
                       domain_name: str,
                       user_query: str,
                       model_response: str,
                       rating: float,
                       feedback_text: str,
                       expected_response: Optional[str] = None) -> bool:
        """
        Submit feedback for a domain LLM response.
        
        Args:
            domain_name: Target domain
            user_query: Original user query
            model_response: Model's response
            rating: User rating (1-5 scale)
            feedback_text: User feedback text
            expected_response: Optional expected response
            
        Returns:
            Success status
        """
        if domain_name not in self.active_models:
            logger.warning(f"Feedback submitted for inactive domain: {domain_name}")
            return False
        
        model_id = self.active_models[domain_name]
        
        feedback_entry = FeedbackEntry(
            domain_name=domain_name,
            model_id=model_id,
            user_query=user_query,
            model_response=model_response,
            user_rating=rating,
            user_feedback=feedback_text,
            expected_response=expected_response
        )
        
        self.feedback_queue.append(feedback_entry)
        
        # Update domain configuration with feedback stats
        domain_config = self.domain_configs[domain_name]
        domain_config.feedback_count += 1
        
        # Update running average rating
        total_rating = (domain_config.average_rating * (domain_config.feedback_count - 1) + rating)
        domain_config.average_rating = total_rating / domain_config.feedback_count
        
        # Save updated configuration
        self._save_domain_configurations()
        
        # Process feedback if queue is large enough
        if len(self.feedback_queue) >= 10:
            asyncio.create_task(self._process_feedback_batch())
        
        logger.info(f"Feedback submitted for domain {domain_name}, rating: {rating}")
        return True

    async def process_continuous_learning(self) -> Dict[str, Any]:
        """
        Process continuous learning for all domains.
        
        Returns:
            Processing results summary
        """
        results = {
            "processed_domains": [],
            "retraining_triggered": [],
            "errors": []
        }
        
        for domain_name, config in self.domain_configs.items():
            try:
                # Check if retraining is needed
                needs_retraining = self._should_retrain_domain(domain_name)
                
                if needs_retraining:
                    logger.info(f"Triggering retraining for domain: {domain_name}")
                    
                    # Collect recent feedback for retraining
                    domain_feedback = [f for f in self.feedback_queue 
                                     if f.domain_name == domain_name and not f.processed]
                    
                    # Generate new training data from feedback
                    new_training_data = await self._generate_training_from_feedback(
                        domain_name, domain_feedback
                    )
                    
                    # Start retraining
                    job_id = await self._start_adaptive_retraining(
                        domain_name, new_training_data
                    )
                    
                    results["retraining_triggered"].append({
                        "domain": domain_name,
                        "job_id": job_id,
                        "feedback_count": len(domain_feedback)
                    })
                    
                    # Mark feedback as processed
                    for feedback in domain_feedback:
                        feedback.processed = True
                
                results["processed_domains"].append(domain_name)
                
            except Exception as e:
                error_msg = f"Error processing domain {domain_name}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results

    def get_domain_performance_metrics(self, domain_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific domain."""
        if domain_name not in self.performance_metrics:
            return {}
        
        domain_config = self.domain_configs.get(domain_name, {})
        metrics = self.performance_metrics[domain_name].copy()
        
        metrics.update({
            "average_rating": getattr(domain_config, 'average_rating', 0.0),
            "feedback_count": getattr(domain_config, 'feedback_count', 0),
            "version": getattr(domain_config, 'version', "1.0"),
            "deployment_status": getattr(domain_config, 'deployment_status', "unknown"),
            "last_training_date": getattr(domain_config, 'last_training_date', None)
        })
        
        return metrics

    def list_active_domains(self) -> List[Dict[str, Any]]:
        """List all active domain LLMs with their status."""
        domains = []
        
        for domain_name, config in self.domain_configs.items():
            domain_info = {
                "domain_name": domain_name,
                "model_id": self.active_models.get(domain_name),
                "deployment_status": config.deployment_status,
                "version": config.version,
                "average_rating": config.average_rating,
                "feedback_count": config.feedback_count,
                "last_training_date": config.last_training_date.isoformat() if config.last_training_date else None
            }
            domains.append(domain_info)
        
        return domains

    # Private helper methods
    
    def _start_domain_training(self, 
                              domain_config: DomainLLMConfig, 
                              training_documents: List[str]) -> str:
        """Start training for a domain-specific LLM."""
        # This would integrate with the existing DomainAwareTrainerBedrock
        # For now, return a mock job ID
        job_id = f"domain_training_{domain_config.domain_name}_{int(datetime.utcnow().timestamp())}"
        
        # Update configuration
        domain_config.deployment_status = "training"
        domain_config.last_training_date = datetime.utcnow()
        
        return job_id

    def _should_retrain_domain(self, domain_name: str) -> bool:
        """Determine if a domain should be retrained."""
        config = self.domain_configs.get(domain_name)
        if not config:
            return False
        
        # Check performance threshold
        if config.average_rating < config.performance_threshold:
            return True
        
        # Check time-based retraining
        if config.last_training_date:
            days_since_training = (datetime.utcnow() - config.last_training_date).days
            if days_since_training >= config.retraining_interval_days:
                return True
        
        # Check feedback volume
        domain_feedback_count = sum(1 for f in self.feedback_queue 
                                  if f.domain_name == domain_name and not f.processed)
        if domain_feedback_count >= 20:  # Threshold for feedback-driven retraining
            return True
        
        return False

    async def _generate_training_from_feedback(self, 
                                             domain_name: str, 
                                             feedback_list: List[FeedbackEntry]) -> List[Dict]:
        """Generate training data from user feedback."""
        training_data = []
        
        for feedback in feedback_list:
            if feedback.user_rating < 3.0 and feedback.expected_response:
                # Use expected response for low-rated interactions
                training_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert in {domain_name}. {self.domain_configs[domain_name].context_paragraph}"
                        },
                        {
                            "role": "user",
                            "content": feedback.user_query
                        },
                        {
                            "role": "assistant",
                            "content": feedback.expected_response
                        }
                    ]
                }
                training_data.append(training_example)
        
        return training_data

    async def _start_adaptive_retraining(self, 
                                       domain_name: str, 
                                       additional_training_data: List[Dict]) -> str:
        """Start adaptive retraining for a domain."""
        # This would integrate with the continuous learning pipeline
        job_id = f"adaptive_retraining_{domain_name}_{int(datetime.utcnow().timestamp())}"
        
        # Update domain configuration
        config = self.domain_configs[domain_name]
        config.deployment_status = "training"
        config.last_training_date = datetime.utcnow()
        
        # Increment version
        version_parts = config.version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        config.version = '.'.join(version_parts)
        
        self._save_domain_configurations()
        
        return job_id

    async def _process_feedback_batch(self):
        """Process a batch of feedback entries."""
        try:
            # Group feedback by domain
            domain_feedback = {}
            for feedback in self.feedback_queue:
                if not feedback.processed:
                    if feedback.domain_name not in domain_feedback:
                        domain_feedback[feedback.domain_name] = []
                    domain_feedback[feedback.domain_name].append(feedback)
            
            # Process each domain's feedback
            for domain_name, feedback_list in domain_feedback.items():
                await self._analyze_domain_feedback(domain_name, feedback_list)
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {str(e)}")

    async def _analyze_domain_feedback(self, domain_name: str, feedback_list: List[FeedbackEntry]):
        """Analyze feedback for a specific domain."""
        if not feedback_list:
            return
        
        # Calculate metrics
        avg_rating = sum(f.user_rating for f in feedback_list) / len(feedback_list)
        low_rating_count = sum(1 for f in feedback_list if f.user_rating < 3.0)
        
        # Update performance metrics
        if domain_name not in self.performance_metrics:
            self.performance_metrics[domain_name] = {}
        
        self.performance_metrics[domain_name].update({
            "recent_avg_rating": avg_rating,
            "low_rating_percentage": low_rating_count / len(feedback_list),
            "feedback_analysis_date": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Analyzed {len(feedback_list)} feedback entries for domain {domain_name}")

    def _track_model_usage(self, domain_name: str, result: Dict[str, Any]):
        """Track model usage for performance monitoring."""
        if domain_name not in self.performance_metrics:
            self.performance_metrics[domain_name] = {
                "total_requests": 0,
                "total_tokens": 0,
                "avg_response_time": 0.0
            }
        
        metrics = self.performance_metrics[domain_name]
        metrics["total_requests"] += 1
        
        if result.get("usage"):
            metrics["total_tokens"] += result["usage"].get("total_tokens", 0)

    def _load_domain_configurations(self):
        """Load domain configurations from disk."""
        config_file = self.base_path / "domain_configurations.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                for domain_name, config_data in data.items():
                    # Convert datetime strings back to datetime objects
                    if config_data.get("last_training_date"):
                        config_data["last_training_date"] = datetime.fromisoformat(
                            config_data["last_training_date"]
                        )
                    
                    self.domain_configs[domain_name] = DomainLLMConfig(**config_data)
                
                logger.info(f"Loaded {len(self.domain_configs)} domain configurations")
                
            except Exception as e:
                logger.error(f"Error loading domain configurations: {str(e)}")

    def _save_domain_configurations(self):
        """Save domain configurations to disk."""
        config_file = self.base_path / "domain_configurations.json"
        
        try:
            data = {}
            for domain_name, config in self.domain_configs.items():
                config_dict = asdict(config)
                
                # Convert datetime objects to strings
                if config_dict.get("last_training_date"):
                    config_dict["last_training_date"] = config.last_training_date.isoformat()
                
                data[domain_name] = config_dict
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving domain configurations: {str(e)}")
