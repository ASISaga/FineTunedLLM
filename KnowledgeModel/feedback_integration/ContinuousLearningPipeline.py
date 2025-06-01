"""
Continuous Learning Pipeline for FineTunedLLM System
Orchestrates iterative model improvement through feedback analysis and adaptive training.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import uuid
from enum import Enum

from ..domain_knowledge_system.DomainKnowledgeBase import DomainKnowledgeBase, DomainKnowledgeEntry
from .FeedbackCollector import FeedbackCollector, FeedbackAnalysis, FeedbackType
from ..domain_knowledge_system.DomainContextManager import DomainContextManager, DomainContext
from ..finetuning_pipeline.DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

logger = logging.getLogger(__name__)

class LearningTrigger(Enum):
    """Types of triggers for continuous learning"""
    PERFORMANCE_DECLINE = "performance_decline"
    FEEDBACK_THRESHOLD = "feedback_threshold"
    SCHEDULED_INTERVAL = "scheduled_interval"
    MANUAL_REQUEST = "manual_request"
    NEW_KNOWLEDGE = "new_knowledge"

class LearningPhase(Enum):
    """Phases of the continuous learning process"""
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    DATA_GENERATION = "data_generation"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class LearningCycle:
    """Represents a complete continuous learning cycle"""
    id: str
    domain: str
    trigger: LearningTrigger
    phase: LearningPhase
    model_id: str
    baseline_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    training_data_count: int
    feedback_count: int
    improvements_identified: List[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class LearningConfiguration:
    """Configuration for continuous learning behavior"""
    min_feedback_count: int = 50
    performance_decline_threshold: float = 0.1  # 10% decline
    feedback_score_threshold: float = 3.0  # Below this triggers learning
    max_parallel_cycles: int = 3
    training_data_ratio: float = 0.8  # Training vs validation split
    validation_improvement_threshold: float = 0.05  # 5% improvement required
    learning_interval_hours: int = 24  # Automatic learning frequency
    knowledge_freshness_days: int = 7  # How recent knowledge should be
    adaptive_batch_size: bool = True
    domain_specific_thresholds: Dict[str, Dict[str, float]] = None

class ContinuousLearningPipeline:
    """
    Orchestrates continuous learning and model improvement.
    
    Features:
    - Automatic performance monitoring
    - Feedback-driven learning triggers
    - Adaptive training data generation
    - Model versioning and rollback
    - Cross-domain learning coordination
    - Performance trend analysis
    """
    
    def __init__(self, 
                 knowledge_base: DomainKnowledgeBase,
                 feedback_collector: FeedbackCollector,
                 domain_manager: DomainContextManager,
                 trainer: DomainAwareTrainer,
                 storage_path: str = "./continuous_learning"):
        self.knowledge_base = knowledge_base
        self.feedback_collector = feedback_collector
        self.domain_manager = domain_manager
        self.trainer = trainer
        
        # Storage setup
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Active learning cycles
        self.active_cycles: Dict[str, LearningCycle] = {}
        self.cycle_history: List[LearningCycle] = []
        
        # Configuration
        self.config = LearningConfiguration()
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = {}
        self.learning_metrics: Dict[str, Dict] = {}
        
        # Load existing state
        self._load_learning_state()
        
        logger.info("Continuous Learning Pipeline initialized")
    
    def start_learning_cycle(self, domain: str, model_id: str, 
                           trigger: LearningTrigger,
                           target_improvements: List[str] = None) -> str:
        """
        Start a new continuous learning cycle.
        
        Args:
            domain: Domain name
            model_id: Current model identifier
            trigger: What triggered this learning cycle
            target_improvements: Specific areas to improve
            
        Returns:
            Learning cycle ID
        """
        # Check if domain is already in active learning
        active_domain_cycles = [
            cycle for cycle in self.active_cycles.values() 
            if cycle.domain == domain and cycle.phase != LearningPhase.COMPLETE
        ]
        
        if len(active_domain_cycles) >= self.config.max_parallel_cycles:
            logger.warning(f"Domain {domain} already has maximum parallel learning cycles")
            return None
        
        # Get baseline metrics
        baseline_metrics = self._get_current_performance(domain, model_id)
        
        # Create new learning cycle
        cycle_id = str(uuid.uuid4())
        cycle = LearningCycle(
            id=cycle_id,
            domain=domain,
            trigger=trigger,
            phase=LearningPhase.MONITORING,
            model_id=model_id,
            baseline_metrics=baseline_metrics,
            target_metrics={},
            current_metrics=baseline_metrics.copy(),
            training_data_count=0,
            feedback_count=0,
            improvements_identified=target_improvements or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            error_message=None,
            metadata={"trigger_details": str(trigger)}
        )
        
        self.active_cycles[cycle_id] = cycle
        self._save_learning_state()
        
        logger.info(f"Started learning cycle {cycle_id} for domain {domain}")
        
        # Start asynchronous processing
        asyncio.create_task(self._process_learning_cycle(cycle_id))
        
        return cycle_id
    
    async def _process_learning_cycle(self, cycle_id: str):
        """Process a complete learning cycle asynchronously."""
        try:
            cycle = self.active_cycles[cycle_id]
            
            # Phase 1: Analysis
            cycle.phase = LearningPhase.ANALYSIS
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
            await self._analyze_learning_needs(cycle)
            
            # Phase 2: Data Generation
            cycle.phase = LearningPhase.DATA_GENERATION
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
            training_data = await self._generate_adaptive_training_data(cycle)
            
            if not training_data:
                cycle.phase = LearningPhase.FAILED
                cycle.error_message = "No adaptive training data generated"
                return
            
            # Phase 3: Training
            cycle.phase = LearningPhase.TRAINING
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
            new_model_id = await self._train_improved_model(cycle, training_data)
            
            if not new_model_id:
                cycle.phase = LearningPhase.FAILED
                cycle.error_message = "Model training failed"
                return
            
            # Phase 4: Validation
            cycle.phase = LearningPhase.VALIDATION
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
            validation_results = await self._validate_improved_model(cycle, new_model_id)
            
            if not validation_results["improved"]:
                cycle.phase = LearningPhase.FAILED
                cycle.error_message = "Model did not show sufficient improvement"
                return
            
            # Phase 5: Deployment
            cycle.phase = LearningPhase.DEPLOYMENT
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
            await self._deploy_improved_model(cycle, new_model_id)
            
            # Complete
            cycle.phase = LearningPhase.COMPLETE
            cycle.completed_at = datetime.now(timezone.utc).isoformat()
            cycle.updated_at = cycle.completed_at
            
            # Update metrics and history
            self._update_learning_metrics(cycle)
            self.cycle_history.append(cycle)
            
            logger.info(f"Completed learning cycle {cycle_id}")
            
        except Exception as e:
            logger.error(f"Error in learning cycle {cycle_id}: {str(e)}")
            cycle = self.active_cycles[cycle_id]
            cycle.phase = LearningPhase.FAILED
            cycle.error_message = str(e)
            cycle.updated_at = datetime.now(timezone.utc).isoformat()
        
        finally:
            # Clean up active cycle
            if cycle_id in self.active_cycles:
                del self.active_cycles[cycle_id]
            self._save_learning_state()
    
    async def _analyze_learning_needs(self, cycle: LearningCycle):
        """Analyze what needs to be learned for this cycle."""
        domain = cycle.domain
        
        # Get recent feedback analysis
        feedback_analysis = self.feedback_collector.analyze_feedback(
            domain, days=30, model_id=cycle.model_id
        )
        
        cycle.feedback_count = feedback_analysis.total_feedback
        
        # Get improvement recommendations
        recommendations = self.feedback_collector.get_improvement_recommendations(
            domain, feedback_analysis
        )
        
        # Extract target improvements
        improvement_areas = []
        target_metrics = {}
        
        for rec in recommendations:
            improvement_areas.append(rec["description"])
            if "target_improvement" in rec:
                target_metrics[rec["target_improvement"]] = rec.get("target_value", 0.1)
        
        cycle.improvements_identified.extend(improvement_areas)
        cycle.target_metrics = target_metrics
        
        logger.info(f"Analysis complete for cycle {cycle.id}: {len(improvement_areas)} improvements identified")
    
    async def _generate_adaptive_training_data(self, cycle: LearningCycle) -> List[Dict[str, Any]]:
        """Generate training data adapted to feedback and knowledge base."""
        domain = cycle.domain
        
        # Get adaptive training data from feedback
        feedback_training_data = self.feedback_collector.get_adaptive_training_data(
            domain, limit=200
        )
        
        # Get enhanced context from knowledge base
        recent_feedback = self.feedback_collector.get_recent_feedback(domain, days=7)
        performance_metrics = cycle.baseline_metrics
        
        adaptive_context = self.knowledge_base.get_adaptive_context(
            domain, recent_feedback, performance_metrics
        )
        
        # Generate additional training examples based on knowledge gaps
        knowledge_training_data = await self._generate_knowledge_based_training(
            domain, adaptive_context, cycle.improvements_identified
        )
        
        # Combine and deduplicate training data
        all_training_data = feedback_training_data + knowledge_training_data
        
        # Remove duplicates and sort by quality
        unique_training_data = self._deduplicate_training_data(all_training_data)
        
        cycle.training_data_count = len(unique_training_data)
        
        logger.info(f"Generated {len(unique_training_data)} adaptive training examples")
        
        return unique_training_data
    
    async def _generate_knowledge_based_training(self, domain: str, 
                                               adaptive_context: Dict[str, Any],
                                               improvements: List[str]) -> List[Dict[str, Any]]:
        """Generate training data based on knowledge base and identified improvements."""
        training_examples = []
        
        # Get domain context
        domain_ctx = self.domain_manager.get_domain_context(domain)
        if not domain_ctx:
            return training_examples
        
        # Use Claude Sonnet 4 to generate training examples
        try:
            # Create focused training prompts based on improvements
            for improvement in improvements[:5]:  # Limit to top 5 improvements
                
                # Get relevant knowledge entries
                relevant_entries = adaptive_context.get("entries", [])[:10]
                
                if relevant_entries:
                    # Generate training examples using the trainer's Bedrock integration
                    knowledge_content = "\n\n".join([
                        entry.get("content", "") for entry in relevant_entries
                    ])
                    
                    # Create domain-specific training data
                    domain_training_data = await self._create_domain_training_examples(
                        domain, knowledge_content, improvement, domain_ctx
                    )
                    
                    training_examples.extend(domain_training_data)
        
        except Exception as e:
            logger.error(f"Error generating knowledge-based training data: {str(e)}")
        
        return training_examples
    
    async def _create_domain_training_examples(self, domain: str, content: str, 
                                             improvement_focus: str,
                                             domain_ctx: DomainContext) -> List[Dict[str, Any]]:
        """Create domain-specific training examples using Bedrock."""
        examples = []
        
        try:
            # Use the existing trainer's Bedrock integration
            bedrock_prompt = f"""
            Based on the following domain knowledge, create 5 high-quality training examples 
            that focus on improving: {improvement_focus}
            
            Domain: {domain_ctx.domain_name}
            Focus Areas: {', '.join(domain_ctx.focus_areas)}
            
            Knowledge Content:
            {content[:2000]}  # Limit content length
            
            Generate examples in this format:
            Question: [specific question about the domain]
            Answer: [detailed, expert-level answer using domain terminology]
            
            Focus on practical, real-world scenarios that address: {improvement_focus}
            """
            
            # Here we would call Bedrock, but for now create structured examples
            # In the actual implementation, this would use the trainer's Bedrock client
            
            # Generate structured training examples
            for i in range(3):  # Generate 3 examples per improvement
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert in {domain_ctx.domain_name}. "
                                     f"Focus on {improvement_focus}. "
                                     f"{domain_ctx.fine_tuning_instructions}"
                        },
                        {
                            "role": "user",
                            "content": f"Please explain a key aspect of {improvement_focus} "
                                     f"in the context of {domain_ctx.domain_name}."
                        },
                        {
                            "role": "assistant",
                            "content": f"Based on the {domain_ctx.domain_name} domain knowledge, "
                                     f"here's an explanation of {improvement_focus}..."
                        }
                    ],
                    "improvement_focus": improvement_focus,
                    "domain": domain,
                    "source": "knowledge_base",
                    "quality_score": 0.8
                }
                examples.append(example)
        
        except Exception as e:
            logger.error(f"Error creating domain training examples: {str(e)}")
        
        return examples
    
    async def _train_improved_model(self, cycle: LearningCycle, 
                                  training_data: List[Dict[str, Any]]) -> Optional[str]:
        """Train an improved model using adaptive training data."""
        try:
            # Prepare training data in JSONL format
            training_file = self._prepare_training_file(training_data, cycle)
            
            # Configure training with adaptive parameters
            config = self._get_adaptive_training_config(cycle)
            
            # Start fine-tuning job
            job_id = self.trainer.start_domain_fine_tuning(config)
            
            # Monitor training progress
            await self._monitor_training_job(cycle, job_id)
            
            # Get the fine-tuned model ID
            job_status = self.trainer.get_job_status(job_id)
            
            if job_status.get("status") == "succeeded":
                new_model_id = job_status.get("fine_tuned_model")
                logger.info(f"Training completed successfully: {new_model_id}")
                return new_model_id
            else:
                logger.error(f"Training failed: {job_status.get('status')}")
                return None
        
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return None
    
    async def _validate_improved_model(self, cycle: LearningCycle, 
                                     new_model_id: str) -> Dict[str, Any]:
        """Validate that the improved model performs better."""
        try:
            # Generate evaluation prompts
            eval_prompts = self._generate_evaluation_prompts(cycle)
            
            # Test both old and new models
            old_results = self.trainer.test_fine_tuned_model(
                cycle.model_id, eval_prompts, cycle.domain
            )
            
            new_results = self.trainer.test_fine_tuned_model(
                new_model_id, eval_prompts, cycle.domain
            )
            
            # Compare performance
            improvement_metrics = self._compare_model_performance(
                old_results, new_results, cycle
            )
            
            # Check if improvement meets threshold
            improved = improvement_metrics["overall_improvement"] >= self.config.validation_improvement_threshold
            
            cycle.current_metrics = improvement_metrics
            
            return {
                "improved": improved,
                "metrics": improvement_metrics,
                "old_results": old_results,
                "new_results": new_results
            }
        
        except Exception as e:
            logger.error(f"Error in model validation: {str(e)}")
            return {"improved": False, "error": str(e)}
    
    async def _deploy_improved_model(self, cycle: LearningCycle, new_model_id: str):
        """Deploy the improved model and update tracking."""
        try:
            # Update domain model tracking
            self._update_domain_model_registry(cycle.domain, new_model_id, cycle)
            
            # Update knowledge base with new model performance
            self.knowledge_base.update_domain_metrics(
                cycle.domain, 
                {"active_model": new_model_id, "last_improvement": cycle.completed_at}
            )
            
            # Log deployment
            logger.info(f"Deployed improved model {new_model_id} for domain {cycle.domain}")
        
        except Exception as e:
            logger.error(f"Error deploying improved model: {str(e)}")
            raise
    
    def monitor_performance_triggers(self) -> List[str]:
        """Monitor for performance decline triggers across all domains."""
        triggered_cycles = []
        
        for domain in self.domain_manager.list_available_domains():
            # Check recent performance trends
            performance_trend = self._analyze_performance_trend(domain)
            
            if performance_trend["decline"] > self.config.performance_decline_threshold:
                # Trigger learning cycle
                model_id = self._get_current_model_id(domain)
                if model_id:
                    cycle_id = self.start_learning_cycle(
                        domain, model_id, LearningTrigger.PERFORMANCE_DECLINE
                    )
                    if cycle_id:
                        triggered_cycles.append(cycle_id)
        
        return triggered_cycles
    
    def monitor_feedback_triggers(self) -> List[str]:
        """Monitor for feedback-based learning triggers."""
        triggered_cycles = []
        
        for domain in self.domain_manager.list_available_domains():
            # Get recent feedback analysis
            feedback_analysis = self.feedback_collector.analyze_feedback(domain, days=7)
            
            # Check feedback thresholds
            if (feedback_analysis.average_rating < self.config.feedback_score_threshold and
                feedback_analysis.total_feedback >= self.config.min_feedback_count):
                
                model_id = self._get_current_model_id(domain)
                if model_id:
                    cycle_id = self.start_learning_cycle(
                        domain, model_id, LearningTrigger.FEEDBACK_THRESHOLD
                    )
                    if cycle_id:
                        triggered_cycles.append(cycle_id)
        
        return triggered_cycles
    
    def get_learning_status(self, domain: str = None) -> Dict[str, Any]:
        """Get current learning status for domains."""
        if domain:
            # Get status for specific domain
            domain_cycles = [
                cycle for cycle in self.active_cycles.values() 
                if cycle.domain == domain
            ]
            
            return {
                "domain": domain,
                "active_cycles": len(domain_cycles),
                "cycles": [asdict(cycle) for cycle in domain_cycles],
                "metrics": self.learning_metrics.get(domain, {}),
                "performance_history": self.performance_history.get(domain, [])
            }
        else:
            # Get overall status
            return {
                "total_active_cycles": len(self.active_cycles),
                "active_cycles_by_domain": {
                    domain: len([c for c in self.active_cycles.values() if c.domain == domain])
                    for domain in self.domain_manager.list_available_domains()
                },
                "completed_cycles": len(self.cycle_history),
                "learning_metrics": self.learning_metrics
            }
    
    def force_learning_cycle(self, domain: str, improvements: List[str] = None) -> str:
        """Manually force a learning cycle for a domain."""
        model_id = self._get_current_model_id(domain)
        if not model_id:
            raise ValueError(f"No active model found for domain {domain}")
        
        return self.start_learning_cycle(
            domain, model_id, LearningTrigger.MANUAL_REQUEST, improvements
        )
    
    # Helper methods
    def _get_current_performance(self, domain: str, model_id: str) -> Dict[str, float]:
        """Get current performance metrics for a model."""
        # This would integrate with your model monitoring system
        # For now, return default metrics
        return {
            "accuracy": 0.85,
            "response_quality": 0.8,
            "user_satisfaction": 0.75,
            "task_completion": 0.9
        }
    
    def _analyze_performance_trend(self, domain: str) -> Dict[str, float]:
        """Analyze performance trend for a domain."""
        history = self.performance_history.get(domain, [])
        if len(history) < 2:
            return {"decline": 0.0, "trend": "stable"}
        
        # Simple trend analysis
        recent = history[-5:]  # Last 5 measurements
        if len(recent) >= 2:
            recent_avg = sum(h.get("overall_score", 0.8) for h in recent) / len(recent)
            older_avg = sum(h.get("overall_score", 0.8) for h in history[-10:-5]) / max(1, len(history[-10:-5]))
            decline = max(0, older_avg - recent_avg)
            return {"decline": decline, "trend": "declining" if decline > 0.05 else "stable"}
        
        return {"decline": 0.0, "trend": "stable"}
    
    def _get_current_model_id(self, domain: str) -> Optional[str]:
        """Get the current model ID for a domain."""
        # This would integrate with your model registry
        # For now, return a default model ID
        return f"ft-{domain}-model-v1"
    
    def _prepare_training_file(self, training_data: List[Dict[str, Any]], 
                             cycle: LearningCycle) -> str:
        """Prepare training data file in JSONL format."""
        filename = f"adaptive_training_{cycle.domain}_{cycle.id}.jsonl"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example) + '\n')
        
        return str(filepath)
    
    def _get_adaptive_training_config(self, cycle: LearningCycle) -> FineTuningConfig:
        """Get adaptive training configuration for a learning cycle."""
        domain_config = self.config.domain_specific_thresholds.get(
            cycle.domain, {}
        ) if self.config.domain_specific_thresholds else {}
        
        return FineTuningConfig(
            model="gpt-4o-2024-11-20",  # GPT-4.1 for fine-tuning
            domain_name=cycle.domain,
            n_epochs=domain_config.get("n_epochs", 3),
            batch_size=domain_config.get("batch_size", 4) if self.config.adaptive_batch_size else 8,
            learning_rate=domain_config.get("learning_rate", 1e-5),
            suffix=f"{cycle.domain}-adaptive-{cycle.id[:8]}",
            validation_split=1 - self.config.training_data_ratio
        )
    
    async def _monitor_training_job(self, cycle: LearningCycle, job_id: str):
        """Monitor training job progress."""
        max_wait_time = 3600  # 1 hour max
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < max_wait_time:
            status = self.trainer.get_job_status(job_id)
            
            if status.get("status") in ["succeeded", "failed", "cancelled"]:
                break
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        logger.info(f"Training job {job_id} monitoring complete")
    
    def _generate_evaluation_prompts(self, cycle: LearningCycle) -> List[str]:
        """Generate evaluation prompts for model validation."""
        domain_ctx = self.domain_manager.get_domain_context(cycle.domain)
        prompts = domain_ctx.example_prompts if domain_ctx else []
        
        # Add improvement-focused prompts
        for improvement in cycle.improvements_identified[:3]:
            prompts.append(f"Explain how to address: {improvement}")
        
        return prompts
    
    def _compare_model_performance(self, old_results: List[Dict], 
                                 new_results: List[Dict], 
                                 cycle: LearningCycle) -> Dict[str, float]:
        """Compare performance between old and new models."""
        # Simple performance comparison
        # In reality, this would use more sophisticated metrics
        
        if not old_results or not new_results:
            return {"overall_improvement": 0.0}
        
        # Calculate improvement based on response quality (simplified)
        old_quality = sum(1 for r in old_results if not r.get("error", False)) / len(old_results)
        new_quality = sum(1 for r in new_results if not r.get("error", False)) / len(new_results)
        
        improvement = new_quality - old_quality
        
        return {
            "overall_improvement": improvement,
            "old_success_rate": old_quality,
            "new_success_rate": new_quality,
            "improvement_percentage": improvement * 100
        }
    
    def _update_domain_model_registry(self, domain: str, model_id: str, cycle: LearningCycle):
        """Update the domain model registry with new model."""
        # This would integrate with your model management system
        registry_file = self.storage_path / "model_registry.json"
        
        registry = {}
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        
        registry[domain] = {
            "current_model": model_id,
            "previous_model": cycle.model_id,
            "updated_at": cycle.completed_at,
            "cycle_id": cycle.id,
            "improvement_metrics": cycle.current_metrics
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _deduplicate_training_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate training examples and sort by quality."""
        seen_content = set()
        unique_data = []
        
        for example in training_data:
            # Create content hash for deduplication
            content_str = str(example.get("messages", ""))
            if content_str not in seen_content:
                seen_content.add(content_str)
                unique_data.append(example)
        
        # Sort by quality score if available
        unique_data.sort(
            key=lambda x: x.get("quality_score", 0.5), 
            reverse=True
        )
        
        return unique_data
    
    def _update_learning_metrics(self, cycle: LearningCycle):
        """Update learning metrics after cycle completion."""
        domain = cycle.domain
        
        if domain not in self.learning_metrics:
            self.learning_metrics[domain] = {
                "total_cycles": 0,
                "successful_cycles": 0,
                "average_improvement": 0.0,
                "last_cycle": None
            }
        
        metrics = self.learning_metrics[domain]
        metrics["total_cycles"] += 1
        
        if cycle.phase == LearningPhase.COMPLETE:
            metrics["successful_cycles"] += 1
            
            # Update average improvement
            current_improvement = cycle.current_metrics.get("overall_improvement", 0.0)
            metrics["average_improvement"] = (
                (metrics["average_improvement"] * (metrics["successful_cycles"] - 1) + current_improvement) 
                / metrics["successful_cycles"]
            )
        
        metrics["last_cycle"] = cycle.completed_at or cycle.updated_at
    
    def _load_learning_state(self):
        """Load learning state from persistent storage."""
        state_file = self.storage_path / "learning_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                # Load metrics
                self.learning_metrics = state.get("learning_metrics", {})
                self.performance_history = state.get("performance_history", {})
                
                # Load cycle history
                for cycle_data in state.get("cycle_history", []):
                    cycle = LearningCycle(**cycle_data)
                    self.cycle_history.append(cycle)
                
                logger.info("Learning state loaded successfully")
            
            except Exception as e:
                logger.error(f"Error loading learning state: {str(e)}")
    
    def _save_learning_state(self):
        """Save learning state to persistent storage."""
        state_file = self.storage_path / "learning_state.json"
        
        try:
            state = {
                "learning_metrics": self.learning_metrics,
                "performance_history": self.performance_history,
                "cycle_history": [asdict(cycle) for cycle in self.cycle_history[-100:]],  # Keep last 100
                "saved_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving learning state: {str(e)}")
