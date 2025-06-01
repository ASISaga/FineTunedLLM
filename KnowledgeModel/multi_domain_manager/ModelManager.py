"""
Model Management and Versioning System
Manages the lifecycle of domain-specific fine-tuned models including versioning,
deployment, rollback, and performance tracking.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ROLLBACK = "rollback"

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    domain: str
    version: str
    parent_model: Optional[str]
    training_config: Dict[str, Any]
    training_data_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    status: ModelStatus
    created_at: str
    last_updated: str
    deployed_at: Optional[str]
    endpoint_url: Optional[str]
    deployment_config: Dict[str, Any]
    feedback_score: float
    usage_count: int
    error_rate: float
    latency_p95: float
    cost_per_request: float
    model_size_mb: float
    training_time_hours: float
    tags: List[str]
    notes: str

@dataclass
class DeploymentRecord:
    """Deployment history record"""
    deployment_id: str
    model_id: str
    domain: str
    version: str
    status: DeploymentStatus
    deployed_at: str
    deployment_config: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    rollback_reason: Optional[str]
    rollback_at: Optional[str]
    deployment_notes: str

@dataclass
class ModelComparison:
    """Model comparison results"""
    base_model: str
    comparison_model: str
    domain: str
    performance_diff: Dict[str, float]
    quality_metrics: Dict[str, float]
    cost_analysis: Dict[str, float]
    recommendation: str
    detailed_analysis: Dict[str, Any]

class ModelManager:
    """
    Manages model lifecycle, versioning, and deployment coordination.
    
    Features:
    - Model versioning and metadata tracking
    - Deployment management and rollback
    - Performance monitoring and comparison
    - Automated model promotion/demotion
    - Cross-domain model coordination
    """
    
    def __init__(self, base_path: str = "./model_registry"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Model registry storage
        self.models: Dict[str, ModelMetadata] = {}
        self.deployments: Dict[str, List[DeploymentRecord]] = {}
        self.domain_active_models: Dict[str, str] = {}
        
        # Performance thresholds
        self.performance_thresholds = {
            "min_feedback_score": 4.0,
            "max_error_rate": 0.05,
            "max_latency_p95": 2000,  # milliseconds
            "min_usage_threshold": 100  # minimum requests before promotion
        }
        
        # Load existing registry
        self._load_model_registry()
    
    def register_model(self, model_id: str, domain: str, training_config: Dict[str, Any],
                      training_data_info: Dict[str, Any], parent_model: str = None,
                      tags: List[str] = None, notes: str = "") -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique model identifier
            domain: Domain the model was trained for
            training_config: Configuration used for training
            training_data_info: Information about training data
            parent_model: ID of parent model (for incremental training)
            tags: Optional tags for model categorization
            notes: Optional notes about the model
            
        Returns:
            Version string for the registered model
        """
        # Generate version
        domain_models = [m for m in self.models.values() if m.domain == domain]
        version = f"v{len(domain_models) + 1}.0.0"
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_id=model_id,
            domain=domain,
            version=version,
            parent_model=parent_model,
            training_config=training_config,
            training_data_info=training_data_info,
            performance_metrics={},
            status=ModelStatus.TRAINING,
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
            deployed_at=None,
            endpoint_url=None,
            deployment_config={},
            feedback_score=0.0,
            usage_count=0,
            error_rate=0.0,
            latency_p95=0.0,
            cost_per_request=0.0,
            model_size_mb=0.0,
            training_time_hours=0.0,
            tags=tags or [],
            notes=notes
        )
        
        # Register model
        self.models[model_id] = model_metadata
        self.deployments[model_id] = []
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Registered model {model_id} version {version} for domain {domain}")
        return version
    
    def update_model_status(self, model_id: str, status: ModelStatus,
                           performance_metrics: Dict[str, float] = None,
                           endpoint_url: str = None,
                           deployment_config: Dict[str, Any] = None) -> bool:
        """
        Update model status and performance metrics.
        
        Args:
            model_id: Model identifier
            status: New model status
            performance_metrics: Updated performance metrics
            endpoint_url: Deployment endpoint URL
            deployment_config: Deployment configuration
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        model = self.models[model_id]
        model.status = status
        model.last_updated = datetime.now(timezone.utc).isoformat()
        
        if performance_metrics:
            model.performance_metrics.update(performance_metrics)
        
        if endpoint_url:
            model.endpoint_url = endpoint_url
        
        if deployment_config:
            model.deployment_config.update(deployment_config)
        
        if status == ModelStatus.DEPLOYED:
            model.deployed_at = datetime.now(timezone.utc).isoformat()
        
        # Save changes
        self._save_model_registry()
        
        logger.info(f"Updated model {model_id} status to {status.value}")
        return True
    
    def deploy_model(self, model_id: str, deployment_config: Dict[str, Any] = None,
                    auto_promote: bool = True) -> str:
        """
        Deploy a model and optionally promote to active status.
        
        Args:
            model_id: Model to deploy
            deployment_config: Deployment configuration
            auto_promote: Whether to automatically promote if performance is good
            
        Returns:
            Deployment ID
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        deployment_id = f"deploy_{model_id}_{int(datetime.now().timestamp())}"
        
        # Get current active model for comparison
        current_active = self.domain_active_models.get(model.domain)
        performance_before = {}
        if current_active and current_active in self.models:
            performance_before = self.models[current_active].performance_metrics.copy()
        
        # Create deployment record
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            domain=model.domain,
            version=model.version,
            status=DeploymentStatus.PENDING,
            deployed_at=datetime.now(timezone.utc).isoformat(),
            deployment_config=deployment_config or {},
            performance_before=performance_before,
            performance_after={},
            rollback_reason=None,
            rollback_at=None,
            deployment_notes=""
        )
        
        # Add to deployment history
        self.deployments[model_id].append(deployment_record)
        
        # Update model status
        self.update_model_status(
            model_id, 
            ModelStatus.DEPLOYED,
            deployment_config=deployment_config
        )
        
        # Auto-promote if enabled
        if auto_promote:
            self._evaluate_model_promotion(model_id)
        
        # Save changes
        self._save_model_registry()
        
        logger.info(f"Deployed model {model_id} with deployment ID {deployment_id}")
        return deployment_id
    
    def promote_model_to_active(self, model_id: str, force: bool = False) -> bool:
        """
        Promote a model to active status for its domain.
        
        Args:
            model_id: Model to promote
            force: Force promotion even if performance thresholds not met
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            logger.error(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        
        # Check performance thresholds unless forced
        if not force and not self._meets_promotion_criteria(model):
            logger.warning(f"Model {model_id} does not meet promotion criteria")
            return False
        
        # Get current active model
        current_active = self.domain_active_models.get(model.domain)
        
        # Demote current active model if exists
        if current_active and current_active in self.models:
            self.models[current_active].status = ModelStatus.DEPRECATED
            logger.info(f"Demoted model {current_active} from active status")
        
        # Promote new model
        self.domain_active_models[model.domain] = model_id
        model.status = ModelStatus.DEPLOYED
        
        # Update deployment record
        latest_deployment = self.deployments[model_id][-1] if self.deployments[model_id] else None
        if latest_deployment:
            latest_deployment.status = DeploymentStatus.DEPLOYED
            latest_deployment.performance_after = model.performance_metrics.copy()
        
        # Save changes
        self._save_model_registry()
        
        logger.info(f"Promoted model {model_id} to active status for domain {model.domain}")
        return True
    
    def rollback_deployment(self, deployment_id: str, reason: str) -> bool:
        """
        Rollback a deployment to the previous active model.
        
        Args:
            deployment_id: Deployment to rollback
            reason: Reason for rollback
            
        Returns:
            Success status
        """
        # Find deployment record
        deployment_record = None
        model_id = None
        
        for mid, deployments in self.deployments.items():
            for deployment in deployments:
                if deployment.deployment_id == deployment_id:
                    deployment_record = deployment
                    model_id = mid
                    break
            if deployment_record:
                break
        
        if not deployment_record:
            logger.error(f"Deployment {deployment_id} not found")
            return False
        
        # Find previous active model
        domain = deployment_record.domain
        domain_models = [
            m for m in self.models.values() 
            if m.domain == domain and m.status == ModelStatus.DEPRECATED
        ]
        
        if not domain_models:
            logger.error(f"No previous model found for rollback in domain {domain}")
            return False
        
        # Get most recent deprecated model
        previous_model = max(domain_models, key=lambda x: x.deployed_at or "")
        
        # Update deployment record
        deployment_record.status = DeploymentStatus.ROLLED_BACK
        deployment_record.rollback_reason = reason
        deployment_record.rollback_at = datetime.now(timezone.utc).isoformat()
        
        # Update model statuses
        if model_id in self.models:
            self.models[model_id].status = ModelStatus.ROLLBACK
        
        previous_model.status = ModelStatus.DEPLOYED
        self.domain_active_models[domain] = previous_model.model_id
        
        # Save changes
        self._save_model_registry()
        
        logger.info(f"Rolled back deployment {deployment_id}, restored model {previous_model.model_id}")
        return True
    
    def compare_models(self, model_id_1: str, model_id_2: str) -> ModelComparison:
        """
        Compare two models across various metrics.
        
        Args:
            model_id_1: First model to compare
            model_id_2: Second model to compare
            
        Returns:
            Detailed comparison results
        """
        if model_id_1 not in self.models or model_id_2 not in self.models:
            raise ValueError("One or both models not found")
        
        model1 = self.models[model_id_1]
        model2 = self.models[model_id_2]
        
        # Calculate performance differences
        performance_diff = {}
        for metric in set(model1.performance_metrics.keys()) | set(model2.performance_metrics.keys()):
            val1 = model1.performance_metrics.get(metric, 0)
            val2 = model2.performance_metrics.get(metric, 0)
            performance_diff[metric] = val2 - val1
        
        # Quality metrics comparison
        quality_metrics = {
            "feedback_score_diff": model2.feedback_score - model1.feedback_score,
            "error_rate_diff": model2.error_rate - model1.error_rate,
            "latency_diff": model2.latency_p95 - model1.latency_p95,
            "usage_ratio": model2.usage_count / max(model1.usage_count, 1)
        }
        
        # Cost analysis
        cost_analysis = {
            "cost_per_request_diff": model2.cost_per_request - model1.cost_per_request,
            "model_size_diff": model2.model_size_mb - model1.model_size_mb,
            "training_time_diff": model2.training_time_hours - model1.training_time_hours
        }
        
        # Generate recommendation
        recommendation = self._generate_model_recommendation(
            model1, model2, performance_diff, quality_metrics, cost_analysis
        )
        
        # Detailed analysis
        detailed_analysis = {
            "model1_summary": {
                "id": model1.model_id,
                "version": model1.version,
                "status": model1.status.value,
                "feedback_score": model1.feedback_score,
                "usage_count": model1.usage_count
            },
            "model2_summary": {
                "id": model2.model_id,
                "version": model2.version,
                "status": model2.status.value,
                "feedback_score": model2.feedback_score,
                "usage_count": model2.usage_count
            },
            "performance_trends": self._calculate_performance_trends(model1, model2),
            "deployment_history": self._get_deployment_comparison(model_id_1, model_id_2)
        }
        
        return ModelComparison(
            base_model=model_id_1,
            comparison_model=model_id_2,
            domain=model1.domain,
            performance_diff=performance_diff,
            quality_metrics=quality_metrics,
            cost_analysis=cost_analysis,
            recommendation=recommendation,
            detailed_analysis=detailed_analysis
        )
    
    def get_domain_models(self, domain: str, status: ModelStatus = None) -> List[ModelMetadata]:
        """
        Get all models for a specific domain.
        
        Args:
            domain: Domain to filter by
            status: Optional status filter
            
        Returns:
            List of models matching criteria
        """
        models = [m for m in self.models.values() if m.domain == domain]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_active_model(self, domain: str) -> Optional[ModelMetadata]:
        """
        Get the currently active model for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Active model metadata if exists
        """
        active_model_id = self.domain_active_models.get(domain)
        if active_model_id and active_model_id in self.models:
            return self.models[active_model_id]
        return None
    
    def update_model_metrics(self, model_id: str, feedback_score: float = None,
                            usage_count: int = None, error_rate: float = None,
                            latency_p95: float = None, cost_per_request: float = None) -> bool:
        """
        Update real-time model metrics.
        
        Args:
            model_id: Model identifier
            feedback_score: Average feedback score
            usage_count: Total usage count
            error_rate: Current error rate
            latency_p95: 95th percentile latency
            cost_per_request: Cost per request
            
        Returns:
            Success status
        """
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        if feedback_score is not None:
            model.feedback_score = feedback_score
        if usage_count is not None:
            model.usage_count = usage_count
        if error_rate is not None:
            model.error_rate = error_rate
        if latency_p95 is not None:
            model.latency_p95 = latency_p95
        if cost_per_request is not None:
            model.cost_per_request = cost_per_request
        
        model.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Check if model should be demoted
        if not self._meets_promotion_criteria(model) and model.status == ModelStatus.DEPLOYED:
            logger.warning(f"Model {model_id} no longer meets performance criteria")
            self._schedule_model_evaluation(model_id)
        
        # Save changes
        self._save_model_registry()
        
        return True
    
    def get_model_deployment_history(self, model_id: str) -> List[DeploymentRecord]:
        """
        Get deployment history for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of deployment records
        """
        return self.deployments.get(model_id, [])
    
    def cleanup_deprecated_models(self, older_than_days: int = 30) -> int:
        """
        Clean up deprecated models older than specified days.
        
        Args:
            older_than_days: Models older than this will be cleaned up
            
        Returns:
            Number of models cleaned up
        """
        cleanup_threshold = datetime.now(timezone.utc).timestamp() - (older_than_days * 24 * 3600)
        
        models_to_remove = []
        for model_id, model in self.models.items():
            if (model.status in [ModelStatus.DEPRECATED, ModelStatus.FAILED] and
                datetime.fromisoformat(model.created_at.replace('Z', '+00:00')).timestamp() < cleanup_threshold):
                models_to_remove.append(model_id)
        
        # Remove models
        for model_id in models_to_remove:
            del self.models[model_id]
            if model_id in self.deployments:
                del self.deployments[model_id]
        
        # Save changes
        self._save_model_registry()
        
        logger.info(f"Cleaned up {len(models_to_remove)} deprecated models")
        return len(models_to_remove)
    
    def export_model_registry(self) -> str:
        """Export complete model registry as JSON."""
        registry_data = {
            "models": {mid: asdict(model) for mid, model in self.models.items()},
            "deployments": {
                mid: [asdict(dep) for dep in deps] 
                for mid, deps in self.deployments.items()
            },
            "active_models": self.domain_active_models.copy(),
            "thresholds": self.performance_thresholds.copy(),
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
        return json.dumps(registry_data, indent=2, default=str)
    
    def import_model_registry(self, registry_json: str) -> bool:
        """
        Import model registry from JSON.
        
        Args:
            registry_json: JSON string containing registry data
            
        Returns:
            Success status
        """
        try:
            registry_data = json.loads(registry_json)
            
            # Import models
            for model_id, model_data in registry_data.get("models", {}).items():
                # Convert status back to enum
                model_data["status"] = ModelStatus(model_data["status"])
                self.models[model_id] = ModelMetadata(**model_data)
            
            # Import deployments
            for model_id, deployment_list in registry_data.get("deployments", {}).items():
                deployments = []
                for dep_data in deployment_list:
                    dep_data["status"] = DeploymentStatus(dep_data["status"])
                    deployments.append(DeploymentRecord(**dep_data))
                self.deployments[model_id] = deployments
            
            # Import active models
            self.domain_active_models.update(registry_data.get("active_models", {}))
            
            # Import thresholds
            self.performance_thresholds.update(registry_data.get("thresholds", {}))
            
            # Save imported data
            self._save_model_registry()
            
            logger.info("Successfully imported model registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import model registry: {str(e)}")
            return False
    
    def _meets_promotion_criteria(self, model: ModelMetadata) -> bool:
        """Check if model meets promotion criteria."""
        return (
            model.feedback_score >= self.performance_thresholds["min_feedback_score"] and
            model.error_rate <= self.performance_thresholds["max_error_rate"] and
            model.latency_p95 <= self.performance_thresholds["max_latency_p95"] and
            model.usage_count >= self.performance_thresholds["min_usage_threshold"]
        )
    
    def _evaluate_model_promotion(self, model_id: str):
        """Evaluate if model should be promoted to active."""
        if model_id not in self.models:
            return
        
        model = self.models[model_id]
        
        # Wait for sufficient usage before promotion
        if model.usage_count < self.performance_thresholds["min_usage_threshold"]:
            logger.info(f"Model {model_id} needs more usage before promotion evaluation")
            return
        
        if self._meets_promotion_criteria(model):
            self.promote_model_to_active(model_id)
        else:
            logger.info(f"Model {model_id} does not meet promotion criteria")
    
    def _schedule_model_evaluation(self, model_id: str):
        """Schedule evaluation for underperforming model."""
        # This would typically integrate with a task scheduler
        logger.info(f"Scheduled evaluation for model {model_id}")
    
    def _generate_model_recommendation(self, model1: ModelMetadata, model2: ModelMetadata,
                                     performance_diff: Dict[str, float],
                                     quality_metrics: Dict[str, float],
                                     cost_analysis: Dict[str, float]) -> str:
        """Generate recommendation based on comparison."""
        
        # Score each model based on multiple criteria
        model1_score = 0
        model2_score = 0
        
        # Performance scoring
        if quality_metrics["feedback_score_diff"] > 0:
            model2_score += 3
        else:
            model1_score += 3
        
        if quality_metrics["error_rate_diff"] < 0:
            model2_score += 2
        else:
            model1_score += 2
        
        if quality_metrics["latency_diff"] < 0:
            model2_score += 2
        else:
            model1_score += 2
        
        # Cost scoring
        if cost_analysis["cost_per_request_diff"] < 0:
            model2_score += 1
        else:
            model1_score += 1
        
        # Usage scoring
        if quality_metrics["usage_ratio"] > 1.2:
            model2_score += 1
        elif quality_metrics["usage_ratio"] < 0.8:
            model1_score += 1
        
        if model2_score > model1_score:
            return f"Recommend {model2.model_id}: Better overall performance and metrics"
        elif model1_score > model2_score:
            return f"Recommend {model1.model_id}: Better overall performance and metrics"
        else:
            return "Models are comparable, consider specific use case requirements"
    
    def _calculate_performance_trends(self, model1: ModelMetadata, model2: ModelMetadata) -> Dict[str, Any]:
        """Calculate performance trends between models."""
        return {
            "improvement_areas": [
                metric for metric, diff in model2.performance_metrics.items()
                if diff > model1.performance_metrics.get(metric, 0)
            ],
            "regression_areas": [
                metric for metric, diff in model2.performance_metrics.items()
                if diff < model1.performance_metrics.get(metric, 0)
            ],
            "version_progression": f"{model1.version} -> {model2.version}"
        }
    
    def _get_deployment_comparison(self, model_id_1: str, model_id_2: str) -> Dict[str, Any]:
        """Get deployment comparison between models."""
        deployments_1 = self.deployments.get(model_id_1, [])
        deployments_2 = self.deployments.get(model_id_2, [])
        
        return {
            "model1_deployments": len(deployments_1),
            "model2_deployments": len(deployments_2),
            "model1_latest_status": deployments_1[-1].status.value if deployments_1 else "none",
            "model2_latest_status": deployments_2[-1].status.value if deployments_2 else "none"
        }
    
    def _load_model_registry(self):
        """Load model registry from disk."""
        registry_file = self.base_path / "model_registry.json"
        
        if not registry_file.exists():
            logger.info("No existing model registry found, starting fresh")
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            for model_id, model_data in registry_data.get("models", {}).items():
                model_data["status"] = ModelStatus(model_data["status"])
                self.models[model_id] = ModelMetadata(**model_data)
            
            # Load deployments
            for model_id, deployment_list in registry_data.get("deployments", {}).items():
                deployments = []
                for dep_data in deployment_list:
                    dep_data["status"] = DeploymentStatus(dep_data["status"])
                    deployments.append(DeploymentRecord(**dep_data))
                self.deployments[model_id] = deployments
            
            # Load active models
            self.domain_active_models = registry_data.get("active_models", {})
            
            # Load thresholds
            if "thresholds" in registry_data:
                self.performance_thresholds.update(registry_data["thresholds"])
            
            logger.info(f"Loaded model registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {str(e)}")
    
    def _save_model_registry(self):
        """Save model registry to disk."""
        registry_file = self.base_path / "model_registry.json"
        
        try:
            registry_data = {
                "models": {
                    mid: {**asdict(model), "status": model.status.value}
                    for mid, model in self.models.items()
                },
                "deployments": {
                    mid: [
                        {**asdict(dep), "status": dep.status.value}
                        for dep in deps
                    ]
                    for mid, deps in self.deployments.items()
                },
                "active_models": self.domain_active_models.copy(),
                "thresholds": self.performance_thresholds.copy(),
                "last_saved": datetime.now(timezone.utc).isoformat()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.debug("Saved model registry to disk")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {str(e)}")
