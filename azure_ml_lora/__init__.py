"""
Azure ML LoRA Training Pipeline
Consolidates all Azure ML and LoRA adapter functionality from BusinessInfinity ml_pipeline
"""
from .unified_manager import UnifiedMLManager
from .manager import MLManager
from .trainer import LoRATrainer
from .pipeline import LoRAPipeline
from .endpoints import AML_ENDPOINTS

# Create singleton instance for backwards compatibility
ml_manager = UnifiedMLManager()

__all__ = [
    'UnifiedMLManager',
    'MLManager', 
    'LoRATrainer',
    'LoRAPipeline',
    'AML_ENDPOINTS',
    'ml_manager'
]