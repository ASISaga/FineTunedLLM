"""
FineTunedLLM Package for Business Infinity

Provides LoRA (Low-Rank Adaptation) fine-tuned language models with legendary
business expertise for autonomous boardroom agents. Integrates Azure Machine
Learning for training and deployment of specialized business domain adapters.
"""

from .lora_manager import LoRAManager, LegendaryProfile, LoRAAdapter
from .mentor_mode import (
    MentorMode, 
    TrainingSession, 
    TrainingFeedback, 
    TrainingPhase, 
    FeedbackType
)

__version__ = "1.0.0"
__author__ = "Business Infinity AI"
__description__ = "FineTunedLLM with Legendary Business Expertise"

__all__ = [
    # Core LoRA Management
    "LoRAManager",
    "LegendaryProfile", 
    "LoRAAdapter",
    
    # Mentor Mode Training
    "MentorMode",
    "TrainingSession",
    "TrainingFeedback", 
    "TrainingPhase",
    "FeedbackType"
]


async def create_finetuned_llm_system():
    """
    Factory function to create and initialize the FineTunedLLM system
    
    Returns:
        Tuple of (LoRAManager, MentorMode) - initialized and ready to use
    """
    # Initialize LoRA Manager
    lora_manager = LoRAManager()
    await lora_manager.initialize()
    
    # Initialize Mentor Mode with LoRA Manager
    mentor_mode = MentorMode(lora_manager)
    await mentor_mode.initialize()
    
    return lora_manager, mentor_mode


class FineTunedLLMSystem:
    """
    Main system class that combines LoRA management and Mentor Mode training
    for legendary business expertise in Business Infinity autonomous boardroom.
    """
    
    def __init__(self):
        self.lora_manager = None
        self.mentor_mode = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the complete FineTunedLLM system"""
        if self._initialized:
            return
        
        self.lora_manager, self.mentor_mode = await create_finetuned_llm_system()
        self._initialized = True
    
    async def get_legendary_expertise(self, legend_name: str, domain: str, prompt: str) -> str:
        """Get legendary expertise response for a given prompt"""
        if not self._initialized:
            await self.initialize()
        
        adapter_id = await self.lora_manager.load_legendary_adapter(legend_name, domain)
        return await self.lora_manager.get_legendary_response(adapter_id, prompt)
    
    async def start_training_session(self, adapter_id: str, expert_id: str = "system") -> str:
        """Start a new mentor mode training session"""
        if not self._initialized:
            await self.initialize()
        
        return await self.mentor_mode.start_training_session(adapter_id, expert_id)
    
    async def provide_training_feedback(self, session_id: str, **kwargs) -> str:
        """Provide feedback for a training session"""
        if not self._initialized:
            await self.initialize()
        
        return await self.mentor_mode.provide_feedback(session_id, **kwargs)
    
    async def get_system_status(self) -> dict:
        """Get status of the entire FineTunedLLM system"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "available_adapters": await self.lora_manager.list_available_adapters(),
            "legendary_profiles": await self.lora_manager.list_legendary_profiles(),
            "active_training_sessions": await self.mentor_mode.get_active_sessions(),
            "training_metrics": await self.mentor_mode.get_training_metrics()
        }
    
    async def shutdown(self):
        """Graceful shutdown of the FineTunedLLM system"""
        if not self._initialized:
            return
        
        if self.mentor_mode:
            await self.mentor_mode.shutdown()
        
        if self.lora_manager:
            await self.lora_manager.shutdown()
        
        self._initialized = False