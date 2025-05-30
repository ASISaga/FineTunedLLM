"""
Domain-Aware Fine-Tuning Trainer
This module provides domain-specific fine-tuning capabilities for OpenAI models
with integrated domain context management.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import openai
from dataclasses import dataclass
from DomainContextManager import DomainContextManager, DomainContext

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for domain-specific fine-tuning"""
    model: str = "gpt-3.5-turbo"
    training_data_path: str = ""
    validation_data_path: str = ""
    domain_name: str = ""
    n_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-5
    suffix: str = ""
    validation_split: float = 0.1

class DomainAwareTrainer:
    """
    Domain-aware trainer for fine-tuning OpenAI models with domain-specific context.
    """
    
    def __init__(self, api_key: str, organization: Optional[str] = None):
        """
        Initialize the domain-aware trainer.
        
        :param api_key: OpenAI API key
        :param organization: OpenAI organization ID (optional)
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization
        )
        self.domain_manager = DomainContextManager()
        self.active_jobs: Dict[str, Dict] = {}
        logger.info("Initialized DomainAwareTrainer")
    
    def prepare_domain_training_data(self, raw_data: List[Dict], domain_name: str, 
                                   config: FineTuningConfig) -> Tuple[str, str]:
        """
        Prepare domain-specific training data with enhanced prompts.
        
        :param raw_data: Raw training data (list of dicts with 'prompt' and 'response')
        :param domain_name: Domain context name
        :param config: Fine-tuning configuration
        :return: Tuple of (training_file_path, validation_file_path)
        """
        domain_context = self.domain_manager.get_domain_context(domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        # Generate domain-specific prompt template
        prompt_template = self.domain_manager.generate_training_prompt_template(domain_name)
        
        # Enhance training data with domain context
        enhanced_data = []
        for item in raw_data:
            enhanced_prompt = prompt_template.format(question=item.get('prompt', ''))
            
            enhanced_item = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an expert in {domain_context.domain_name}. {domain_context.description}"
                    },
                    {
                        "role": "user", 
                        "content": item.get('prompt', '')
                    },
                    {
                        "role": "assistant",
                        "content": item.get('response', '')
                    }
                ]
            }
            enhanced_data.append(enhanced_item)
        
        # Split data into training and validation
        split_idx = int(len(enhanced_data) * (1 - config.validation_split))
        training_data = enhanced_data[:split_idx]
        validation_data = enhanced_data[split_idx:] if split_idx < len(enhanced_data) else []
        
        # Save training data
        training_file = f"training_data_{domain_name}_{int(time.time())}.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        # Save validation data if available
        validation_file = None
        if validation_data:
            validation_file = f"validation_data_{domain_name}_{int(time.time())}.jsonl"
            with open(validation_file, 'w', encoding='utf-8') as f:
                for item in validation_data:
                    f.write(json.dumps(item) + '\n')
        
        logger.info(f"Prepared {len(training_data)} training examples and {len(validation_data)} validation examples")
        return training_file, validation_file
    
    def upload_training_file(self, file_path: str) -> str:
        """
        Upload training file to OpenAI.
        
        :param file_path: Path to the training file
        :return: File ID from OpenAI
        """
        try:
            with open(file_path, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            logger.info(f"Uploaded file {file_path} with ID: {file_response.id}")
            return file_response.id
            
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {str(e)}")
            raise
    
    def start_domain_fine_tuning(self, config: FineTuningConfig) -> str:
        """
        Start domain-specific fine-tuning job.
        
        :param config: Fine-tuning configuration
        :return: Fine-tuning job ID
        """
        domain_context = self.domain_manager.get_domain_context(config.domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {config.domain_name}")
        
        # Upload training file
        training_file_id = self.upload_training_file(config.training_data_path)
        
        # Upload validation file if provided
        validation_file_id = None
        if config.validation_data_path and os.path.exists(config.validation_data_path):
            validation_file_id = self.upload_training_file(config.validation_data_path)
        
        # Create fine-tuning job with domain-specific suffix
        suffix = config.suffix or f"{config.domain_name.lower().replace(' ', '-')}"
        
        job_params = {
            "training_file": training_file_id,
            "model": config.model,
            "suffix": suffix,
            "hyperparameters": {
                "n_epochs": config.n_epochs,
                "batch_size": config.batch_size,
                "learning_rate_multiplier": config.learning_rate
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        try:
            job_response = self.client.fine_tuning.jobs.create(**job_params)
            
            job_id = job_response.id
            self.active_jobs[job_id] = {
                "domain": config.domain_name,
                "status": job_response.status,
                "created_at": job_response.created_at,
                "model": config.model,
                "training_file": training_file_id,
                "validation_file": validation_file_id,
                "config": config
            }
            
            logger.info(f"Started fine-tuning job {job_id} for domain: {config.domain_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {str(e)}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        :param job_id: Fine-tuning job ID
        :return: Job status information
        """
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            status_info = {
                "job_id": job_id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "training_file": job.training_file,
                "validation_file": job.validation_file,
                "hyperparameters": job.hyperparameters.__dict__ if job.hyperparameters else None,
                "result_files": job.result_files,
                "trained_tokens": job.trained_tokens
            }
            
            # Update local job tracking
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = job.status
                self.active_jobs[job_id]["fine_tuned_model"] = job.fine_tuned_model
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {str(e)}")
            raise
    
    def list_fine_tuned_models(self, domain_name: Optional[str] = None) -> List[Dict]:
        """
        List fine-tuned models, optionally filtered by domain.
        
        :param domain_name: Optional domain filter
        :return: List of fine-tuned model information
        """
        try:
            models = self.client.models.list()
            fine_tuned_models = []
            
            for model in models.data:
                if model.id.startswith('ft:'):
                    model_info = {
                        "id": model.id,
                        "created": model.created,
                        "owned_by": model.owned_by,
                        "object": model.object
                    }
                    
                    # Check if model belongs to specified domain
                    if domain_name:
                        domain_suffix = domain_name.lower().replace(' ', '-')
                        if domain_suffix in model.id:
                            model_info["domain"] = domain_name
                            fine_tuned_models.append(model_info)
                    else:
                        fine_tuned_models.append(model_info)
            
            return fine_tuned_models
            
        except Exception as e:
            logger.error(f"Failed to list fine-tuned models: {str(e)}")
            raise
    
    def test_fine_tuned_model(self, model_id: str, test_prompts: List[str], 
                             domain_name: Optional[str] = None) -> List[Dict]:
        """
        Test a fine-tuned model with domain-specific prompts.
        
        :param model_id: Fine-tuned model ID
        :param test_prompts: List of test prompts
        :param domain_name: Domain context for testing
        :return: List of test results
        """
        results = []
        
        # Get domain context if provided
        system_message = "You are a helpful assistant."
        if domain_name:
            domain_context = self.domain_manager.get_domain_context(domain_name)
            if domain_context:
                system_message = f"You are an expert in {domain_context.domain_name}. {domain_context.description}"
        
        for prompt in test_prompts:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                result = {
                    "prompt": prompt,
                    "response": response.choices[0].message.content,
                    "model": model_id,
                    "usage": response.usage.__dict__ if response.usage else None
                }
                results.append(result)
                
                logger.info(f"Test completed for prompt: {prompt[:50]}...")
                
            except Exception as e:
                logger.error(f"Test failed for prompt '{prompt[:50]}...': {str(e)}")
                results.append({
                    "prompt": prompt,
                    "response": f"Error: {str(e)}",
                    "model": model_id,
                    "error": True
                })
        
        return results
    
    def generate_domain_evaluation_prompts(self, domain_name: str, count: int = 5) -> List[str]:
        """
        Generate evaluation prompts specific to a domain.
        
        :param domain_name: Domain name
        :param count: Number of prompts to generate
        :return: List of evaluation prompts
        """
        domain_context = self.domain_manager.get_domain_context(domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        # Use existing example prompts and generate variations
        base_prompts = domain_context.example_prompts
        
        # If we need more prompts than available examples, extend the list
        if len(base_prompts) < count:
            # Duplicate and modify existing prompts
            extended_prompts = base_prompts.copy()
            while len(extended_prompts) < count:
                for prompt in base_prompts:
                    if len(extended_prompts) >= count:
                        break
                    # Create variations by adding context
                    variation = f"In the context of {domain_context.focus_areas[0]}, {prompt.lower()}"
                    extended_prompts.append(variation)
            return extended_prompts[:count]
        
        return base_prompts[:count]
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running fine-tuning job.
        
        :param job_id: Job ID to cancel
        :return: True if successful, False otherwise
        """
        try:
            self.client.fine_tuning.jobs.cancel(job_id)
            
            if job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = "cancelled"
            
            logger.info(f"Cancelled fine-tuning job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {str(e)}")
            return False
    
    def get_training_metrics(self, job_id: str) -> Optional[Dict]:
        """
        Get training metrics for a completed job.
        
        :param job_id: Job ID
        :return: Training metrics if available
        """
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            
            if job.result_files:
                # Download and parse result files for metrics
                # This would require additional implementation to parse OpenAI's result files
                pass
            
            return {
                "job_id": job_id,
                "status": job.status,
                "trained_tokens": job.trained_tokens,
                "hyperparameters": job.hyperparameters.__dict__ if job.hyperparameters else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get training metrics for {job_id}: {str(e)}")
            return None
