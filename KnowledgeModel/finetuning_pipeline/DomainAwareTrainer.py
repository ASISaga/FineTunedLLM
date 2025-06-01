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
import boto3
from dataclasses import dataclass
from DomainContextManager import DomainContextManager, DomainContext

logger = logging.getLogger(__name__)

@dataclass
class FineTuningConfig:
    """Configuration for domain-specific fine-tuning"""
    model: str = "gpt-4-turbo-2024-04-09"  # GPT-4.1 equivalent for fine-tuning
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
    
    def __init__(self, api_key: str, organization: Optional[str] = None, 
                 aws_access_key_id: Optional[str] = None, aws_secret_access_key: Optional[str] = None,
                 aws_region: str = "us-east-1"):
        """
        Initialize the domain-aware trainer.
        
        :param api_key: OpenAI API key
        :param organization: OpenAI organization ID (optional)
        :param aws_access_key_id: AWS access key for Bedrock
        :param aws_secret_access_key: AWS secret key for Bedrock
        :param aws_region: AWS region for Bedrock
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization
        )
          # Initialize Bedrock client for Claude Sonnet 4
        if aws_access_key_id and aws_secret_access_key:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=aws_region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            self.bedrock_client = None
            
        self.domain_manager = DomainContextManager()
        self.active_jobs: Dict[str, Dict] = {}
        logger.info("Initialized DomainAwareTrainer with OpenAI and Bedrock integration")
    
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
            
            if job_id in self.active_jobs:            self.active_jobs[job_id]["status"] = "cancelled"
            
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
    
    def generate_training_data_with_bedrock(self, text_documents: List[str], domain_name: str,
                                           config: FineTuningConfig) -> Tuple[str, str]:
        """
        Generate training data using Claude Sonnet 4 via Amazon Bedrock for summarization and insight extraction.
        
        :param text_documents: List of text documents to process
        :param domain_name: Domain context name
        :param config: Fine-tuning configuration
        :return: Tuple of (training_file_path, validation_file_path)
        """
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized. Please provide AWS credentials when creating DomainAwareTrainer.")
        
        domain_context = self.domain_manager.get_domain_context(domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        logger.info(f"Generating training data using Claude Sonnet 4 via Bedrock for {len(text_documents)} documents")
        
        # Generate training examples using Claude
        training_examples = []
        
        for doc_idx, document in enumerate(text_documents):
            try:
                # Create prompts for different types of training examples
                example_types = [
                    ("summarization", f"Summarize the following {domain_context.domain_name} content:"),
                    ("qa", f"Generate a question and answer based on this {domain_context.domain_name} content:"),
                    ("explanation", f"Explain the key concepts in this {domain_context.domain_name} content:"),
                    ("analysis", f"Analyze the main points in this {domain_context.domain_name} content:")
                ]
                
                for example_type, prompt_prefix in example_types:
                    try:
                        # Format the prompt for Claude via Bedrock
                        bedrock_prompt = f"""You are an expert in {domain_context.domain_name}. 
{domain_context.description}

{prompt_prefix}

{document}

Please provide a comprehensive and accurate response that demonstrates deep understanding of {domain_context.domain_name} concepts."""
                        
                        # Call Claude via Bedrock
                        response = self.bedrock_client.invoke_model(
                            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                            contentType="application/json",
                            accept="application/json",
                            body=json.dumps({
                                "anthropic_version": "bedrock-2023-05-31",
                                "max_tokens": 4096,
                                "temperature": 0.1,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": bedrock_prompt
                                    }
                                ]
                            })
                        )
                        
                        # Parse response
                        response_body = json.loads(response['body'].read())
                        assistant_response = response_body['content'][0]['text']
                        
                        # Create training example in OpenAI fine-tuning format
                        training_example = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": f"You are an expert in {domain_context.domain_name}. {domain_context.description}"
                                },
                                {
                                    "role": "user",
                                    "content": f"{prompt_prefix}\n\n{document}"
                                },
                                {
                                    "role": "assistant",
                                    "content": assistant_response
                                }
                            ]
                        }
                        training_examples.append(training_example)
                        
                        logger.info(f"Generated {example_type} example for document {doc_idx + 1}")
                        
                        # Add small delay to respect rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Failed to generate {example_type} example for document {doc_idx + 1}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to process document {doc_idx + 1}: {str(e)}")
                continue
        
        if not training_examples:
            raise ValueError("No training examples were generated successfully")
        
        # Split data into training and validation
        split_idx = int(len(training_examples) * (1 - config.validation_split))
        training_data = training_examples[:split_idx]
        validation_data = training_examples[split_idx:] if split_idx < len(training_examples) else []
        
        # Save training data
        training_file = f"bedrock_training_data_{domain_name}_{int(time.time())}.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
          # Save validation data if available
        validation_file = None
        if validation_data:
            validation_file = f"bedrock_validation_data_{domain_name}_{int(time.time())}.jsonl"
            with open(validation_file, 'w', encoding='utf-8') as f:
                for item in validation_data:
                    f.write(json.dumps(item) + '\n')
        
        logger.info(f"Generated {len(training_data)} training examples and {len(validation_data)} validation examples using Claude Sonnet 4 via Bedrock")
        return training_file, validation_file
    
    def run_complete_training_pipeline(self, text_documents: List[str], domain_name: str,
                                     config: Optional[FineTuningConfig] = None) -> str:
        """
        Run the complete training pipeline: Claude Sonnet 4 for JSONL generation + OpenAI GPT-4.1 fine-tuning.
        
        :param text_documents: List of text documents to process
        :param domain_name: Domain context name
        :param config: Fine-tuning configuration (optional, will use defaults if not provided)
        :return: Fine-tuning job ID
        """
        if config is None:
            config = FineTuningConfig(domain_name=domain_name)
        
        logger.info(f"Starting complete training pipeline for domain: {domain_name}")
        logger.info(f"Using Claude Sonnet 4 via Bedrock for JSONL generation and OpenAI {config.model} for fine-tuning")
        
        # Step 1: Generate training data using Claude Sonnet 4 via Bedrock
        try:
            training_file, validation_file = self.generate_training_data_with_bedrock(
                text_documents, domain_name, config
            )
            config.training_data_path = training_file
            if validation_file:
                config.validation_data_path = validation_file
                
            logger.info("Successfully generated training data using Claude Sonnet 4 via Bedrock")
        except Exception as e:
            logger.error(f"Failed to generate training data with Claude via Bedrock: {str(e)}")
            raise
        
        # Step 2: Start fine-tuning with OpenAI GPT-4.1
        try:
            job_id = self.start_domain_fine_tuning(config)
            logger.info(f"Successfully started fine-tuning job {job_id} with OpenAI {config.model}")
            return job_id
        except Exception as e:
            logger.error(f"Failed to start fine-tuning job: {str(e)}")
            raise
