# Azure Function for Domain-Aware Fine-Tuning Pipeline
import logging
import json
import os
import sys
import time
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
import re

# Add the KnowledgeModel directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'KnowledgeModel'))

try:
    from DomainContextManager import DomainContextManager
    from DomainAwareTrainer import DomainAwareTrainer
except ImportError:
    # Fallback for when modules are not available
    DomainContextManager = None
    DomainAwareTrainer = None

# Initialize function app
app = func.FunctionApp()

# Configuration from environment variables
KEY_VAULT_URL = os.environ.get("KEY_VAULT_URL")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
SUMMARIES_CONTAINER = os.environ.get("SUMMARIES_CONTAINER", "processed-summaries")
TRAINING_DATA_CONTAINER = os.environ.get("TRAINING_DATA_CONTAINER", "training-data")
MODELS_CONTAINER = os.environ.get("MODELS_CONTAINER", "fine-tuned-models")
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")

# Initialize Azure clients
credential = DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID)

# Initialize Key Vault client for secret retrieval
if KEY_VAULT_URL:
    kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
else:
    kv_client = None

# Initialize OpenAI client
openai_client = AzureOpenAI(endpoint=AZURE_OPENAI_ENDPOINT, credential=credential)

# Initialize storage client
if STORAGE_ACCOUNT_NAME:
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=credential
    )
elif AZURE_STORAGE_CONNECTION_STRING:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
else:
    blob_service_client = None

# Initialize domain manager
domain_manager = DomainContextManager() if DomainContextManager else None


# Initialize domain manager
domain_manager = DomainContextManager() if DomainContextManager else None


def detect_domain_from_filename(filename: str) -> Optional[str]:
    """
    Extract domain context from filename patterns.
    
    :param filename: The blob filename to analyze
    :return: Detected domain name or None
    """
    if not domain_manager:
        return None
        
    filename_lower = filename.lower()
    
    # Domain detection patterns
    domain_patterns = {
        'technical': ['tech', 'software', 'code', 'programming', 'api', 'system', 'architecture'],
        'medical': ['medical', 'health', 'patient', 'clinical', 'diagnosis', 'treatment', 'pharma'],
        'legal': ['legal', 'law', 'contract', 'compliance', 'regulation', 'policy', 'court'],
        'financial': ['financial', 'finance', 'banking', 'investment', 'market', 'economic', 'trading']
    }
    
    for domain, keywords in domain_patterns.items():
        if any(keyword in filename_lower for keyword in keywords):
            return domain
    
    return None


class DomainAwareTrainingDataGenerator:
    """
    Enhanced training data generator with domain-specific context.
    """
    
    def __init__(self, openai_client: AzureOpenAI, deployment: str):
        self.client = openai_client
        self.deployment = deployment
        self.domain_manager = domain_manager
    
    def generate_domain_aware_qa_pairs(self, summary: str, domain_name: Optional[str] = None, 
                                     domain_context: str = "") -> List[Dict[str, str]]:
        """
        Generate domain-aware question-answer pairs from summaries.
        """
        # Build enhanced system prompt with domain context
        system_prompt = (
            "You are an expert at creating high-quality training data for language models. "
            "Generate diverse, insightful question-answer pairs based on the provided summary."
        )
        
        if domain_name and self.domain_manager:
            domain_ctx = self.domain_manager.get_domain_context(domain_name)
            if domain_ctx:
                system_prompt += f"\n\nDOMAIN EXPERTISE: You are an expert in {domain_ctx.domain_name}."
                system_prompt += f"\nFOCUS AREAS: {', '.join(domain_ctx.focus_areas)}"
                system_prompt += f"\nKEY CONCEPTS: {', '.join(domain_ctx.key_concepts)}"
                system_prompt += f"\nTERMINOLOGY: Use appropriate {domain_ctx.domain_name} terminology."
        elif domain_context:
            system_prompt += f"\nDOMAIN CONTEXT: {domain_context}"
        
        user_prompt = (
            f"Based on the following summary, generate 3-5 diverse question-answer pairs "
            f"that capture the key insights and domain-specific concepts. "
            f"Format as JSON array with 'question' and 'answer' fields:\n\n"
            f"Summary: {summary}\n\n"
            f"Question-Answer Pairs:"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                qa_pairs = json.loads(content)
                return qa_pairs if isinstance(qa_pairs, list) else []
            except json.JSONDecodeError:
                # Fallback: create domain-aware QA pair
                fallback_question = f"What are the key {domain_name or 'domain-specific'} insights from this summary?"
                return [{
                    "question": fallback_question,
                    "answer": summary
                }]
                
        except Exception as e:
            logging.error(f"Error generating domain-aware QA pairs: {e}")
            return []
    
    def create_domain_enhanced_training_examples(self, qa_pairs: List[Dict[str, str]], 
                                               domain_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Convert QA pairs to OpenAI fine-tuning format with domain enhancement.
        """
        training_examples = []
        
        # Add domain context to system message if available
        system_message = "You are a helpful assistant."
        if domain_name and self.domain_manager:
            domain_ctx = self.domain_manager.get_domain_context(domain_name)
            if domain_ctx:
                system_message = f"You are an expert assistant specializing in {domain_ctx.domain_name}. " \
                               f"You provide accurate, detailed responses using appropriate {domain_ctx.domain_name} terminology."
        
        for qa in qa_pairs:
            if 'question' in qa and 'answer' in qa:
                training_example = {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": qa['question']},
                        {"role": "assistant", "content": qa['answer']}
                    ]
                }
                training_examples.append(training_example)
        
        return training_examples


class DomainAwareFineTuningOrchestrator:
    """
    Enhanced orchestrator for domain-aware fine-tuning using Azure OpenAI.
    """
    
    def __init__(self, openai_client: AzureOpenAI):
        self.client = openai_client
        self.domain_manager = domain_manager
      def create_domain_fine_tuning_job(self, training_file_id: str, domain_name: Optional[str] = None, 
                                    model: str = "gpt-4", custom_suffix: Optional[str] = None) -> str:
        """
        Create a domain-aware fine-tuning job with Azure OpenAI.
        """
        try:
            # Build domain-specific suffix for model naming
            if not custom_suffix:
                domain_suffix = f"-{domain_name}" if domain_name else "-general"
                timestamp_suffix = str(int(time.time()))
                custom_suffix = f"{domain_suffix}-{timestamp_suffix}"
            
            # Get domain-specific hyperparameters if available
            hyperparameters = self._get_domain_hyperparameters(domain_name)
            
            fine_tuning_job = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model,
                suffix=custom_suffix,
                hyperparameters=hyperparameters
            )
            
            logging.info(f"Created domain-aware fine-tuning job: {fine_tuning_job.id} for domain: {domain_name or 'general'}")
            return fine_tuning_job.id
            
        except Exception as e:
            logging.error(f"Error creating domain-aware fine-tuning job: {e}")
            raise
    
    def _get_domain_hyperparameters(self, domain_name: Optional[str]) -> Dict[str, Any]:
        """
        Get domain-specific hyperparameters for fine-tuning.
        """
        # Default hyperparameters
        default_params = {
            "n_epochs": 3,
            "batch_size": 1,
            "learning_rate_multiplier": 0.1
        }
        
        # Domain-specific adjustments
        if domain_name and self.domain_manager:
            domain_ctx = self.domain_manager.get_domain_context(domain_name)
            if domain_ctx:
                # Adjust parameters based on domain complexity
                if domain_name in ['medical', 'legal']:
                    # More conservative training for critical domains
                    default_params["n_epochs"] = 4
                    default_params["learning_rate_multiplier"] = 0.05
                elif domain_name == 'technical':
                    # More epochs for technical precision
                    default_params["n_epochs"] = 5
                    default_params["learning_rate_multiplier"] = 0.08
        
        return default_params
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job with enhanced information.
        """
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "training_file": job.training_file,
                "result_files": job.result_files if hasattr(job, 'result_files') else None,
                "trained_tokens": job.trained_tokens if hasattr(job, 'trained_tokens') else None
            }
        except Exception as e:
            logging.error(f"Error checking job status: {e}")
            return {"error": str(e)}
    
    def get_domain_training_progress(self, domain_name: str) -> Dict[str, Any]:
        """
        Get training progress and statistics for a specific domain.
        """
        try:
            # List recent fine-tuning jobs
            jobs = self.client.fine_tuning.jobs.list(limit=50)
            
            domain_jobs = []
            for job in jobs.data:
                if hasattr(job, 'suffix') and domain_name in (job.suffix or ''):
                    domain_jobs.append({
                        "id": job.id,
                        "status": job.status,
                        "created_at": job.created_at,
                        "model": job.fine_tuned_model
                    })
            
            return {
                "domain": domain_name,
                "total_jobs": len(domain_jobs),
                "recent_jobs": domain_jobs[:10],
                "summary": {
                    "completed": len([j for j in domain_jobs if j["status"] == "succeeded"]),
                    "failed": len([j for j in domain_jobs if j["status"] == "failed"]),
                    "running": len([j for j in domain_jobs if j["status"] in ["running", "validating_files"]])
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting domain training progress: {e}")
            return {"error": str(e)}


@app.blob_trigger(
    arg_name="myblob", 
    path=f"{SUMMARIES_CONTAINER}/{{name}}", 
    connection="AzureWebJobsStorage"
)
def training_data_generator(myblob: func.InputStream) -> None:
    """
    Azure Function triggered by summary blob uploads to generate domain-aware training data.
    """
    logging.info(f"Processing summary blob: {myblob.name}, Size: {myblob.length} bytes")
    
    try:
        # Read the summary data
        summary_data = json.loads(myblob.read().decode('utf-8'))
        
        # Extract filename for processing
        blob_name = myblob.name.split('/')[-1]
        base_name = os.path.splitext(blob_name)[0]
        
        # Detect domain from filename
        detected_domain = detect_domain_from_filename(blob_name)
        logging.info(f"Detected domain: {detected_domain or 'generic'} for file: {blob_name}")
        
        # Initialize domain-aware training data generator
        generator = DomainAwareTrainingDataGenerator(openai_client, AZURE_OPENAI_DEPLOYMENT)
        
        # Process all summary segments with domain context
        all_training_examples = []
        
        for segment in summary_data.get('processed_segments', []):
            summary_text = segment.get('summary', '')
            
            if summary_text:
                # Generate domain-aware QA pairs
                qa_pairs = generator.generate_domain_aware_qa_pairs(
                    summary_text, 
                    domain_name=detected_domain
                )
                
                # Convert to domain-enhanced training format
                training_examples = generator.create_domain_enhanced_training_examples(
                    qa_pairs, 
                    domain_name=detected_domain
                )
                all_training_examples.extend(training_examples)
                
                logging.info(f"Generated {len(training_examples)} domain-aware training examples from segment")
        
        # Save training data as JSONL with domain metadata
        training_data_lines = []
        for example in all_training_examples:
            # Add domain metadata to each example
            if detected_domain:
                example['domain'] = detected_domain
                example['source_file'] = blob_name
            training_data_lines.append(json.dumps(example))
        
        training_data_content = '\n'.join(training_data_lines)
        
        # Upload to training data container with domain prefix
        domain_prefix = f"{detected_domain}_" if detected_domain else ""
        output_blob_name = f"{domain_prefix}{base_name}_training_data.jsonl"
        training_container_client = blob_service_client.get_container_client(TRAINING_DATA_CONTAINER)
        
        training_container_client.upload_blob(
            name=output_blob_name,
            data=training_data_content,
            overwrite=True
        )
        
        logging.info(f"Successfully generated {len(all_training_examples)} domain-aware training examples from {blob_name}")
        
    except Exception as e:
        logging.error(f"Error processing summary blob {myblob.name}: {e}")
        raise


@app.route(route="start-finetuning", auth_level=func.AuthLevel.FUNCTION)
def start_fine_tuning(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to start domain-aware fine-tuning process with training data.
    """
    try:
        req_body = req.get_json()
        training_file_blob = req_body.get('training_file_blob')
        model_name = req_body.get('model', 'gpt-4')  # Changed default to GPT-4
        domain_name = req_body.get('domain')
        custom_suffix = req_body.get('suffix')
        
        if not training_file_blob:
            return func.HttpResponse(
                json.dumps({"error": "training_file_blob is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Download training data from blob storage
        training_container_client = blob_service_client.get_container_client(TRAINING_DATA_CONTAINER)
        blob_client = training_container_client.get_blob_client(training_file_blob)
        training_data = blob_client.download_blob().readall()
        
        # Upload training file to OpenAI
        training_file = openai_client.files.create(
            file=training_data,
            purpose='fine-tune'
        )
        
        # Start domain-aware fine-tuning job
        orchestrator = DomainAwareFineTuningOrchestrator(openai_client)
        job_id = orchestrator.create_domain_fine_tuning_job(
            training_file.id, 
            domain_name=domain_name,
            model=model_name,
            custom_suffix=custom_suffix
        )
        
        response_data = {
            "job_id": job_id,
            "training_file_id": training_file.id,
            "domain": domain_name,
            "model": model_name,
            "status": "started",
            "message": f"Domain-aware fine-tuning job initiated successfully for {domain_name or 'general'} domain"
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error starting fine-tuning: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="check-status/{job_id}", auth_level=func.AuthLevel.FUNCTION)
def check_fine_tuning_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to check fine-tuning job status with enhanced information.
    """
    try:
        job_id = req.route_params.get('job_id')
        
        if not job_id:
            return func.HttpResponse(
                json.dumps({"error": "job_id is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        orchestrator = DomainAwareFineTuningOrchestrator(openai_client)
        status = orchestrator.check_job_status(job_id)
        
        return func.HttpResponse(
            json.dumps(status),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error checking fine-tuning status: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="domains", auth_level=func.AuthLevel.FUNCTION)
def list_available_domains(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to list available domains for fine-tuning.
    """
    try:
        if not domain_manager:
            return func.HttpResponse(
                json.dumps({"error": "Domain manager not available"}),
                status_code=500,
                mimetype="application/json"
            )
        
        domains = domain_manager.list_domains()
        domain_info = []
        
        for domain_name in domains:
            context = domain_manager.get_domain_context(domain_name)
            if context:
                domain_info.append({
                    "name": domain_name,
                    "type": context.domain_type.value,
                    "focus_areas": context.focus_areas,
                    "key_concepts": context.key_concepts[:5],  # Limit for response size
                    "description": f"Fine-tuning optimized for {context.domain_name} domain"
                })
        
        return func.HttpResponse(
            json.dumps({
                "available_domains": domain_info,
                "total_domains": len(domain_info)
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error listing domains: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="domain-progress/{domain_name}", auth_level=func.AuthLevel.FUNCTION)
def get_domain_training_progress(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to get training progress for a specific domain.
    """
    try:
        domain_name = req.route_params.get('domain_name')
        
        if not domain_name:
            return func.HttpResponse(
                json.dumps({"error": "domain_name is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        orchestrator = DomainAwareFineTuningOrchestrator(openai_client)
        progress = orchestrator.get_domain_training_progress(domain_name)
        
        return func.HttpResponse(
            json.dumps(progress),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error getting domain progress: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="batch-process", auth_level=func.AuthLevel.FUNCTION)
def batch_process_training_data(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to process multiple training files with domain context.
    """
    try:        req_body = req.get_json()
        training_files = req_body.get('training_files', [])
        domain_mapping = req_body.get('domain_mapping', {})
        model_name = req_body.get('model', 'gpt-4')  # Changed default to GPT-4
        
        if not training_files:
            return func.HttpResponse(
                json.dumps({"error": "training_files list is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        orchestrator = DomainAwareFineTuningOrchestrator(openai_client)
        job_results = []
        
        for file_blob in training_files:
            # Get domain for this file
            domain_name = domain_mapping.get(file_blob)
            
            try:
                # Download training data
                training_container_client = blob_service_client.get_container_client(TRAINING_DATA_CONTAINER)
                blob_client = training_container_client.get_blob_client(file_blob)
                training_data = blob_client.download_blob().readall()
                
                # Upload to OpenAI
                training_file = openai_client.files.create(
                    file=training_data,
                    purpose='fine-tune'
                )
                
                # Start fine-tuning job
                job_id = orchestrator.create_domain_fine_tuning_job(
                    training_file.id,
                    domain_name=domain_name,
                    model=model_name
                )
                
                job_results.append({
                    "file": file_blob,
                    "domain": domain_name,
                    "job_id": job_id,
                    "training_file_id": training_file.id,
                    "status": "started"
                })
                
            except Exception as e:
                job_results.append({
                    "file": file_blob,
                    "domain": domain_name,
                    "error": str(e),
                    "status": "failed"
                })
        
        return func.HttpResponse(
            json.dumps({
                "batch_results": job_results,
                "total_files": len(training_files),
                "successful_jobs": len([r for r in job_results if r.get("status") == "started"]),
                "failed_jobs": len([r for r in job_results if r.get("status") == "failed"])
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the domain-aware fine-tuning pipeline.
    """
    health_status = {
        "status": "healthy",
        "service": "domain-aware-finetuning-pipeline",
        "timestamp": func.datetime.utcnow().isoformat(),
        "features": {
            "domain_context": domain_manager is not None,
            "azure_openai": AZURE_OPENAI_ENDPOINT is not None,
            "blob_storage": blob_service_client is not None,
            "key_vault": kv_client is not None
        }
    }
    
    if domain_manager:
        health_status["available_domains"] = domain_manager.list_domains()
    
    return func.HttpResponse(
        json.dumps(health_status),
        status_code=200,
        mimetype="application/json"
    )
