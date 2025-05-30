# Azure Function for Summarization Pipeline with Domain Context Support
import logging
import json
import os
import sys
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import anthropic
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add the KnowledgeModel directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'KnowledgeModel'))

try:
    from DomainContextManager import DomainContextManager
except ImportError:
    # Fallback for when modules are not available
    DomainContextManager = None
    logging.warning("DomainContextManager not available, using basic functionality")

# Initialize function app
app = func.FunctionApp()

# Configuration from environment variables
KEY_VAULT_URL = os.environ.get("KEY_VAULT_URL")
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME")
INPUT_CONTAINER = os.environ.get("INPUT_CONTAINER", "input-documents")
SUMMARY_CONTAINER = os.environ.get("SUMMARY_CONTAINER", "summaries")

# Initialize Azure clients
credential = DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID)

# Initialize Key Vault client for secret retrieval
if KEY_VAULT_URL:
    kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
else:
    kv_client = None

# Initialize storage client
if STORAGE_ACCOUNT_NAME:
    blob_service_client = BlobServiceClient(
        account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=credential
    )
else:
    blob_service_client = None

# Initialize domain manager
domain_manager = DomainContextManager() if DomainContextManager else None


def get_secret(secret_name: str) -> Optional[str]:
    """Retrieve secret from Azure Key Vault"""
    if not kv_client:
        return os.environ.get(secret_name.upper().replace('-', '_'))
    
    try:
        secret = kv_client.get_secret(secret_name)
        return secret.value
    except Exception as e:
        logging.error(f"Failed to retrieve secret {secret_name}: {str(e)}")
        return None


class DomainAwareSummarizer:
    """
    Domain-aware text summarization for serverless deployment with Claude integration.
    """
    
    def __init__(self):
        self.anthropic_api_key = get_secret("anthropic-api-key")
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key not found in Key Vault or environment")
        
        self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.model = CLAUDE_MODEL
        self.domain_manager = domain_manager
    
    def summarize_with_domain_context(self, text: str, domain_name: str = "general", 
                                    focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Summarize text with domain-specific context.
        
        :param text: Input text to summarize
        :param domain_name: Domain context name
        :param focus_areas: Specific focus areas within the domain
        :return: Structured summary with metadata
        """
        try:
            # Build domain-aware prompt
            system_prompt = self._build_domain_prompt(domain_name)
            
            focus_instruction = ""
            if focus_areas:
                focus_instruction = f"\nSpecifically focus on: {', '.join(focus_areas)}"
            
            user_prompt = f"""
            Analyze the following text and provide a comprehensive summary with these sections:
            
            1. EXECUTIVE SUMMARY: Main points and key takeaways
            2. DOMAIN INSIGHTS: Relevant domain-specific insights and implications
            3. KEY CONCEPTS: Important concepts and terminology identified
            4. ACTIONABLE ITEMS: Specific recommendations or next steps
            5. TECHNICAL DETAILS: Relevant technical information (if applicable)
            
            {focus_instruction}
            
            Text to analyze:
            {text}
            
            Please structure your response clearly with the above sections:
            """
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            summary_text = response.content[0].text.strip()
            
            return {
                "summary": summary_text,
                "domain": domain_name,
                "focus_areas": focus_areas,
                "model": self.model,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "input_length": len(text),
                    "summary_length": len(summary_text),
                    "tokens_used": getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0
                }
            }
            
        except Exception as e:
            logging.error(f"Summarization error: {str(e)}")
            raise
    
    def _build_domain_prompt(self, domain_name: str) -> str:
        """Build domain-specific system prompt"""
        base_prompt = (
            "You are an expert analyst capable of creating insightful, comprehensive summaries "
            "that extract key information, concepts, and actionable insights from text."
        )
        
        if self.domain_manager and domain_name != "general":
            domain_context = self.domain_manager.get_domain_context(domain_name)
            if domain_context:
                enhanced_prompt = f"""
                {base_prompt}
                
                DOMAIN EXPERTISE: {domain_context.domain_name}
                {domain_context.description}
                
                KEY FOCUS AREAS: {', '.join(domain_context.focus_areas)}
                IMPORTANT CONCEPTS: {', '.join(domain_context.key_concepts)}
                
                DOMAIN-SPECIFIC INSTRUCTIONS:
                {domain_context.summarization_instructions}
                
                Apply your domain expertise to provide the most valuable insights.
                """
                return enhanced_prompt
        
        return base_prompt
    
    def generate_training_examples(self, text: str, domain_name: str, 
                                 num_examples: int = 3) -> List[Dict[str, str]]:
        """Generate training examples from text for fine-tuning"""
        try:
            system_prompt = self._build_domain_prompt(domain_name)
            
            user_prompt = f"""
            Based on the following text, generate {num_examples} high-quality training examples
            for fine-tuning a domain-specific model. Each example should consist of a realistic
            question/prompt and a comprehensive expert response.
            
            Format as JSON array with objects containing "prompt" and "response" fields.
            
            Source text:
            {text}
            """
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Parse the response to extract training examples
            response_text = response.content[0].text.strip()
            
            # Try to extract JSON or parse manually
            try:
                examples = json.loads(response_text)
                return examples if isinstance(examples, list) else []
            except json.JSONDecodeError:
                # Fallback parsing if JSON is not properly formatted
                return self._parse_examples_from_text(response_text)
                
        except Exception as e:
            logging.error(f"Error generating training examples: {str(e)}")
            return []
    
    def _parse_examples_from_text(self, text: str) -> List[Dict[str, str]]:
        """Fallback method to parse training examples from text"""
        examples = []
        # Simple parsing logic - this can be enhanced based on Claude's output format
        sections = text.split("EXAMPLE")
        
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            prompt = ""
            response = ""
            
            for line in lines:
                if "PROMPT:" in line or "Question:" in line:
                    prompt = line.split(":", 1)[1].strip()
                elif "RESPONSE:" in line or "Answer:" in line:
                    response = line.split(":", 1)[1].strip()
            
            if prompt and response:
                examples.append({"prompt": prompt, "response": response})
        
        return examples


# Initialize the summarizer globally
summarizer = None

def get_summarizer():
    """Get or create the summarizer instance"""
    global summarizer
    if summarizer is None:
        summarizer = DomainAwareSummarizer()
    return summarizer


@app.blob_trigger(
    arg_name="myblob", 
    path=f"{INPUT_CONTAINER}/{{name}}", 
    connection="AzureWebJobsStorage"
)
def summarization_trigger(myblob: func.InputStream) -> None:
    """
    Azure Function triggered by blob uploads for domain-aware text summarization.
    """
    logging.info(f"Processing blob: {myblob.name}, Size: {myblob.length} bytes")
    
    try:
        # Read the blob content
        blob_content = myblob.read().decode('utf-8')
        
        # Extract filename and metadata
        blob_name = myblob.name.split('/')[-1]
        base_name = os.path.splitext(blob_name)[0]
        
        # Try to extract domain from filename or use default
        domain_name = "general"
        if "_" in base_name:
            potential_domain = base_name.split("_")[0].lower()
            if domain_manager and domain_manager.get_domain_context(potential_domain):
                domain_name = potential_domain
        
        # Initialize summarizer
        text_summarizer = get_summarizer()
        
        # Process text with domain context
        summary_result = text_summarizer.summarize_with_domain_context(
            blob_content, 
            domain_name=domain_name
        )
        
        # Generate training examples for fine-tuning
        training_examples = text_summarizer.generate_training_examples(
            blob_content, 
            domain_name=domain_name,
            num_examples=3
        )
        
        # Prepare output data
        output_data = {
            "source_document": blob_name,
            "domain": domain_name,
            "summary_result": summary_result,
            "training_examples": training_examples,
            "processing_timestamp": datetime.utcnow().isoformat(),
            "processing_metadata": {
                "function_name": "summarization_trigger",
                "model_used": CLAUDE_MODEL,
                "domain_manager_available": domain_manager is not None
            }
        }
        
        # Save to summary container
        output_blob_name = f"{base_name}_domain_summary.json"
        summary_container_client = blob_service_client.get_container_client(SUMMARY_CONTAINER)
        
        summary_container_client.upload_blob(
            name=output_blob_name,
            data=json.dumps(output_data, indent=2),
            overwrite=True
        )
        
        # Also save training examples separately for fine-tuning pipeline
        if training_examples:
            training_blob_name = f"{base_name}_training_examples.jsonl"
            training_data = "\n".join([json.dumps(example) for example in training_examples])
            
            training_container_client = blob_service_client.get_container_client("training-data")
            training_container_client.upload_blob(
                name=training_blob_name,
                data=training_data,
                overwrite=True
            )
        
        logging.info(f"Successfully processed {blob_name} with domain: {domain_name}")
        
    except Exception as e:
        logging.error(f"Error processing blob {myblob.name}: {e}")
        raise


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the summarization pipeline.
    """
    try:
        # Check if we can initialize the summarizer
        test_summarizer = get_summarizer()
        
        health_data = {
            "status": "healthy",
            "service": "summarization-pipeline",
            "timestamp": datetime.utcnow().isoformat(),
            "model": CLAUDE_MODEL,
            "domain_manager_available": domain_manager is not None,
            "available_domains": domain_manager.list_available_domains() if domain_manager else []
        }
        
        return func.HttpResponse(
            json.dumps(health_data),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="process-document", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def manual_process_document(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manual trigger endpoint for processing specific documents with domain context.
    """
    try:
        req_body = req.get_json()
        blob_name = req_body.get('blob_name')
        domain_name = req_body.get('domain_name', 'general')
        focus_areas = req_body.get('focus_areas', [])
        
        if not blob_name:
            return func.HttpResponse(
                json.dumps({"error": "blob_name is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Process the specified blob
        input_container_client = blob_service_client.get_container_client(INPUT_CONTAINER)
        blob_client = input_container_client.get_blob_client(blob_name)
        blob_content = blob_client.download_blob().readall().decode('utf-8')
        
        # Initialize summarizer with domain context
        text_summarizer = get_summarizer()
        
        # Process and return summary
        summary_result = text_summarizer.summarize_with_domain_context(
            blob_content, 
            domain_name=domain_name, 
            focus_areas=focus_areas
        )
        
        return func.HttpResponse(
            json.dumps({
                "blob_name": blob_name,
                "summary_result": summary_result,
                "status": "completed"
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in manual processing: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="domains", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def list_available_domains(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get list of available domain contexts.
    """
    try:
        if not domain_manager:
            return func.HttpResponse(
                json.dumps({"domains": [], "message": "Domain manager not available"}),
                status_code=200,
                mimetype="application/json"
            )
        
        domains = domain_manager.list_available_domains()
        domain_details = {}
        
        for domain_name in domains:
            context = domain_manager.get_domain_context(domain_name)
            if context:
                domain_details[domain_name] = {
                    "description": context.description,
                    "focus_areas": context.focus_areas,
                    "key_concepts": context.key_concepts[:5]  # Limit for response size
                }
        
        return func.HttpResponse(
            json.dumps({
                "domains": domains,
                "domain_details": domain_details,
                "total_count": len(domains)
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


@app.route(route="process-batch", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def batch_process(req: func.HttpRequest) -> func.HttpResponse:
    """
    Batch processing endpoint for multiple documents with domain context.
    """
    try:
        req_body = req.get_json()
        blob_names = req_body.get('blob_names', [])
        domain_name = req_body.get('domain_name', 'general')
        focus_areas = req_body.get('focus_areas', [])
        
        if not blob_names:
            return func.HttpResponse(
                json.dumps({"error": "blob_names array is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        results = []
        text_summarizer = get_summarizer()
        input_container_client = blob_service_client.get_container_client(INPUT_CONTAINER)
        
        for blob_name in blob_names:
            try:
                blob_client = input_container_client.get_blob_client(blob_name)
                blob_content = blob_client.download_blob().readall().decode('utf-8')
                
                summary_result = text_summarizer.summarize_with_domain_context(
                    blob_content,
                    domain_name=domain_name,
                    focus_areas=focus_areas
                )
                
                results.append({
                    "blob_name": blob_name,
                    "status": "success",
                    "summary_result": summary_result
                })
                
            except Exception as e:
                logging.error(f"Error processing {blob_name}: {e}")
                results.append({
                    "blob_name": blob_name,
                    "status": "error",
                    "error": str(e)
                })
        
        return func.HttpResponse(
            json.dumps({
                "batch_results": results,
                "total_processed": len(results),
                "successful": len([r for r in results if r["status"] == "success"]),
                "failed": len([r for r in results if r["status"] == "error"])
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
