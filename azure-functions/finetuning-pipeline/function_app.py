# Azure Function for Fine-Tuning Pipeline
import logging
import json
import os
import time
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from typing import List, Dict, Any

# Initialize function app
app = func.FunctionApp()

# Configuration from environment variables
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
SUMMARIES_CONTAINER = os.environ.get("SUMMARIES_CONTAINER", "processed-summaries")
TRAINING_DATA_CONTAINER = os.environ.get("TRAINING_DATA_CONTAINER", "training-data")
MODELS_CONTAINER = os.environ.get("MODELS_CONTAINER", "fine-tuned-models")

# Initialize clients
credential = DefaultAzureCredential()
openai_client = AzureOpenAI(endpoint=AZURE_OPENAI_ENDPOINT, credential=credential)
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


class TrainingDataGenerator:
    """
    Generates training data in JSONL format for fine-tuning.
    """
    
    def __init__(self, openai_client: AzureOpenAI, deployment: str):
        self.client = openai_client
        self.deployment = deployment
    
    def generate_qa_pairs(self, summary: str, domain_context: str = "") -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from summaries using Azure OpenAI.
        """
        system_prompt = (
            "You are an expert at creating high-quality training data for language models. "
            "Generate diverse, insightful question-answer pairs based on the provided summary."
        )
        
        if domain_context:
            system_prompt += f" Domain context: {domain_context}"
        
        user_prompt = (
            f"Based on the following summary, generate 3-5 diverse question-answer pairs "
            f"that capture the key insights and concepts. Format as JSON array with 'question' and 'answer' fields:\n\n"
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
                # Fallback: create simple QA pair
                return [{
                    "question": f"What are the key insights from this summary?",
                    "answer": summary
                }]
                
        except Exception as e:
            logging.error(f"Error generating QA pairs: {e}")
            return []
    
    def create_training_examples(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert QA pairs to OpenAI fine-tuning format.
        """
        training_examples = []
        
        for qa in qa_pairs:
            if 'question' in qa and 'answer' in qa:
                training_example = {
                    "messages": [
                        {"role": "user", "content": qa['question']},
                        {"role": "assistant", "content": qa['answer']}
                    ]
                }
                training_examples.append(training_example)
        
        return training_examples


class FineTuningOrchestrator:
    """
    Orchestrates the fine-tuning process using Azure OpenAI.
    """
    
    def __init__(self, openai_client: AzureOpenAI):
        self.client = openai_client
    
    def create_fine_tuning_job(self, training_file_id: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Create a fine-tuning job with Azure OpenAI.
        """
        try:
            fine_tuning_job = self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model,
                hyperparameters={
                    "n_epochs": 3,
                    "batch_size": 1,
                    "learning_rate_multiplier": 0.1
                }
            )
            
            return fine_tuning_job.id
            
        except Exception as e:
            logging.error(f"Error creating fine-tuning job: {e}")
            raise
    
    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.
        """
        try:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            return {
                "id": job.id,
                "status": job.status,
                "model": job.fine_tuned_model,
                "created_at": job.created_at,
                "finished_at": job.finished_at
            }
        except Exception as e:
            logging.error(f"Error checking job status: {e}")
            return {"error": str(e)}


@app.blob_trigger(
    arg_name="myblob", 
    path=f"{SUMMARIES_CONTAINER}/{{name}}", 
    connection="AzureWebJobsStorage"
)
def training_data_generator(myblob: func.InputStream) -> None:
    """
    Azure Function triggered by summary blob uploads to generate training data.
    """
    logging.info(f"Processing summary blob: {myblob.name}, Size: {myblob.length} bytes")
    
    try:
        # Read the summary data
        summary_data = json.loads(myblob.read().decode('utf-8'))
        
        # Extract filename for processing
        blob_name = myblob.name.split('/')[-1]
        base_name = os.path.splitext(blob_name)[0]
        
        # Initialize training data generator
        generator = TrainingDataGenerator(openai_client, AZURE_OPENAI_DEPLOYMENT)
        
        # Process all summary segments
        all_training_examples = []
        
        for segment in summary_data.get('processed_segments', []):
            summary_text = segment.get('summary', '')
            
            if summary_text:
                # Generate QA pairs for this summary
                qa_pairs = generator.generate_qa_pairs(summary_text)
                
                # Convert to training format
                training_examples = generator.create_training_examples(qa_pairs)
                all_training_examples.extend(training_examples)
                
                logging.info(f"Generated {len(training_examples)} training examples from segment")
        
        # Save training data as JSONL
        training_data_lines = []
        for example in all_training_examples:
            training_data_lines.append(json.dumps(example))
        
        training_data_content = '\n'.join(training_data_lines)
        
        # Upload to training data container
        output_blob_name = f"{base_name}_training_data.jsonl"
        training_container_client = blob_service_client.get_container_client(TRAINING_DATA_CONTAINER)
        
        training_container_client.upload_blob(
            name=output_blob_name,
            data=training_data_content,
            overwrite=True
        )
        
        logging.info(f"Successfully generated {len(all_training_examples)} training examples from {blob_name}")
        
    except Exception as e:
        logging.error(f"Error processing summary blob {myblob.name}: {e}")
        raise


@app.route(route="start-finetuning", auth_level=func.AuthLevel.FUNCTION)
def start_fine_tuning(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to start fine-tuning process with training data.
    """
    try:
        req_body = req.get_json()
        training_file_blob = req_body.get('training_file_blob')
        model_name = req_body.get('model', 'gpt-3.5-turbo')
        
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
        
        # Start fine-tuning job
        orchestrator = FineTuningOrchestrator(openai_client)
        job_id = orchestrator.create_fine_tuning_job(training_file.id, model_name)
        
        return func.HttpResponse(
            json.dumps({
                "job_id": job_id,
                "training_file_id": training_file.id,
                "status": "started",
                "message": "Fine-tuning job initiated successfully"
            }),
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
    HTTP trigger to check fine-tuning job status.
    """
    try:
        job_id = req.route_params.get('job_id')
        
        if not job_id:
            return func.HttpResponse(
                json.dumps({"error": "job_id is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        orchestrator = FineTuningOrchestrator(openai_client)
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


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the fine-tuning pipeline.
    """
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "finetuning-pipeline",
            "timestamp": func.datetime.utcnow().isoformat()
        }),
        status_code=200,
        mimetype="application/json"
    )
