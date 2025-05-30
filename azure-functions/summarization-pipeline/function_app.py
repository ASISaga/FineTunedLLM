# Azure Function for Summarization Pipeline
import logging
import json
import os
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import anthropic
from typing import Optional

# Initialize function app
app = func.FunctionApp()

# Configuration from environment variables
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
INPUT_CONTAINER = os.environ.get("INPUT_CONTAINER", "raw-documents")
OUTPUT_CONTAINER = os.environ.get("OUTPUT_CONTAINER", "processed-summaries")

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


class ClaudeSummarizer:
    """
    Claude-based text summarization for serverless deployment.
    """
    
    def __init__(self, api_key: str, model: str = CLAUDE_MODEL):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def summarize_text(self, input_text: str, domain_context: str = "") -> str:
        """
        Generate a summary using Claude Sonnet 4.
        """
        system_prompt = (
            "You are an expert at creating concise, insightful summaries that capture "
            "the key concepts, insights, and actionable information from text. "
            "Focus on extracting the most valuable and relevant information."
        )
        
        if domain_context:
            system_prompt += f" Context: {domain_context}"
        
        user_prompt = (
            f"Please provide a comprehensive yet concise summary of the following text. "
            f"Focus on key insights, main concepts, and actionable information:\n\n"
            f"{input_text}\n\n"
            f"Summary:"
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            raise


@app.blob_trigger(
    arg_name="myblob", 
    path=f"{INPUT_CONTAINER}/{{name}}", 
    connection="AzureWebJobsStorage"
)
def summarization_trigger(myblob: func.InputStream) -> None:
    """
    Azure Function triggered by blob uploads for text summarization.
    """
    logging.info(f"Processing blob: {myblob.name}, Size: {myblob.length} bytes")
    
    try:
        # Read the blob content
        blob_content = myblob.read().decode('utf-8')
        
        # Extract filename for processing
        blob_name = myblob.name.split('/')[-1]
        base_name = os.path.splitext(blob_name)[0]
        
        # Initialize summarizer
        summarizer = ClaudeSummarizer(ANTHROPIC_API_KEY)
        
        # Process text in paragraphs
        paragraphs = [p.strip() for p in blob_content.split('\n\n') if p.strip()]
        processed_segments = []
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.split()) < 50:  # Skip short paragraphs
                continue
                
            try:
                # Generate summary for this paragraph
                summary = summarizer.summarize_text(paragraph)
                
                segment_data = {
                    "segment_id": f"{base_name}_segment_{i}",
                    "original_text": paragraph,
                    "summary": summary,
                    "word_count": len(paragraph.split()),
                    "source_file": blob_name
                }
                
                processed_segments.append(segment_data)
                logging.info(f"Processed segment {i+1}/{len(paragraphs)}")
                
            except Exception as e:
                logging.error(f"Error processing segment {i}: {e}")
                continue
        
        # Save processed data to output container
        output_data = {
            "source_document": blob_name,
            "processed_segments": processed_segments,
            "total_segments": len(processed_segments),
            "processing_timestamp": func.datetime.utcnow().isoformat()
        }
        
        # Upload to output container
        output_blob_name = f"{base_name}_summaries.json"
        output_container_client = blob_service_client.get_container_client(OUTPUT_CONTAINER)
        
        output_container_client.upload_blob(
            name=output_blob_name,
            data=json.dumps(output_data, indent=2),
            overwrite=True
        )
        
        logging.info(f"Successfully processed {len(processed_segments)} segments from {blob_name}")
        
    except Exception as e:
        logging.error(f"Error processing blob {myblob.name}: {e}")
        raise


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the summarization pipeline.
    """
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "summarization-pipeline",
            "timestamp": func.datetime.utcnow().isoformat()
        }),
        status_code=200,
        mimetype="application/json"
    )


@app.route(route="process", auth_level=func.AuthLevel.FUNCTION)
def manual_process(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manual trigger endpoint for processing specific documents.
    """
    try:
        req_body = req.get_json()
        blob_name = req_body.get('blob_name')
        domain_context = req_body.get('domain_context', '')
        
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
        summarizer = ClaudeSummarizer(ANTHROPIC_API_KEY)
        
        # Process and return summary
        summary = summarizer.summarize_text(blob_content, domain_context)
        
        return func.HttpResponse(
            json.dumps({
                "blob_name": blob_name,
                "summary": summary,
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
