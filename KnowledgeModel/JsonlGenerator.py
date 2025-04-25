# Import necessary libraries
import os
import json
import time
from azure.identity import DefaultAzureCredential
from azure.ai.openai import OpenAIClient
from azure.storage.blob import BlobServiceClient

# Azure environment setup
AZURE_OPENAI_ENDPOINT = "https://<your-openai-endpoint>.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT = "<your-deployment-name>" # Azure OpenAI deployment name
AZURE_STORAGE_CONNECTION_STRING = "<your-storage-connection-string>"
CONTAINER_NAME = "essays"

# Initialize Azure clients
credential = DefaultAzureCredential()
openai_client = OpenAIClient(endpoint=AZURE_OPENAI_ENDPOINT, credential=credential)
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)


def extract_insights(text_segment):
    """
    Uses Azure OpenAI to extract key investment insights from a text segment.
    """
    prompt = (
        "Read the following excerpt from a Warren Buffett essay and extract the key "
        "investment insights in concise bullet points:\n\n"
        f"{text_segment}\n\n"
        "Insights:"
    )
    try:
        response = openai_client.completions.create(
            deployment_id=AZURE_OPENAI_DEPLOYMENT,
            prompt=prompt,
            max_tokens=150,
            temperature=0.5,
        )
    except Exception as e:
        print("Error during insight extraction:", e)
        return ""
    
    return response.choices[0].text.strip()


def generate_qa_pair(insight):
    """
    Uses Azure OpenAI to generate a question and answer pair based on a given insight.
    """
    prompt = (
        "Based on the following investment insight from Warren Buffett, "
        "generate a question and a concise answer that captures this idea.\n\n"
        f"Insight: {insight}\n\n"
        "Format:\n"
        "Question: <your question here>\n"
        "Answer: <your answer here>"
    )
    try:
        response = openai_client.completions.create(
            deployment_id=AZURE_OPENAI_DEPLOYMENT,
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
        )
    except Exception as e:
        print("Error during QA pair generation:", e)
        return {"prompt": insight, "completion": ""}
    
    output = response.choices[0].text.strip()
    
    # Attempt to parse the output into a question and answer.
    question, answer = None, None
    for line in output.splitlines():
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
    
    if not question:
        question = insight[:100] + "..."
    if not answer:
        answer = output

    return {"prompt": question, "completion": answer}


def process_text_segment(segment, min_word_count=50):
    """
    Processes a text segment by extracting insights and generating a QA pair if the segment is sufficiently long.
    """
    if len(segment.split()) < min_word_count:
        return None
    
    insight = extract_insights(segment)
    if not insight:
        return None

    qa_pair = generate_qa_pair(insight)
    return qa_pair


def process_blob(blob_name):
    """
    Processes an essay file stored in Azure Blob Storage.
    """
    blob_client = container_client.get_blob_client(blob_name)
    essay_content = blob_client.download_blob().readall().decode("utf-8")

    segments = essay_content.split("\n\n")
    qa_pairs = []

    for segment in segments:
        segment = segment.strip()
        if segment:
            qa = process_text_segment(segment)
            if qa:
                qa_pairs.append(qa)
                time.sleep(1) # Rate limit to avoid overloading the API

    return qa_pairs


def main():
    output_file = "data.jsonl" # Output JSONL file

    # List all blobs (essay files) in the container
    blobs = container_client.list_blobs()
    all_pairs = []

    for blob in blobs:
        print(f"Processing blob: {blob.name}")
        pairs = process_blob(blob.name)
        all_pairs.extend(pairs)

    # Save prompt-completion pairs to a JSONL file
    with open(output_file, "w", encoding="utf8") as out_file:
        for pair in all_pairs:
            out_file.write(json.dumps(pair) + "\n")
    
    print(f"Generated {len(all_pairs)} prompt-completion pairs in {output_file}")


# Key Modifications for Azure AI Foundry:
# 1. Azure OpenAI Integration:
#    - The code uses the azure-ai-openai SDK to connect to Azure OpenAI resources.
#    - Replace '<your-openai-endpoint>' and '<your-deployment-name>' with the endpoint URL and deployment name from your Azure OpenAI instance.
#
# 2. Azure Blob Storage Integration:
#    - Essays are assumed to be stored as .txt files in an Azure Blob Storage container ('essays'). The BlobServiceClient is used to list, download, and process blobs.
#
# 3. Default Azure Credential:
#    - The script uses DefaultAzureCredential, which works seamlessly with Azure AI Foundry's managed identity.
#
# 4. Run on Azure AI Foundry:
#    - Deploy the script as part of an AI Foundry module or job. Ensure that the necessary Azure services (OpenAI, Blob Storage) and permissions are configured.

# Steps to Deploy on Azure AI Foundry:
# 1. Prepare the Script:
#    - Save the script as 'process_essays.py' or any suitable name.
#    - Include any required dependencies (e.g., azure-ai-openai, azure-storage-blob) in your environment or requirements.txt.
#
# 2. Set Up Blob Storage:
#    - Upload all essay .txt files to a container (e.g., 'essays') in Azure Blob Storage.
#
# 3. Provision Resources:
#    - Ensure your Azure AI Foundry workspace has access to:
#      - An Azure OpenAI deployment.
#      - Azure Blob Storage (with appropriate permissions).
#
# 4. Execute on Foundry:
#    - Upload the script to the compute environment.
#    - Run the script as a job or workflow module within Azure AI Foundry.
#
# 5. Review Output:
#    - The final 'data.jsonl' file will be generated and can be downloaded from your compute environment or processed further.