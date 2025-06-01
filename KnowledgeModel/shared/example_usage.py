"""
Example usage of DomainAwareTrainer with Amazon Bedrock and Azure OpenAI integration.
This script demonstrates the complete pipeline: Claude Sonnet 4 for JSONL generation 
via Amazon Bedrock and OpenAI GPT-4.1 for fine-tuning.
"""

import os
import logging
from typing import List
from DomainAwareTrainerBedrock import DomainAwareTrainer, FineTuningConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Example of running the complete training pipeline.
    """
    
    # Configuration
    openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "your-aws-access-key")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "your-aws-secret-key")
    aws_region = "us-east-1"  # or your preferred region
    
    # Sample domain-specific documents for training
    sample_documents = [
        """
        Machine learning is a subset of artificial intelligence that enables computers to learn 
        and improve from experience without being explicitly programmed. It focuses on developing 
        algorithms that can access data and use it to learn for themselves.
        """,
        """
        Deep learning is a machine learning technique that teaches computers to do what comes 
        naturally to humans: learn by example. Deep learning is a key technology behind driverless 
        cars, enabling them to recognize a stop sign, or to distinguish a pedestrian from a lamppost.
        """,
        """
        Neural networks are computing systems inspired by biological neural networks. They consist 
        of interconnected nodes (neurons) that process information using a connectionist approach 
        to computation. These networks can learn and model complex relationships between inputs and outputs.
        """,
        """
        Natural language processing (NLP) is a branch of artificial intelligence that helps computers 
        understand, interpret and manipulate human language. NLP draws from many disciplines, including 
        computer science and computational linguistics, in its pursuit to fill the gap between human 
        communication and computer understanding.
        """,
        """
        Computer vision is a field of artificial intelligence that trains computers to interpret 
        and understand the visual world. Using digital images from cameras and videos and deep 
        learning models, machines can accurately identify and classify objects.
        """
    ]
    
    try:
        # Initialize the trainer with both OpenAI and AWS credentials
        trainer = DomainAwareTrainer(
            api_key=openai_api_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region
        )
        
        # Configure the fine-tuning job
        config = FineTuningConfig(
            domain_name="machine_learning",
            model="gpt-4-turbo-2024-04-09",  # GPT-4.1 equivalent
            n_epochs=3,
            batch_size=4,
            learning_rate=1e-5,
            validation_split=0.2
        )
        
        logger.info("Starting complete training pipeline...")
        
        # Run the complete pipeline: Bedrock + OpenAI
        job_id = trainer.run_complete_training_pipeline(
            text_documents=sample_documents,
            domain_name="machine_learning",
            config=config
        )
        
        logger.info(f"Training pipeline started successfully! Job ID: {job_id}")
        
        # Monitor the job status
        status = trainer.get_job_status(job_id)
        logger.info(f"Initial job status: {status['status']}")
        
        # Example of monitoring job progress (in production, you'd poll this)
        logger.info("To monitor job progress, use:")
        logger.info(f"status = trainer.get_job_status('{job_id}')")
        
        return job_id
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def demonstrate_bedrock_only():
    """
    Demonstrate just the Bedrock JSONL generation without fine-tuning.
    """
    
    # Configuration
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "your-aws-access-key")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "your-aws-secret-key")
    aws_region = "us-east-1"
    
    sample_documents = [
        "Artificial intelligence is transforming healthcare through diagnostic tools and treatment recommendations.",
        "Machine learning algorithms can analyze medical images to detect diseases earlier and more accurately than traditional methods."
    ]
    
    try:
        # Initialize trainer with minimal OpenAI config for Bedrock-only usage
        trainer = DomainAwareTrainer(
            api_key="dummy-key",  # Not used for Bedrock-only
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region
        )
        
        config = FineTuningConfig(domain_name="healthcare_ai")
        
        # Generate training data using only Bedrock
        training_file, validation_file = trainer.generate_training_data_with_bedrock(
            text_documents=sample_documents,
            domain_name="healthcare_ai",
            config=config
        )
        
        logger.info(f"Generated training data files:")
        logger.info(f"Training: {training_file}")
        logger.info(f"Validation: {validation_file}")
        
        return training_file, validation_file
        
    except Exception as e:
        logger.error(f"Error in Bedrock generation: {str(e)}")
        raise

if __name__ == "__main__":
    print("DomainAwareTrainer with Amazon Bedrock and Azure OpenAI")
    print("=" * 60)
    
    # Set environment variables or modify the values above
    print("Make sure to set the following environment variables:")
    print("- OPENAI_API_KEY: Your OpenAI API key")
    print("- AWS_ACCESS_KEY_ID: Your AWS access key")
    print("- AWS_SECRET_ACCESS_KEY: Your AWS secret key")
    print()
    
    # Uncomment the function you want to run:
    
    # Full pipeline (Bedrock + OpenAI fine-tuning)
    # main()
    
    # Bedrock-only JSONL generation
    # demonstrate_bedrock_only()
    
    print("Uncomment the desired function in the script to run the example.")
