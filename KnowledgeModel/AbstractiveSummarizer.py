import os
import time
from typing import Optional
import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE


class AbstractiveSummarizer:
    """
    A class to perform abstractive summarization using Claude Sonnet 4.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude client for summarization.

        :param api_key: The Anthropic API key. If None, uses config value.
        :param model: The Claude model to use. If None, uses config value.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or CLAUDE_MODEL
        
        if not self.api_key or self.api_key == "<your-anthropic-api-key>":
            raise ValueError("Please set your Anthropic API key in config.py or pass it as a parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def summarize_text(self, input_text: str, max_tokens: int = None, 
                      temperature: float = None, domain_context: str = "") -> str:
        """
        Generate a summary for the given input text using Claude Sonnet 4.

        :param input_text: The text to be summarized.
        :param max_tokens: The maximum number of tokens for the summary.
        :param temperature: The temperature for response generation.
        :param domain_context: Additional domain-specific context for focused summarization.
        :return: The generated summary as a string.
        """
        max_tokens = max_tokens or CLAUDE_MAX_TOKENS
        temperature = temperature or CLAUDE_TEMPERATURE
        
        # Construct the prompt for Claude
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
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except anthropic.APIError as e:
            print(f"Claude API error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during summarization: {e}")
            raise

    def extract_key_insights(self, input_text: str, domain_context: str = "") -> str:
        """
        Extract key insights and actionable points from the input text.

        :param input_text: The text to extract insights from.
        :param domain_context: Domain-specific context for focused extraction.
        :return: Key insights formatted as bullet points.
        """
        system_prompt = (
            "You are an expert at extracting key insights and actionable information "
            "from text. Focus on identifying the most valuable and practical insights."
        )
        
        if domain_context:
            system_prompt += f" Context: {domain_context}"
        
        user_prompt = (
            f"Extract the key insights and actionable points from the following text. "
            f"Format your response as clear, concise bullet points:\n\n"
            f"{input_text}\n\n"
            f"Key Insights:"
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=CLAUDE_TEMPERATURE,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except anthropic.APIError as e:
            print(f"Claude API error: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during insight extraction: {e}")
            raise