import os
import time
import logging
from typing import Optional, Dict, Any
import anthropic
from ..shared.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_TEMPERATURE
from ..domain_knowledge_system.DomainContextManager import DomainContextManager

logger = logging.getLogger(__name__)


class AbstractiveSummarizer:
    """
    A class to perform abstractive summarization using Claude Sonnet 4 with domain-specific context.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude client for summarization.

        :param api_key: The Anthropic API key. If None, uses config value.
        :param model: The Claude model to use. If None, uses config value.
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or CLAUDE_MODEL
        self.domain_manager = DomainContextManager()
        
        if not self.api_key or self.api_key == "<your-anthropic-api-key>":
            raise ValueError("Please set your Anthropic API key in config.py or pass it as a parameter")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Initialized AbstractiveSummarizer with model: {self.model}")

    def summarize_text(self, input_text: str, max_tokens: int = None, 
                      temperature: float = None, domain_context: str = "", 
                      domain_name: Optional[str] = None) -> str:
        """
        Generate a summary for the given input text using Claude Sonnet 4 with domain context.

        :param input_text: The text to be summarized.
        :param max_tokens: The maximum number of tokens for the summary.
        :param temperature: The temperature for response generation.
        :param domain_context: Additional domain-specific context for focused summarization.
        :param domain_name: Name of the domain for automatic context enhancement.
        :return: The generated summary as a string.
        """
        max_tokens = max_tokens or CLAUDE_MAX_TOKENS
        temperature = temperature or CLAUDE_TEMPERATURE
        
        # Build system prompt with domain awareness
        system_prompt = self._build_domain_aware_prompt(domain_context, domain_name)
        
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
                ]            )
            
            return response.content[0].text.strip()
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {e}")
            raise

    def extract_key_insights(self, input_text: str, domain_context: str = "", 
                           domain_name: Optional[str] = None) -> str:
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

    def _build_domain_aware_prompt(self, domain_context: str, domain_name: Optional[str]) -> str:
        """
        Build a domain-aware system prompt for Claude.
        
        :param domain_context: Additional domain-specific context.
        :param domain_name: Name of the domain for automatic context enhancement.
        :return: Enhanced system prompt with domain awareness.
        """
        base_prompt = (
            "You are an expert at creating concise, insightful summaries that capture "
            "the key concepts, insights, and actionable information from text. "
            "Focus on extracting the most valuable and relevant information."
        )
        
        # If domain name is provided, use domain manager for enhanced context
        if domain_name:
            enhanced_prompt = self.domain_manager.get_summarization_prompt(domain_name, base_prompt)
            logger.info(f"Using domain-specific prompt for: {domain_name}")
            return enhanced_prompt
        
        # Otherwise, use basic domain context if provided
        if domain_context:
            return f"{base_prompt}\n\nDOMAIN CONTEXT: {domain_context}"
        
        return base_prompt

    def summarize_with_domain_focus(self, input_text: str, domain_name: str, 
                                   focus_areas: Optional[list] = None, 
                                   max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate a domain-focused summary with structured output.
        
        :param input_text: The text to be summarized.
        :param domain_name: Name of the domain for context.
        :param focus_areas: Specific areas to focus on within the domain.
        :param max_tokens: Maximum tokens for the response.
        :return: Structured summary with domain insights.
        """
        domain_context = self.domain_manager.get_domain_context(domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        max_tokens = max_tokens or CLAUDE_MAX_TOKENS
        
        # Build focused prompt
        system_prompt = self._build_domain_aware_prompt("", domain_name)
        
        focus_instruction = ""
        if focus_areas:
            focus_instruction = f"\nSpecifically focus on these areas: {', '.join(focus_areas)}"
        
        user_prompt = f"""
        Please analyze the following text from a {domain_context.domain_name} perspective and provide:
        
        1. EXECUTIVE SUMMARY: A concise overview of the main points
        2. KEY INSIGHTS: The most important domain-specific insights
        3. TECHNICAL DETAILS: Relevant technical or specialized information
        4. ACTIONABLE RECOMMENDATIONS: Specific next steps or applications
        5. DOMAIN TERMINOLOGY: Important domain-specific terms and concepts mentioned
        
        {focus_instruction}
        
        Text to analyze:
        {input_text}
        
        Please structure your response clearly with these sections:
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=CLAUDE_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            summary_text = response.content[0].text.strip()
            
            return {
                "domain": domain_name,
                "summary": summary_text,
                "focus_areas": focus_areas or domain_context.focus_areas,
                "key_concepts": domain_context.key_concepts,
                "metadata": {
                    "model": self.model,
                    "tokens_used": response.usage.input_tokens if hasattr(response, 'usage') else None,
                    "domain_type": domain_context.domain_type.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error in domain-focused summarization: {str(e)}")
            raise

    def generate_training_examples(self, input_text: str, domain_name: str, 
                                 num_examples: int = 3) -> list:
        """
        Generate training examples for fine-tuning based on domain context.
        
        :param input_text: Source text for generating examples.
        :param domain_name: Domain context for training examples.
        :param num_examples: Number of training examples to generate.
        :return: List of training examples in prompt-response format.
        """
        domain_context = self.domain_manager.get_domain_context(domain_name)
        if not domain_context:
            raise ValueError(f"Unknown domain: {domain_name}")
        
        system_prompt = f"""
        You are an expert in {domain_context.domain_name}. Generate {num_examples} high-quality 
        training examples based on the provided text. Each example should consist of a 
        realistic question or prompt that someone might ask about this domain, along with 
        a comprehensive, expert-level response.
        
        Focus on: {', '.join(domain_context.focus_areas)}
        Key concepts: {', '.join(domain_context.key_concepts)}
        
        Format each example as:
        EXAMPLE N:
        PROMPT: [question or request]
        RESPONSE: [detailed, expert response]
        """
        
        user_prompt = f"""
        Based on this {domain_context.domain_name} content, generate {num_examples} training examples:
        
        {input_text}
        
        Make sure the examples demonstrate deep domain expertise and follow best practices.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS * 2,  # More tokens for multiple examples
                temperature=0.7,  # Slightly higher temperature for variety
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            examples_text = response.content[0].text.strip()
            
            # Parse the examples into structured format
            examples = self._parse_training_examples(examples_text)
            
            return examples
            
        except Exception as e:
            logger.error(f"Error generating training examples: {str(e)}")
            raise

    def _parse_training_examples(self, examples_text: str) -> list:
        """
        Parse the generated training examples into structured format.
        
        :param examples_text: Raw text containing training examples.
        :return: List of structured training examples.
        """
        examples = []
        sections = examples_text.split("EXAMPLE")
        
        for section in sections[1:]:  # Skip the first empty section
            lines = section.strip().split('\n')
            prompt = ""
            response = ""
            capture_response = False
            
            for line in lines:
                if line.startswith("PROMPT:"):
                    prompt = line.replace("PROMPT:", "").strip()
                elif line.startswith("RESPONSE:"):
                    response = line.replace("RESPONSE:", "").strip()
                    capture_response = True
                elif capture_response and line.strip():
                    response += " " + line.strip()
            
            if prompt and response:
                examples.append({
                    "prompt": prompt,
                    "response": response
                })
        
        return examples