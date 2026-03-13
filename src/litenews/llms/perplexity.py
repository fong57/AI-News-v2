"""Perplexity LLM integration for LiteNews AI.

This module provides integration with Perplexity's API using langchain-perplexity.
Perplexity models have built-in web search capabilities, making them excellent
for news research and fact-checking.
"""

from langchain_core.language_models import BaseChatModel
from langchain_perplexity import ChatPerplexity

from litenews.config.llm_config import LLMConfig
from litenews.llms.base import BaseLLM


class PerplexityLLM(BaseLLM):
    """Perplexity LLM wrapper with web search capabilities.
    
    Perplexity models (sonar, sonar-pro, sonar-reasoning) have built-in
    access to the web, making them ideal for news research tasks.
    
    Available models:
        - sonar: Fast, general-purpose model
        - sonar-pro: More capable model with better reasoning
        - sonar-reasoning: Best for complex reasoning tasks
    """
    
    def _create_model(self) -> BaseChatModel:
        """Create the ChatPerplexity model instance.
        
        Returns:
            BaseChatModel: The ChatPerplexity model.
        """
        return ChatPerplexity(
            model=self.config.model,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
    
    def invoke_with_search(
        self,
        query: str,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: str | None = None,
    ):
        """Invoke Perplexity with search parameters.
        
        Args:
            query: The query to process.
            search_domain_filter: Optional list of domains to restrict search.
            search_recency_filter: Optional recency filter (e.g., "day", "week", "month").
            
        Returns:
            The model's response with web search context.
        """
        model = ChatPerplexity(
            model=self.config.model,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            search_domain_filter=search_domain_filter,
            search_recency_filter=search_recency_filter,
        )
        return model.invoke(query)


def test_perplexity_connection(api_key: str, model: str = "sonar") -> dict:
    """Test Perplexity API connection.
    
    Args:
        api_key: Perplexity API key.
        model: Model name to test.
        
    Returns:
        dict: Test result with status and response.
    """
    try:
        llm = ChatPerplexity(
            model=model,
            api_key=api_key,
            max_tokens=100,
        )
        response = llm.invoke("What is 2+2? Reply in one word.")
        return {
            "status": "success",
            "model": model,
            "response": response.content,
        }
    except Exception as e:
        return {
            "status": "error",
            "model": model,
            "error": str(e),
        }
