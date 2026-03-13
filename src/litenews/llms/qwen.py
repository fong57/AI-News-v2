"""Qwen (DashScope) LLM integration for LiteNews AI.

This module provides integration with Alibaba's Qwen models via DashScope API
using langchain-qwq package.
"""

from langchain_core.language_models import BaseChatModel
from langchain_qwq import ChatQwen

from litenews.config.llm_config import LLMConfig
from litenews.llms.base import BaseLLM


class QwenLLM(BaseLLM):
    """Qwen LLM wrapper for DashScope API.
    
    Qwen models are powerful multilingual models with strong reasoning
    and code generation capabilities.
    
    Available models:
        - qwen-turbo: Fast, cost-effective model
        - qwen-plus: Balanced performance and cost
        - qwen-max: Most capable model
        - qwen-flash: Ultra-fast for simple tasks
    """
    
    def _create_model(self) -> BaseChatModel:
        """Create the ChatQwen model instance.
        
        Returns:
            BaseChatModel: The ChatQwen model.
        """
        return ChatQwen(
            model=self.config.model,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )


def test_qwen_connection(api_key: str, model: str = "qwen-turbo") -> dict:
    """Test Qwen/DashScope API connection.
    
    Args:
        api_key: DashScope API key.
        model: Model name to test.
        
    Returns:
        dict: Test result with status and response.
    """
    try:
        llm = ChatQwen(
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
