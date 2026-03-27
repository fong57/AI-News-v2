"""Base LLM interface for LiteNews AI.

This module provides a unified interface for different LLM providers,
allowing easy swapping of LLMs without changing application code.
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from litenews.config.llm_config import LLMConfig, get_llm_config
from litenews.config.settings import Settings, get_settings


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM with configuration.
        
        Args:
            config: LLM configuration.
        """
        self.config = config
        self._model: BaseChatModel | None = None
    
    @property
    def model(self) -> BaseChatModel:
        """Get the underlying LangChain chat model.
        
        Returns:
            BaseChatModel: The LangChain chat model instance.
        """
        if self._model is None:
            self._model = self._create_model()
        return self._model
    
    @abstractmethod
    def _create_model(self) -> BaseChatModel:
        """Create the underlying LangChain chat model.
        
        Returns:
            BaseChatModel: The LangChain chat model instance.
        """
        pass
    
    def invoke(self, messages: list[BaseMessage] | str, **kwargs: Any) -> BaseMessage:
        """Invoke the LLM with messages.
        
        Args:
            messages: List of messages or a single string prompt.
            **kwargs: Additional arguments passed to the model.
            
        Returns:
            BaseMessage: The model's response.
        """
        return self.model.invoke(messages, **kwargs)
    
    async def ainvoke(self, messages: list[BaseMessage] | str, **kwargs: Any) -> BaseMessage:
        """Async invoke the LLM with messages.
        
        Args:
            messages: List of messages or a single string prompt.
            **kwargs: Additional arguments passed to the model.
            
        Returns:
            BaseMessage: The model's response.
        """
        return await self.model.ainvoke(messages, **kwargs)
    
    def bind_tools(self, tools: list[Any]) -> BaseChatModel:
        """Bind tools to the model for function calling.
        
        Args:
            tools: List of tools to bind.
            
        Returns:
            BaseChatModel: Model with tools bound.
        """
        return self.model.bind_tools(tools)
    
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "BaseLLM":
        """Create LLM instance from settings.
        
        Args:
            settings: Optional settings instance.
            
        Returns:
            BaseLLM: The LLM instance.
        """
        settings = settings or get_settings()
        config = get_llm_config(settings=settings)
        return cls(config)


def get_llm(
    provider: str | None = None,
    settings: Settings | None = None,
    *,
    model_override: str | None = None,
) -> BaseLLM:
    """Factory function to get an LLM instance.
    
    Args:
        provider: LLM provider name ('perplexity', 'qwen', or 'bailian').
                  If None, uses primary_llm from settings.
        settings: Optional settings instance.
        model_override: If set, use this model id instead of the env default for the provider.
        
    Returns:
        BaseLLM: The LLM instance.
        
    Raises:
        ValueError: If provider is not supported.
    """
    from litenews.llms.bailian import BailianLLM
    from litenews.llms.perplexity import PerplexityLLM
    from litenews.llms.qwen import QwenLLM
    
    settings = settings or get_settings()
    provider = provider or settings.primary_llm
    config = get_llm_config(provider, settings)
    mo = (model_override or "").strip()
    if mo:
        config = replace(config, model=mo)
    
    if provider == "perplexity":
        return PerplexityLLM(config)
    elif provider == "qwen":
        return QwenLLM(config)
    elif provider == "bailian":
        return BailianLLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
