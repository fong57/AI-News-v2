"""LLM-specific configuration for LiteNews AI.

This module provides configuration classes for different LLM providers,
making it easy to customize LLM behavior without modifying code.
"""

from dataclasses import dataclass, field
from typing import Any

from litenews.config.settings import Settings, get_settings


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    
    provider: str
    model: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for LLM initialization."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_params,
        }


def get_perplexity_config(settings: Settings | None = None) -> LLMConfig:
    """Get Perplexity LLM configuration.
    
    Args:
        settings: Optional settings instance. Uses default if not provided.
        
    Returns:
        LLMConfig: Perplexity configuration.
    """
    settings = settings or get_settings()
    return LLMConfig(
        provider="perplexity",
        model=settings.perplexity_model,
        api_key=settings.pplx_api_key,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


def get_qwen_config(settings: Settings | None = None) -> LLMConfig:
    """Get Qwen/DashScope LLM configuration.
    
    Args:
        settings: Optional settings instance. Uses default if not provided.
        
    Returns:
        LLMConfig: Qwen configuration.
    """
    settings = settings or get_settings()
    return LLMConfig(
        provider="qwen",
        model=settings.qwen_model,
        api_key=settings.dashscope_api_key,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )


def get_llm_config(provider: str | None = None, settings: Settings | None = None) -> LLMConfig:
    """Get LLM configuration for the specified provider.
    
    Args:
        provider: LLM provider name. If None, uses primary_llm from settings.
        settings: Optional settings instance. Uses default if not provided.
        
    Returns:
        LLMConfig: Configuration for the specified LLM provider.
        
    Raises:
        ValueError: If provider is not supported.
    """
    settings = settings or get_settings()
    provider = provider or settings.primary_llm
    
    if provider == "perplexity":
        return get_perplexity_config(settings)
    elif provider == "qwen":
        return get_qwen_config(settings)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Use 'perplexity' or 'qwen'.")
