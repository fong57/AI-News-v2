"""User-configurable settings for LiteNews AI.

This module provides a centralized configuration system using Pydantic Settings.
All settings can be overridden via environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main application settings.
    
    All settings can be configured via environment variables.
    Environment variables take precedence over defaults.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys
    pplx_api_key: str = Field(default="", description="Perplexity API key")
    dashscope_api_key: str = Field(default="", description="DashScope/Qwen API key")
    tavily_api_key: str = Field(default="", description="Tavily API key")
    
    # LangSmith Tracing
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key",
    )
    langchain_project: str = Field(
        default="litenews-ai",
        description="LangSmith project name",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )
    
    # LLM Configuration
    primary_llm: Literal["perplexity", "qwen"] = Field(
        default="perplexity",
        description="Primary reasoning LLM to use",
    )
    perplexity_model: str = Field(
        default="sonar-pro",
        description="Perplexity model name",
    )
    qwen_model: str = Field(
        default="qwen-plus",
        description="Qwen model name",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=32768,
        description="Maximum tokens for LLM responses",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )
    
    # Search Configuration
    tavily_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="advanced",
        description="Tavily search depth",
    )
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results",
    )
    tavily_topic: Literal["general", "news", "finance"] = Field(
        default="news",
        description="Search topic for Tavily",
    )
    
    def has_perplexity_key(self) -> bool:
        """Check if Perplexity API key is configured."""
        return bool(self.pplx_api_key and self.pplx_api_key != "your_perplexity_api_key_here")
    
    def has_qwen_key(self) -> bool:
        """Check if DashScope/Qwen API key is configured."""
        return bool(self.dashscope_api_key and self.dashscope_api_key != "your_dashscope_api_key_here")
    
    def has_tavily_key(self) -> bool:
        """Check if Tavily API key is configured."""
        return bool(self.tavily_api_key and self.tavily_api_key != "your_tavily_api_key_here")
    
    def has_langsmith_key(self) -> bool:
        """Check if LangSmith API key is configured."""
        return bool(self.langchain_api_key and self.langchain_api_key != "your_langsmith_api_key_here")
    
    def is_tracing_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled and configured."""
        return self.langchain_tracing_v2 and self.has_langsmith_key()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Settings: Application settings instance.
    """
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache).
    
    Use this when environment variables have changed.
    
    Returns:
        Settings: Fresh settings instance.
    """
    get_settings.cache_clear()
    return get_settings()
