"""User-configurable settings for LiteNews AI.

This module provides a centralized configuration system using Pydantic Settings.
All settings can be overridden via environment variables or .env file.
"""

import json
from functools import lru_cache
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_tavily_exclude_domains_csv(s: str) -> list[str]:
    """Parse TAVILY_EXCLUDE_DOMAINS from env or kwargs (comma-separated or JSON array string)."""
    s = (s or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [
                str(x).strip().lower().lstrip(".")
                for x in parsed
                if str(x).strip()
            ]
    return [p.strip().lower().lstrip(".") for p in s.split(",") if p.strip()]


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
    bailian_api_key: str = Field(
        default="",
        description="Alibaba Bailian (Model Studio) API key (OpenAI-compatible)",
    )
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
    primary_llm: Literal["perplexity", "qwen", "bailian"] = Field(
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
    bailian_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Bailian OpenAI-compatible API base URL",
    )
    bailian_model: str = Field(
        default="qwen-plus",
        description="Model id for Bailian compatible-mode API",
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

    # Target word count (configure node defaults and validation bounds)
    default_target_word_count: int = Field(
        default=800,
        ge=1,
        description="Default article body word-count target when omitted from workflow state",
    )
    min_target_word_count: int = Field(
        default=200,
        ge=1,
        description="Minimum allowed target_word_count from workflow state",
    )
    max_target_word_count: int = Field(
        default=20000,
        ge=1,
        description="Maximum allowed target_word_count from workflow state",
    )
    
    # Search Configuration (shared by all Tavily-using nodes unless overridden per node)
    tavily_search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="advanced",
        description="Tavily search depth (research, fact-check, write)",
    )
    tavily_topic: Literal["general", "news", "finance"] = Field(
        default="news",
        description="Tavily topic (research, fact-check, write)",
    )
    tavily_research_max_results: int = Field(
        default=20,
        ge=1,
        le=20,
        description="Max Tavily results per query in the research node",
    )
    tavily_fact_check_max_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Max Tavily results per claim search in the fact-check node",
    )
    tavily_write_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max Tavily results per outline query in the write node",
    )
    # Stored as str so pydantic-settings does not JSON-decode the env var (empty or CSV breaks json.loads).
    tavily_exclude_domains_env: str = Field(
        default="",
        exclude=True,
        description=(
            "Comma-separated or JSON-array string of hostnames excluded from Tavily "
            "(API exclude_domains + client post-filter). Env: TAVILY_EXCLUDE_DOMAINS."
        ),
        validation_alias=AliasChoices("TAVILY_EXCLUDE_DOMAINS", "tavily_exclude_domains_env"),
    )

    @model_validator(mode="before")
    @classmethod
    def _tavily_exclude_domains_kwarg(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "tavily_exclude_domains" not in data:
            return data
        raw = data.pop("tavily_exclude_domains")
        if isinstance(raw, list):
            data["tavily_exclude_domains_env"] = ",".join(
                str(x).strip() for x in raw if str(x).strip()
            )
        elif raw is None:
            data.setdefault("tavily_exclude_domains_env", "")
        else:
            data["tavily_exclude_domains_env"] = str(raw).strip()
        return data

    @computed_field
    @property
    def tavily_exclude_domains(self) -> list[str]:
        return _parse_tavily_exclude_domains_csv(self.tavily_exclude_domains_env)

    def model_copy(
        self,
        *,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Support model_copy(update={\"tavily_exclude_domains\": ...}) for tests and overrides."""
        if update and "tavily_exclude_domains" in update:
            u = dict(update)
            raw = u.pop("tavily_exclude_domains")
            if isinstance(raw, list):
                u["tavily_exclude_domains_env"] = ",".join(
                    str(x).strip() for x in raw if str(x).strip()
                )
            elif raw is None:
                u.setdefault("tavily_exclude_domains_env", "")
            else:
                u["tavily_exclude_domains_env"] = str(raw).strip()
            update = u
        return super().model_copy(update=update, deep=deep)

    def has_perplexity_key(self) -> bool:
        """Check if Perplexity API key is configured."""
        return bool(self.pplx_api_key and self.pplx_api_key != "your_perplexity_api_key_here")
    
    def has_qwen_key(self) -> bool:
        """Check if DashScope/Qwen API key is configured."""
        return bool(self.dashscope_api_key and self.dashscope_api_key != "your_dashscope_api_key_here")

    def has_bailian_key(self) -> bool:
        """Check if Alibaba Bailian API key is configured."""
        return bool(
            self.bailian_api_key and self.bailian_api_key != "your_bailian_api_key_here"
        )
    
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
