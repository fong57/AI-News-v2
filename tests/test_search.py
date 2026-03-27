"""Tests for Tavily search integration."""

import os
from unittest.mock import patch

import pytest

from litenews.config.settings import Settings
from litenews.tools.search import (
    get_tavily_search_tool,
    search_news_sync,
    verify_tavily_connection,
)


class TestTavilyIntegration:
    """Tests for Tavily search integration."""
    
    @pytest.fixture
    def api_key(self):
        """Get Tavily API key from environment."""
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key or key == "your_tavily_api_key_here":
            pytest.skip("TAVILY_API_KEY not set")
        return key
    
    def test_connection(self, api_key):
        """Test basic Tavily API connection."""
        result = verify_tavily_connection(api_key)
        assert result["status"] == "success", f"Connection failed: {result.get('error')}"
    
    def test_search_tool_creation(self, api_key):
        """Test search tool can be created with settings."""
        settings = Settings(tavily_api_key=api_key)
        tool = get_tavily_search_tool(settings)
        assert tool is not None
    
    def test_news_search(self, api_key):
        """Test searching for news."""
        settings = Settings(
            tavily_api_key=api_key,
            tavily_research_max_results=3,
            tavily_search_depth="fast",
        )
        results = search_news_sync("latest technology news", settings=settings)
        assert results is not None
    
    @pytest.mark.asyncio
    async def test_async_search(self, api_key):
        """Test async search."""
        from litenews.tools.search import search_news
        
        settings = Settings(
            tavily_api_key=api_key,
            tavily_research_max_results=2,
            tavily_search_depth="fast",
        )
        results = await search_news("AI developments", settings=settings)
        assert results is not None


class TestSearchConfiguration:
    """Tests for search configuration."""
    
    def test_default_settings(self):
        """Test default search settings."""
        settings = Settings(_env_file=None)
        assert settings.tavily_search_depth == "advanced"
        assert settings.tavily_topic == "news"
        assert settings.tavily_research_max_results == 20
        assert settings.tavily_fact_check_max_results == 10
        assert settings.tavily_write_max_results == 5
        assert settings.tavily_exclude_domains == []

    def test_custom_settings(self):
        """Test custom per-node max results and shared depth/topic."""
        settings = Settings(
            _env_file=None,
            tavily_search_depth="basic",
            tavily_topic="finance",
            tavily_research_max_results=15,
            tavily_fact_check_max_results=8,
            tavily_write_max_results=4,
        )
        assert settings.tavily_search_depth == "basic"
        assert settings.tavily_topic == "finance"
        assert settings.tavily_research_max_results == 15
        assert settings.tavily_fact_check_max_results == 8
        assert settings.tavily_write_max_results == 4

    def test_tavily_exclude_domains_parsed_from_string(self):
        settings = Settings(
            _env_file=None,
            tavily_exclude_domains="Foo.COM , .bar.org ",
        )
        assert settings.tavily_exclude_domains == ["foo.com", "bar.org"]


def test_get_tavily_search_tool_passes_exclude_domains():
    settings = Settings(
        _env_file=None,
        tavily_api_key="k",
        tavily_exclude_domains=["spam.com", "evil.net"],
    )
    with patch("litenews.tools.search.TavilySearch") as mock_ts:
        get_tavily_search_tool(settings)
    kwargs = mock_ts.call_args.kwargs
    assert kwargs["exclude_domains"] == ["spam.com", "evil.net"]


def test_get_tavily_search_tool_omits_exclude_when_empty():
    settings = Settings(_env_file=None, tavily_api_key="k")
    with patch("litenews.tools.search.TavilySearch") as mock_ts:
        get_tavily_search_tool(settings)
    assert "exclude_domains" not in mock_ts.call_args.kwargs
