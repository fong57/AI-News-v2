"""Tests for Tavily search integration."""

import os
import pytest

from litenews.config.settings import Settings
from litenews.tools.search import (
    get_tavily_search_tool,
    search_news_sync,
    test_tavily_connection,
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
        result = test_tavily_connection(api_key)
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
            tavily_max_results=3,
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
            tavily_max_results=2,
            tavily_search_depth="fast",
        )
        results = await search_news("AI developments", settings=settings)
        assert results is not None


class TestSearchConfiguration:
    """Tests for search configuration."""
    
    def test_default_settings(self):
        """Test default search settings."""
        settings = Settings()
        assert settings.tavily_search_depth == "advanced"
        assert settings.tavily_max_results == 5
        assert settings.tavily_topic == "news"
    
    def test_custom_settings(self):
        """Test custom search settings."""
        settings = Settings(
            tavily_search_depth="basic",
            tavily_max_results=10,
            tavily_topic="finance",
        )
        assert settings.tavily_search_depth == "basic"
        assert settings.tavily_max_results == 10
        assert settings.tavily_topic == "finance"
