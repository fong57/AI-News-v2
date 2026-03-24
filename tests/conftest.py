"""Shared pytest fixtures for workflow tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from litenews.config.settings import Settings
from litenews.state.news_state import NewsSource


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        pplx_api_key="test_pplx_key",
        dashscope_api_key="test_dashscope_key",
        tavily_api_key="test_tavily_key",
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return AIMessage(content="This is a mock LLM response.")


@pytest.fixture
def mock_search_results():
    """Create mock search results from Tavily."""
    return [
        {
            "title": "Test Article 1",
            "url": "https://example.com/article1",
            "content": "This is the content of article 1.",
            "published_date": "2024-01-15",
        },
        {
            "title": "Test Article 2",
            "url": "https://example.com/article2",
            "content": "This is the content of article 2.",
            "published_date": "2024-01-16",
        },
    ]


@pytest.fixture
def mock_news_sources():
    """Create mock NewsSource objects."""
    return [
        NewsSource(
            title="Test Article 1",
            url="https://example.com/article1",
            snippet="This is the content of article 1.",
            published_date="2024-01-15",
        ),
        NewsSource(
            title="Test Article 2",
            url="https://example.com/article2",
            snippet="This is the content of article 2.",
            published_date="2024-01-16",
        ),
    ]


@pytest.fixture
def base_state():
    """Create a base state dict for testing."""
    return {
        "topic": "artificial intelligence",
        "article_type": "其他",
        "query": "",
        "messages": [],
        "search_results": [],
        "tavily_evidence_pool": [],
        "sources": [],
        "research_notes": "",
        "outline": "",
        "draft": "",
        "final_article": None,
        "feedback": "",
        "status": "initialized",
        "error": "",
    }


@pytest.fixture
def mock_tavily_tool():
    """Create a mock Tavily search tool."""
    mock_tool = AsyncMock()
    mock_tool.ainvoke = AsyncMock(return_value=[
        {
            "title": "Test Article",
            "url": "https://example.com/article",
            "content": "Test content",
            "published_date": "2024-01-15",
        }
    ])
    return mock_tool


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Mock LLM response"))
    return mock
