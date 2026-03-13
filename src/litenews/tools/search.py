"""Tavily search tool integration for LiteNews AI.

This module provides Tavily search functionality for news research,
including search tools for agents and retriever for RAG workflows.
"""

from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch

from litenews.config.settings import Settings, get_settings


def get_tavily_search_tool(
    settings: Settings | None = None,
    max_results: int | None = None,
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] | None = None,
    topic: Literal["general", "news", "finance"] | None = None,
    include_answer: bool = True,
    include_raw_content: bool = False,
) -> TavilySearch:
    """Get a configured Tavily search tool.
    
    Args:
        settings: Optional settings instance.
        max_results: Maximum search results (overrides settings).
        search_depth: Search depth (overrides settings).
        topic: Search topic (overrides settings).
        include_answer: Whether to include AI-generated answer.
        include_raw_content: Whether to include raw page content.
        
    Returns:
        TavilySearch: Configured Tavily search tool.
    """
    settings = settings or get_settings()
    
    return TavilySearch(
        api_key=settings.tavily_api_key,
        max_results=max_results or settings.tavily_max_results,
        search_depth=search_depth or settings.tavily_search_depth,
        topic=topic or settings.tavily_topic,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
    )


def get_tavily_retriever(
    settings: Settings | None = None,
    k: int = 5,
) -> Any:
    """Get a Tavily retriever for RAG workflows.
    
    Args:
        settings: Optional settings instance.
        k: Number of documents to retrieve.
        
    Returns:
        TavilySearchAPIRetriever: Configured retriever.
    """
    from langchain_community.retrievers import TavilySearchAPIRetriever
    
    settings = settings or get_settings()
    
    return TavilySearchAPIRetriever(
        api_key=settings.tavily_api_key,
        k=k,
    )


async def search_news(
    query: str,
    settings: Settings | None = None,
    max_results: int | None = None,
    time_range: str | None = None,
) -> dict[str, Any]:
    """Search for news using Tavily.
    
    This is a high-level function for news-specific searches.
    
    Args:
        query: Search query.
        settings: Optional settings instance.
        max_results: Maximum results to return.
        time_range: Time range filter (e.g., "day", "week", "month").
        
    Returns:
        dict: Search results with answer and sources.
    """
    settings = settings or get_settings()
    
    tool = TavilySearch(
        api_key=settings.tavily_api_key,
        max_results=max_results or settings.tavily_max_results,
        search_depth=settings.tavily_search_depth,
        topic="news",
        include_answer=True,
        time_range=time_range,
    )
    
    results = await tool.ainvoke(query)
    return results


def search_news_sync(
    query: str,
    settings: Settings | None = None,
    max_results: int | None = None,
    time_range: str | None = None,
) -> dict[str, Any]:
    """Synchronous version of search_news.
    
    Args:
        query: Search query.
        settings: Optional settings instance.
        max_results: Maximum results to return.
        time_range: Time range filter.
        
    Returns:
        dict: Search results with answer and sources.
    """
    settings = settings or get_settings()
    
    tool = TavilySearch(
        api_key=settings.tavily_api_key,
        max_results=max_results or settings.tavily_max_results,
        search_depth=settings.tavily_search_depth,
        topic="news",
        include_answer=True,
        time_range=time_range,
    )
    
    results = tool.invoke(query)
    return results


def test_tavily_connection(api_key: str) -> dict[str, Any]:
    """Test Tavily API connection.
    
    Args:
        api_key: Tavily API key.
        
    Returns:
        dict: Test result with status and sample response.
    """
    try:
        tool = TavilySearch(
            api_key=api_key,
            max_results=1,
            search_depth="fast",
            topic="general",
        )
        results = tool.invoke("Test query: What is the weather today?")
        return {
            "status": "success",
            "results_count": len(results.get("results", [])) if isinstance(results, dict) else 1,
            "sample": str(results)[:200] + "..." if len(str(results)) > 200 else str(results),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
