"""Research node for the news writing workflow.

This node searches for news sources on the given topic using Tavily search.
"""

from typing import Any

from litenews.config.settings import get_settings
from litenews.state.news_state import NewsState, validate_article_type
from litenews.tools.search import get_tavily_search_tool
from litenews.workflow.tavily_pool import merge_into_pool
from litenews.workflow.utils import create_error_response


def _normalize_tavily_error(err: Any) -> str:
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err)
    return str(err)


def _parse_search_response(raw: Any) -> tuple[list[Any], str | None]:
    """Parse Tavily tool output. Returns (results, api_error_message_or_none)."""
    if isinstance(raw, dict):
        if raw.get("error"):
            return [], _normalize_tavily_error(raw["error"])
        if raw.get("detail") is not None and "results" not in raw:
            return [], _normalize_tavily_error(raw["detail"])
        return list(raw.get("results") or []), None
    if isinstance(raw, list):
        return raw, None
    return [], f"unexpected response type: {type(raw).__name__}"


async def research_node(state: NewsState) -> dict:
    """Research node: Search for news sources on the topic.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with search results.
    """
    settings = get_settings()
    topic = state.get("topic", "")
    raw_article_type = state.get("article_type")

    if raw_article_type is None or (isinstance(raw_article_type, str) and not raw_article_type.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        validate_article_type(str(raw_article_type))
    except ValueError as e:
        return create_error_response(str(e))

    if not topic:
        return create_error_response("No topic provided")

    if not settings.has_tavily_key():
        return create_error_response(
            "Tavily API key is missing or still set to the placeholder. "
            "Set TAVILY_API_KEY in your environment."
        )

    # Use custom query if exists, else auto-generate Chinese-focused query
    base_query = state.get("query") or f"Latest news about {topic}"
    # Add Chinese source/language priority for auto-generated queries only
    if state.get("query"):
        search_query = base_query
    else:
        search_query = (
            f"{base_query}" # site:.cn OR site:.com.cn OR 中文 新闻"
        )

    try:
        search_tool = get_tavily_search_tool(settings, node="research")
        raw = await search_tool.ainvoke(search_query)
        search_results, api_error = _parse_search_response(raw)

        if api_error:
            return create_error_response(f"Research failed (Tavily): {api_error}")

        if not search_results:
            return create_error_response(
                "Research failed: Tavily returned no search results. "
                "Check TAVILY_API_KEY, quota, and network access."
            )

        prior_pool = state.get("tavily_evidence_pool") or []
        pool = merge_into_pool(prior_pool, search_results)
        return {
            "query": base_query,
            "search_results": search_results,
            "tavily_evidence_pool": pool,
            "status": "researched",
        }
    except Exception as e:
        return create_error_response(f"Research failed: {str(e)}")
