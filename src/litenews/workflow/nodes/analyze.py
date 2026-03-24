"""Analyze node for the news writing workflow.

This node processes search results and extracts key information using an LLM.
"""

from litenews.config.settings import get_settings
from litenews.state.news_state import NewsState, NewsSource, validate_article_type
from litenews.workflow.prompts import research_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    create_llm_messages,
    invoke_llm_with_messages,
    workflow_llm_options,
)


async def analyze_node(state: NewsState) -> dict:
    """Analyze node: Process search results and extract key information.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with sources and research notes.
    """
    settings = get_settings()
    search_results = state.get("search_results", [])
    topic = state.get("topic", "")
    raw_at = state.get("article_type")

    if raw_at is None or (isinstance(raw_at, str) and not raw_at.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        article_type = validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    if not search_results:
        return create_error_response("No search results to analyze")
    
    sources = []
    for result in search_results:
        if isinstance(result, dict):
            sources.append(NewsSource(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", result.get("snippet", "")),
                published_date=result.get("published_date"),
            ))
    
    sources_text = "\n\n".join([
        f"Source: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
        for s in sources
    ])
    
    user_content = (
        f"Topic: {topic}\n"
        f"Article type: {article_type}\n\n"
        f"Search Results:\n{sources_text}"
    )
    messages = create_llm_messages(research_system_prompt(article_type), user_content)
    
    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        return {
            "sources": sources,
            "research_notes": response.content,
            "status": "analyzed",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Analysis failed: {str(e)}")
