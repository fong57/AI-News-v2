"""Review node for the news writing workflow.

This node polishes and finalizes the article draft.
"""

from litenews.config.settings import get_settings
from litenews.state.news_state import (
    DEFAULT_TARGET_WORD_COUNT,
    NewsArticle,
    NewsState,
    validate_article_type,
)
from litenews.workflow.prompts import review_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    create_llm_messages,
    invoke_llm_with_messages,
    workflow_llm_options,
)


def parse_article_response(content: str, topic: str, sources: list) -> NewsArticle:
    """Parse the LLM response into a NewsArticle object.
    
    Args:
        content: The raw LLM response content.
        topic: The article topic (used as fallback headline).
        sources: List of sources used in the article.
        
    Returns:
        NewsArticle: The parsed article object.
    """
    stripped = content.strip()
    lines = stripped.split("\n") if stripped else []
    headline_raw = lines[0].replace("#", "").strip() if lines else ""
    headline = headline_raw or topic
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else content
    
    summary_end = body.find("\n\n")
    summary = body[:summary_end] if summary_end > 0 else body[:200]
    
    return NewsArticle(
        headline=headline,
        summary=summary,
        body=body,
        sources=sources,
        keywords=[topic],
    )


async def review_node(state: NewsState) -> dict:
    """Review node: Polish and finalize the article.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with final article.
    """
    settings = get_settings()
    draft = state.get("draft", "")
    sources = state.get("sources", [])
    topic = state.get("topic", "")
    raw_at = state.get("article_type")
    try:
        target_word_count = int(
            state.get("target_word_count", DEFAULT_TARGET_WORD_COUNT) or DEFAULT_TARGET_WORD_COUNT
        )
    except (TypeError, ValueError):
        return create_error_response("target_word_count must be an integer")

    if raw_at is None or (isinstance(raw_at, str) and not raw_at.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        article_type = validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    if not draft:
        return create_error_response("No draft to review")
    
    user_content = f"""Topic: {topic}
Article type: {article_type}

Draft Article:
{draft}

Please review and provide the final polished version."""
    
    messages = create_llm_messages(
        review_system_prompt(article_type, target_word_count),
        user_content,
    )

    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        final_article = parse_article_response(response.content, topic, sources)
        
        return {
            "final_article": final_article,
            "status": "completed",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Review failed: {str(e)}")
