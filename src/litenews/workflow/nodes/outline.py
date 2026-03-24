"""Outline node for the news writing workflow.

This node creates an article outline based on research notes.
"""

from litenews.config.settings import get_settings
from litenews.state.news_state import NewsState, validate_article_type
from litenews.workflow.prompts import outline_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    create_llm_messages,
    invoke_llm_with_messages,
    workflow_llm_options,
)


async def outline_node(state: NewsState) -> dict:
    """Outline node: Create article outline based on research.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with article outline.
    """
    settings = get_settings()
    research_notes = state.get("research_notes", "")
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

    if not research_notes:
        return create_error_response("No research notes for outline")
    
    user_content = (
        f"Topic: {topic}\n"
        f"Article type: {article_type}\n\n"
        f"Research Notes:\n{research_notes}"
    )
    messages = create_llm_messages(outline_system_prompt(article_type), user_content)
    
    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        return {
            "outline": response.content,
            "status": "outlined",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Outline creation failed: {str(e)}")
