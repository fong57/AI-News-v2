"""Publish node: produce a clean public-facing copy of the final article."""

from litenews.config.settings import get_settings
from litenews.state.news_state import NewsState, validate_article_type
from litenews.workflow.nodes.review import parse_article_response
from litenews.workflow.prompts import publish_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    create_llm_messages,
    invoke_llm_with_messages,
    workflow_llm_options,
)


async def publish_node(state: NewsState) -> dict:
    """Strip internal fact-check notes and source attributions for platform paste."""
    settings = get_settings()
    final = state.get("final_article")
    topic = state.get("topic", "")
    raw_at = state.get("article_type")

    if raw_at is None or (isinstance(raw_at, str) and not raw_at.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    if final is None:
        return create_error_response("No final article to publish")

    if hasattr(final, "headline"):
        headline = final.headline or topic
        body = final.body or ""
        summary = final.summary or ""
    elif isinstance(final, dict):
        headline = (final.get("headline") or "").strip() or topic
        body = (final.get("body") or "").strip()
        summary = (final.get("summary") or "").strip()
    else:
        return create_error_response("Invalid final_article")

    internal_full = (
        f"# {headline}\n\n{summary}\n\n{body}".strip()
        if summary
        else f"# {headline}\n\n{body}".strip()
    )

    user_content = f"""Topic: {topic}

Internal final article (produce the public version per system instructions):
{internal_full}"""

    messages = create_llm_messages(publish_system_prompt(), user_content)

    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        published = parse_article_response(response.content, topic, sources=[])

        return {
            "published_article": published,
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Publish failed: {str(e)}")
