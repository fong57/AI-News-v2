"""Write node for the news writing workflow.

This node runs fresh searches aligned to the confirmed outline, then generates
the article draft from the outline and those results (not from prior research notes).
"""

import asyncio
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from litenews.config.settings import get_settings
from litenews.state.news_state import (
    DEFAULT_TARGET_WORD_COUNT,
    NewsState,
    validate_article_type,
)
from litenews.tools.search import get_tavily_search_tool
from litenews.workflow.prompts import write_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    invoke_llm_with_messages,
    workflow_llm_options,
)
from litenews.workflow.write_few_shots import build_write_few_shot_messages
from litenews.workflow.nodes.research import _parse_search_response
from litenews.workflow.tavily_pool import filter_blocked_tavily_rows, merge_into_pool

_MAX_OUTLINE_QUERIES = 6
_MAX_QUERY_LEN = 400
_MIN_LINE_LEN = 6


def _strip_outline_line_prefix(line: str) -> str:
    s = line.strip()
    if not s:
        return ""
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = re.sub(r"^[-*•]\s+", "", s)
    s = re.sub(r"^\d+[.、)\]]\s*", "", s)
    s = re.sub(r"^[一二三四五六七八九十百千]+[、.．]\s*", "", s)
    return s.strip()


def _outline_to_search_queries(outline: str, topic: str) -> list[str]:
    """Build Tavily queries from outline lines (headings / bullet points)."""
    topic_t = topic.strip()
    seen: set[str] = set()
    queries: list[str] = []
    for raw in outline.splitlines():
        line = _strip_outline_line_prefix(raw)
        if len(line) < _MIN_LINE_LEN:
            continue
        if line in seen:
            continue
        seen.add(line)
        q = f"{topic_t} {line}" if topic_t else line
        queries.append(q[:_MAX_QUERY_LEN])
        if len(queries) >= _MAX_OUTLINE_QUERIES:
            break
    if not queries and topic_t:
        queries.append(f"Latest news about {topic_t}")
    return queries


def _dedupe_results_by_url(results: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or "").strip()
        if url:
            if url in seen:
                continue
            seen.add(url)
        out.append(r)
    return out


def _format_write_sources(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))
        blocks.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
    return "\n\n".join(blocks) if blocks else ""


async def write_node(state: NewsState) -> dict:
    """Write node: Search by outline, then generate draft from outline + results.

    Args:
        state: Current workflow state.

    Returns:
        dict: Updated state with article draft.
    """
    settings = get_settings()
    outline = state.get("outline", "")
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

    if not outline:
        return create_error_response("No outline for writing")

    if not settings.has_tavily_key():
        return create_error_response(
            "Tavily API key is missing or still set to the placeholder. "
            "Set TAVILY_API_KEY in your environment."
        )

    queries = _outline_to_search_queries(outline, topic)
    if not queries:
        return create_error_response(
            "Could not derive search queries from the outline; add clearer section lines."
        )

    try:
        search_tool = get_tavily_search_tool(settings, node="write")

        async def _invoke_one(q: str) -> tuple[list[Any], str | None]:
            raw = await search_tool.ainvoke(q)
            return _parse_search_response(raw)

        parsed_batches = await asyncio.gather(*[_invoke_one(q) for q in queries])

        merged_raw: list[Any] = []
        failed_queries: list[str] = []
        for batch, api_error in parsed_batches:
            if api_error:
                failed_queries.append(api_error)
                continue
            merged_raw.extend(batch)

        merged = _dedupe_results_by_url(merged_raw)
        merged = filter_blocked_tavily_rows(merged, settings.tavily_exclude_domains)
        if not merged:
            suffix = (
                f" Last Tavily errors: {' | '.join(failed_queries[:3])}"
                if failed_queries
                else ""
            )
            return create_error_response(
                "Outline search returned no results. Check TAVILY_API_KEY, quota, and network."
                + suffix
            )

        prior_pool = state.get("tavily_evidence_pool") or []
        tavily_evidence_pool = merge_into_pool(prior_pool, merged)

        sources_text = _format_write_sources(merged)
    except Exception as e:
        return create_error_response(f"Outline search failed: {str(e)}")

    user_content = f"""Topic: {topic}
Article type: {article_type}

Outline:
{outline}

Sources (outline-guided search; use for facts and attribution):
{sources_text}

Please write the full article following the outline."""

    messages = [
        SystemMessage(content=write_system_prompt(article_type, target_word_count)),
        *build_write_few_shot_messages(article_type),
        HumanMessage(content=user_content),
    ]

    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        return {
            "draft": response.content,
            "fact_check_revision_round": 0,
            "fact_check_evidence_cache": {},
            "last_fact_checked_draft": "",
            "tavily_evidence_pool": tavily_evidence_pool,
            "status": "drafted",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Writing failed: {str(e)}")
