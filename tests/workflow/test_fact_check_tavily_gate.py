"""Fact-check should not call Tavily when fact_check_revision_round > 0."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from litenews.workflow.nodes.fact_check import fact_check_node


@pytest.mark.asyncio
async def test_fact_check_second_pass_does_not_call_tavily(mock_settings):
    extract_json = (
        '{"claims": [{"id": "c1", "text": "台灣經濟成長", "importance": 4}]}'
    )
    check_json = '{"status": "uncertain", "reason": "from pool"}'

    with patch(
        "litenews.workflow.nodes.fact_check.get_settings",
        return_value=mock_settings,
    ), patch(
        "litenews.workflow.nodes.fact_check.get_tavily_search_tool",
    ) as mock_get_tool, patch(
        "litenews.workflow.nodes.fact_check.invoke_llm_with_messages",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.side_effect = [
            AIMessage(content=extract_json),
            AIMessage(content=check_json),
        ]
        state = {
            "topic": "t",
            "article_type": "其他",
            "draft": "本文提及台灣經濟成長。",
            "fact_check_revision_round": 1,
            "llm_provider": "perplexity",
            "tavily_evidence_pool": [
                {
                    "title": "GDP",
                    "url": "https://example.com/gdp",
                    "snippet": "台灣經濟成長數據與預測",
                },
            ],
        }
        out = await fact_check_node(state)

    mock_get_tool.assert_not_called()
    assert "error" not in out
    fr = out.get("fact_check_results") or {}
    claims = fr.get("claims") or []
    assert len(claims) == 1
    assert claims[0].get("status") == "uncertain"


@pytest.mark.asyncio
async def test_fact_check_first_pass_still_uses_tavily(mock_settings):
    extract_json = (
        '{"claims": [{"id": "c1", "text": "one claim", "importance": 4}]}'
    )
    check_json = '{"status": "supported", "reason": "ok"}'

    mock_tool = AsyncMock()
    mock_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Src",
                    "url": "https://src.example",
                    "content": "evidence text",
                }
            ]
        }
    )

    with patch(
        "litenews.workflow.nodes.fact_check.get_settings",
        return_value=mock_settings,
    ), patch(
        "litenews.workflow.nodes.fact_check.get_tavily_search_tool",
        return_value=mock_tool,
    ) as mock_get_tool, patch(
        "litenews.workflow.nodes.fact_check.invoke_llm_with_messages",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.side_effect = [
            AIMessage(content=extract_json),
            AIMessage(content=check_json),
        ]
        state = {
            "topic": "t",
            "article_type": "其他",
            "draft": "Something about one claim here.",
            "fact_check_revision_round": 0,
            "llm_provider": "perplexity",
            "tavily_evidence_pool": [],
        }
        out = await fact_check_node(state)

    mock_get_tool.assert_called_once()
    mock_tool.ainvoke.assert_awaited()
    assert "error" not in out
    pool = out.get("tavily_evidence_pool") or []
    assert any(r.get("url") == "https://src.example" for r in pool)


@pytest.mark.asyncio
async def test_fact_check_first_pass_drops_blocked_domains_from_pool(mock_settings):
    extract_json = (
        '{"claims": [{"id": "c1", "text": "one claim", "importance": 4}]}'
    )
    check_json = '{"status": "supported", "reason": "ok"}'

    mock_tool = AsyncMock()
    mock_tool.ainvoke = AsyncMock(
        return_value={
            "results": [
                {
                    "title": "Bad",
                    "url": "https://blocked.fc.test/x",
                    "content": "spam",
                },
                {
                    "title": "Good",
                    "url": "https://good.fc.test/y",
                    "content": "evidence text",
                },
            ]
        }
    )

    settings = mock_settings.model_copy(
        update={"tavily_exclude_domains": ["blocked.fc.test"]}
    )

    with patch(
        "litenews.workflow.nodes.fact_check.get_settings",
        return_value=settings,
    ), patch(
        "litenews.workflow.nodes.fact_check.get_tavily_search_tool",
        return_value=mock_tool,
    ), patch(
        "litenews.workflow.nodes.fact_check.invoke_llm_with_messages",
        new_callable=AsyncMock,
    ) as mock_llm:
        mock_llm.side_effect = [
            AIMessage(content=extract_json),
            AIMessage(content=check_json),
        ]
        state = {
            "topic": "t",
            "article_type": "其他",
            "draft": "Something about one claim here.",
            "fact_check_revision_round": 0,
            "llm_provider": "perplexity",
            "tavily_evidence_pool": [],
        }
        out = await fact_check_node(state)

    assert "error" not in out
    pool = out.get("tavily_evidence_pool") or []
    assert not any("blocked.fc.test" in (r.get("url") or "") for r in pool)
    assert any(r.get("url") == "https://good.fc.test/y" for r in pool)
    fr = out.get("fact_check_results") or {}
    claims = fr.get("claims") or []
    assert len(claims) == 1
    ev = claims[0].get("evidence_snippets") or []
    assert all("blocked.fc.test" not in (e.get("url") or "") for e in ev)
