"""Tests for the write node."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from litenews.workflow.nodes.write import write_node


def _last_human_content(messages):
    """Final user turn (after optional few-shot pairs)."""
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    return humans[-1].content


def _stub_tavily_tool():
    """Minimal Tavily-shaped async tool for write_node tests."""
    tool = AsyncMock()
    tool.ainvoke = AsyncMock(
        return_value=[
            {
                "title": "Test Article",
                "url": "https://example.com/write-search",
                "content": "Outline-guided snippet for the draft.",
            }
        ]
    )
    return tool


def _stub_tavily_tool_mixed_domains():
    tool = AsyncMock()
    tool.ainvoke = AsyncMock(
        return_value=[
            {
                "title": "Good",
                "url": "https://good.write.test/a",
                "content": "keep",
            },
            {
                "title": "Bad",
                "url": "https://bad.write.test/b",
                "content": "drop",
            },
        ]
    )
    return tool


def _stub_tavily_tool_partial_error():
    tool = AsyncMock()
    responses = iter(
        [
            "Rate limit exceeded",
            [
                {
                    "title": "Recovered Result",
                    "url": "https://example.com/recovered",
                    "content": "Valid result despite one failed query.",
                }
            ],
        ]
    )
    tool.ainvoke = AsyncMock(side_effect=lambda _: next(responses))
    return tool


class TestWriteNode:
    """Tests for the write_node function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_outline(self, base_state):
        """Test that it returns an error when no outline exists."""
        base_state["outline"] = ""

        result = await write_node(base_state)

        assert result["status"] == "error"
        assert "No outline for writing" in result["error"]

    @pytest.mark.asyncio
    async def test_filters_blocked_domains_before_pool_and_prompt(
        self, base_state, mock_settings
    ):
        """Excluded domains must not appear in tavily_evidence_pool or writer prompt."""
        base_state["outline"] = "# Section\nSome heading here"
        base_state["topic"] = "news"
        settings = mock_settings.model_copy(
            update={"tavily_exclude_domains": ["bad.write.test"]}
        )
        mock_response = AIMessage(content="Draft body")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool_mixed_domains(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 mock_invoke,
             ):
            result = await write_node(base_state)

        assert result["status"] == "drafted"
        pool = result.get("tavily_evidence_pool") or []
        assert len(pool) == 1
        assert pool[0].get("url") == "https://good.write.test/a"
        messages = mock_invoke.call_args[0][0]
        content = _last_human_content(messages)
        assert "good.write.test" in content
        assert "bad.write.test" not in content

    @pytest.mark.asyncio
    async def test_generates_draft_from_outline(self, base_state, mock_settings):
        """Test that it generates a draft from the outline."""
        base_state["outline"] = "# Headline\n## Section 1\n## Section 2"
        base_state["topic"] = "technology"

        expected_draft = """# The Future of Technology

Technology continues to evolve at a rapid pace...

## Section 1
Details about section 1...

## Section 2
Details about section 2..."""

        mock_response = AIMessage(content=expected_draft)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 AsyncMock(return_value=mock_response),
             ):
            result = await write_node(base_state)

        assert result["draft"] == expected_draft
        assert result["status"] == "drafted"

    @pytest.mark.asyncio
    async def test_includes_outline_in_llm_request(self, base_state, mock_settings):
        """Test that the outline is included in the LLM request."""
        outline = "# Test Outline\n- Point 1\n- Point 2"
        base_state["outline"] = outline
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch("litenews.workflow.nodes.write.invoke_llm_with_messages", mock_invoke):
            await write_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        assert outline in _last_human_content(messages)

    @pytest.mark.asyncio
    async def test_includes_outline_guided_sources_not_research_notes(
        self, base_state, mock_settings
    ):
        """Search snippets from outline-guided Tavily runs are passed to the writer, not research_notes."""
        research_notes = "Important research findings from analyze phase"
        base_state["outline"] = "Outline section line long enough"
        base_state["research_notes"] = research_notes
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch("litenews.workflow.nodes.write.invoke_llm_with_messages", mock_invoke):
            await write_node(base_state)

        messages = mock_invoke.call_args[0][0]
        human = _last_human_content(messages)
        assert research_notes not in human
        assert "Sources (outline-guided search" in human
        assert "Outline-guided snippet for the draft." in human

    @pytest.mark.asyncio
    async def test_includes_topic_in_llm_request(self, base_state, mock_settings):
        """Test that the topic is included in the LLM request."""
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "artificial intelligence"

        mock_response = AIMessage(content="Draft")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch("litenews.workflow.nodes.write.invoke_llm_with_messages", mock_invoke):
            await write_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        assert "artificial intelligence" in _last_human_content(messages)

    @pytest.mark.asyncio
    async def test_includes_write_instruction_in_request(self, base_state, mock_settings):
        """Test that it includes the write instruction in the request."""
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch("litenews.workflow.nodes.write.invoke_llm_with_messages", mock_invoke):
            await write_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        assert "write the full article" in _last_human_content(messages).lower()

    @pytest.mark.asyncio
    async def test_multi_perspective_prepends_few_shot_turns(self, base_state, mock_settings):
        """多方觀點 requests include style demo Human/AI pairs before the real task."""
        base_state["article_type"] = "多方觀點"
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch("litenews.workflow.nodes.write.invoke_llm_with_messages", mock_invoke):
            await write_node(base_state)

        messages = mock_invoke.call_args[0][0]
        assert len(messages) == 6  # system + 2×(human+AI) few-shot + final human
        assert "【多方觀點】版權修例未列明戲仿定義" in messages[4].content

    @pytest.mark.asyncio
    async def test_includes_messages_in_result(self, base_state, mock_settings):
        """Test that it includes messages in the result."""
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft")

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 AsyncMock(return_value=mock_response),
             ):
            result = await write_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 3

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(self, base_state, mock_settings):
        """Test that it returns an error when LLM fails."""
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "test"

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 AsyncMock(side_effect=Exception("Rate limit exceeded")),
             ):
            result = await write_node(base_state)

        assert result["status"] == "error"
        assert "Writing failed" in result["error"]
        assert "Rate limit exceeded" in result["error"]

    @pytest.mark.asyncio
    async def test_ignores_research_notes_empty_or_not(self, base_state, mock_settings):
        """Drafting uses outline searches; research_notes content is not required."""
        base_state["outline"] = "Outline section line long enough"
        base_state["research_notes"] = ""
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Draft without research")

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 AsyncMock(return_value=mock_response),
             ):
            result = await write_node(base_state)

        assert result["status"] == "drafted"
        assert result["draft"] == "Draft without research"

    @pytest.mark.asyncio
    async def test_returns_error_when_outline_search_returns_no_results(
        self, base_state, mock_settings
    ):
        """Empty merged results after Tavily should surface a clear error."""
        base_state["outline"] = "Outline section line long enough"
        base_state["topic"] = "test"

        empty_tool = AsyncMock()
        empty_tool.ainvoke = AsyncMock(return_value=[])

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=empty_tool,
             ):
            result = await write_node(base_state)

        assert result["status"] == "error"
        assert "no results" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_continues_when_some_outline_queries_fail(
        self, base_state, mock_settings
    ):
        """Write should proceed if at least one outline-guided query succeeds."""
        base_state["outline"] = "# A\nLong enough line one\n## B\nLong enough line two"
        base_state["topic"] = "test"
        mock_response = AIMessage(content="Draft from partial results")

        with patch("litenews.workflow.nodes.write.get_settings", return_value=mock_settings), \
             patch(
                 "litenews.workflow.nodes.write.get_tavily_search_tool",
                 return_value=_stub_tavily_tool_partial_error(),
             ), \
             patch(
                 "litenews.workflow.nodes.write.invoke_llm_with_messages",
                 AsyncMock(return_value=mock_response),
             ):
            result = await write_node(base_state)

        assert result["status"] == "drafted"
        assert result["draft"] == "Draft from partial results"
