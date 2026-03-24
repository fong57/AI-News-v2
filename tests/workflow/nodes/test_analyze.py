"""Tests for the analyze node."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from litenews.state.news_state import NewsSource
from litenews.workflow.nodes.analyze import analyze_node


class TestAnalyzeNode:
    """Tests for the analyze_node function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_search_results(self, base_state):
        """Test that it returns an error when no search results exist."""
        base_state["search_results"] = []

        result = await analyze_node(base_state)

        assert result["status"] == "error"
        assert "No search results to analyze" in result["error"]

    @pytest.mark.asyncio
    async def test_converts_search_results_to_news_sources(
        self, base_state, mock_settings, mock_search_results
    ):
        """Test that search results are converted to NewsSource objects."""
        base_state["search_results"] = mock_search_results
        base_state["topic"] = "AI"

        mock_response = AIMessage(content="Analysis of the articles...")

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert len(result["sources"]) == 2
        assert all(isinstance(s, NewsSource) for s in result["sources"])
        assert result["sources"][0].title == "Test Article 1"
        assert result["sources"][1].url == "https://example.com/article2"

    @pytest.mark.asyncio
    async def test_extracts_content_from_snippet_fallback(self, base_state, mock_settings):
        """Test that it uses snippet as fallback when content is not available."""
        base_state["search_results"] = [
            {
                "title": "Article",
                "url": "https://example.com",
                "snippet": "This is the snippet",
            }
        ]
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Analysis")

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert result["sources"][0].snippet == "This is the snippet"

    @pytest.mark.asyncio
    async def test_returns_research_notes_from_llm(
        self, base_state, mock_settings, mock_search_results
    ):
        """Test that it returns research notes from the LLM response."""
        base_state["search_results"] = mock_search_results
        base_state["topic"] = "technology"

        expected_notes = "Key findings:\n1. Finding one\n2. Finding two"
        mock_response = AIMessage(content=expected_notes)

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert result["research_notes"] == expected_notes
        assert result["status"] == "analyzed"

    @pytest.mark.asyncio
    async def test_includes_messages_in_result(
        self, base_state, mock_settings, mock_search_results
    ):
        """Test that it includes messages in the result."""
        base_state["search_results"] = mock_search_results
        base_state["topic"] = "AI"

        mock_response = AIMessage(content="Analysis")

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 3  # system + human + response

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(
        self, base_state, mock_settings, mock_search_results
    ):
        """Test that it returns an error when LLM fails."""
        base_state["search_results"] = mock_search_results
        base_state["topic"] = "AI"

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(side_effect=Exception("LLM API error"))):
            result = await analyze_node(base_state)

        assert result["status"] == "error"
        assert "Analysis failed" in result["error"]
        assert "LLM API error" in result["error"]

    @pytest.mark.asyncio
    async def test_skips_non_dict_results(self, base_state, mock_settings):
        """Test that it skips non-dict items in search results."""
        base_state["search_results"] = [
            {"title": "Valid", "url": "https://example.com", "content": "Content"},
            "invalid_item",
            None,
            {"title": "Also Valid", "url": "https://example2.com", "content": "Content 2"},
        ]
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Analysis")

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert len(result["sources"]) == 2

    @pytest.mark.asyncio
    async def test_handles_missing_published_date(self, base_state, mock_settings):
        """Test that it handles missing published_date gracefully."""
        base_state["search_results"] = [
            {"title": "Article", "url": "https://example.com", "content": "Content"},
        ]
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Analysis")

        with patch("litenews.workflow.nodes.analyze.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.analyze.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await analyze_node(base_state)

        assert result["sources"][0].published_date is None
