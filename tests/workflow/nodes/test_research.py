"""Tests for the research node."""

from unittest.mock import AsyncMock, patch

import pytest

from litenews.config.settings import Settings
from litenews.workflow.nodes.research import research_node


class TestResearchNode:
    """Tests for the research_node function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_article_type_missing(self, base_state):
        """article_type is required before search."""
        del base_state["article_type"]

        result = await research_node(base_state)

        assert result["status"] == "error"
        assert "article_type" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_when_article_type_invalid(self, base_state):
        """article_type must be one of the allowed labels."""
        base_state["article_type"] = "專訪"

        result = await research_node(base_state)

        assert result["status"] == "error"
        assert "article_type" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_when_no_topic(self, base_state):
        """Test that it returns an error when no topic is provided."""
        base_state["topic"] = ""

        result = await research_node(base_state)

        assert result["status"] == "error"
        assert "No topic provided" in result["error"]

    @pytest.mark.asyncio
    async def test_uses_default_query_when_not_provided(self, base_state, mock_settings, mock_search_results):
        """Test that it creates a default query from the topic."""
        base_state["topic"] = "climate change"
        base_state["query"] = ""

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={"results": mock_search_results})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["query"] == "搜索關於「climate change」的最新消息"
        assert result["status"] == "researched"

    @pytest.mark.asyncio
    async def test_uses_provided_query(self, base_state, mock_settings, mock_search_results):
        """Test that it uses the provided query when available."""
        base_state["topic"] = "climate change"
        base_state["query"] = "Custom query about climate"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={"results": mock_search_results})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["query"] == "Custom query about climate"
        assert result["status"] == "researched"

    @pytest.mark.asyncio
    async def test_filters_blocked_domains_from_search_results(
        self, base_state, mock_settings
    ):
        """Blocked hosts must not appear in search_results or pool."""
        base_state["topic"] = "AI"
        mixed = [
            {
                "title": "Keep",
                "url": "https://keep.test/article",
                "content": "ok",
            },
            {
                "title": "Drop",
                "url": "https://blocked.test/x",
                "content": "no",
            },
        ]
        settings = mock_settings.model_copy(
            update={"tavily_exclude_domains": ["blocked.test"]}
        )
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={"results": mixed})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=settings), \
             patch(
                 "litenews.workflow.nodes.research.get_tavily_search_tool",
                 return_value=mock_search_tool,
             ):
            result = await research_node(base_state)

        assert result["status"] == "researched"
        assert len(result["search_results"]) == 1
        assert result["search_results"][0]["url"] == "https://keep.test/article"
        pool = result.get("tavily_evidence_pool") or []
        assert all("blocked.test" not in (r.get("url") or "") for r in pool)

    @pytest.mark.asyncio
    async def test_returns_search_results_as_list(self, base_state, mock_settings, mock_search_results):
        """Test that it returns search results when results is a list."""
        base_state["topic"] = "AI"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value=mock_search_results)

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "researched"
        assert result["search_results"] == mock_search_results
        assert len(result["search_results"]) == 2

    @pytest.mark.asyncio
    async def test_returns_search_results_from_dict(self, base_state, mock_settings, mock_search_results):
        """Test that it extracts results from dict response."""
        base_state["topic"] = "AI"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={"results": mock_search_results})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "researched"
        assert result["search_results"] == mock_search_results

    @pytest.mark.asyncio
    async def test_returns_error_when_empty_dict_results(self, base_state, mock_settings):
        """Empty Tavily payload should surface as an error, not success."""
        base_state["topic"] = "AI"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "no search results" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_empty_list_response(self, base_state, mock_settings):
        base_state["topic"] = "AI"
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value=[])

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "no search results" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_returns_error_when_tavily_key_missing(self, base_state, mock_settings):
        bad_settings = mock_settings.model_copy(update={"tavily_api_key": ""})
        base_state["topic"] = "AI"

        with patch("litenews.workflow.nodes.research.get_settings", return_value=bad_settings):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "TAVILY_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_when_tavily_placeholder_key(self, base_state):
        settings = Settings(
            pplx_api_key="k",
            dashscope_api_key="k",
            tavily_api_key="your_tavily_api_key_here",
        )
        base_state["topic"] = "AI"

        with patch("litenews.workflow.nodes.research.get_settings", return_value=settings):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "TAVILY_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_tavily_error_field(self, base_state, mock_settings):
        base_state["topic"] = "AI"
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(
            return_value={"error": "Invalid API key"},
        )

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "Invalid API key" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_tavily_detail_without_results(self, base_state, mock_settings):
        base_state["topic"] = "AI"
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(
            return_value={"detail": "Unauthorized"},
        )

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "Unauthorized" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_search_exception(self, base_state, mock_settings):
        """Test that it returns an error when search fails."""
        base_state["topic"] = "AI"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(side_effect=Exception("Search API error"))

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "Research failed" in result["error"]
        assert "Search API error" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_error_on_string_response(self, base_state, mock_settings):
        """String payloads from Tavily are treated as API error text."""
        base_state["topic"] = "AI"
        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value="Rate limit exceeded")

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            result = await research_node(base_state)

        assert result["status"] == "error"
        assert "Rate limit exceeded" in result["error"]

    @pytest.mark.asyncio
    async def test_invokes_search_tool_with_query(self, base_state, mock_settings, mock_search_results):
        """Test that it invokes the search tool with the correct query."""
        base_state["topic"] = "technology"
        base_state["query"] = "latest tech news"

        mock_search_tool = AsyncMock()
        mock_search_tool.ainvoke = AsyncMock(return_value={"results": mock_search_results})

        with patch("litenews.workflow.nodes.research.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.research.get_tavily_search_tool", return_value=mock_search_tool):
            await research_node(base_state)

        mock_search_tool.ainvoke.assert_called_once_with("latest tech news")
