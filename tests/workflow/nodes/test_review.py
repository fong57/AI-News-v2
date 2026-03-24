"""Tests for the review node."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from litenews.state.news_state import NewsArticle, NewsSource
from litenews.workflow.nodes.review import review_node, parse_article_response


class TestParseArticleResponse:
    """Tests for the parse_article_response function."""

    def test_extracts_headline_from_first_line(self):
        """Test that it extracts the headline from the first line."""
        content = "# Breaking News\n\nThis is the article body."
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert article.headline == "Breaking News"

    def test_removes_markdown_heading_symbols(self):
        """Test that it removes markdown heading symbols from headline."""
        content = "### Some Headline\n\nBody text"
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert article.headline == "Some Headline"
        assert "#" not in article.headline

    def test_uses_topic_as_fallback_headline(self):
        """Test that it uses topic as fallback when content is empty."""
        content = ""
        sources = []

        article = parse_article_response(content, "Fallback Topic", sources)

        assert article.headline == "Fallback Topic"

    def test_extracts_body_from_remaining_lines(self):
        """Test that it extracts body from lines after headline."""
        content = "Headline\n\nFirst paragraph.\n\nSecond paragraph."
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert "First paragraph." in article.body
        assert "Second paragraph." in article.body
        assert "Headline" not in article.body

    def test_extracts_summary_until_double_newline(self):
        """Test that it extracts summary until double newline."""
        content = "Headline\n\nThis is the summary paragraph.\n\nThis is more content."
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert article.summary == "This is the summary paragraph."

    def test_uses_first_200_chars_when_no_double_newline(self):
        """Test that it uses first 200 chars when no paragraph break."""
        long_text = "A" * 300
        content = f"Headline\n{long_text}"
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert len(article.summary) == 200

    def test_includes_sources_in_article(self, mock_news_sources):
        """Test that sources are included in the article."""
        content = "Headline\n\nBody"

        article = parse_article_response(content, "topic", mock_news_sources)

        assert article.sources == mock_news_sources
        assert len(article.sources) == 2

    def test_includes_topic_in_keywords(self):
        """Test that topic is included in keywords."""
        content = "Headline\n\nBody"
        sources = []

        article = parse_article_response(content, "technology", sources)

        assert "technology" in article.keywords

    def test_returns_news_article_instance(self):
        """Test that it returns a NewsArticle instance."""
        content = "Headline\n\nBody"
        sources = []

        article = parse_article_response(content, "topic", sources)

        assert isinstance(article, NewsArticle)

    def test_handles_single_line_content(self):
        """Test that it handles single line content."""
        content = "Just a headline"
        sources = []

        article = parse_article_response(content, "fallback", sources)

        assert article.headline == "Just a headline"
        assert article.body == "Just a headline"


class TestReviewNode:
    """Tests for the review_node function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_draft(self, base_state):
        """Test that it returns an error when no draft exists."""
        base_state["draft"] = ""

        result = await review_node(base_state)

        assert result["status"] == "error"
        assert "No draft to review" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_final_article(self, base_state, mock_settings, mock_news_sources):
        """Test that it returns a final article."""
        base_state["draft"] = "Draft article content"
        base_state["sources"] = mock_news_sources
        base_state["topic"] = "AI news"

        mock_response = AIMessage(content="# Final Headline\n\nPolished article body.")

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await review_node(base_state)

        assert result["status"] == "completed"
        assert isinstance(result["final_article"], NewsArticle)
        assert result["final_article"].headline == "Final Headline"

    @pytest.mark.asyncio
    async def test_includes_sources_in_final_article(
        self, base_state, mock_settings, mock_news_sources
    ):
        """Test that sources are included in the final article."""
        base_state["draft"] = "Draft"
        base_state["sources"] = mock_news_sources
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Headline\n\nBody")

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await review_node(base_state)

        assert result["final_article"].sources == mock_news_sources

    @pytest.mark.asyncio
    async def test_includes_topic_in_keywords(self, base_state, mock_settings):
        """Test that topic is included in keywords."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "cryptocurrency"

        mock_response = AIMessage(content="Headline\n\nBody")

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await review_node(base_state)

        assert "cryptocurrency" in result["final_article"].keywords

    @pytest.mark.asyncio
    async def test_includes_draft_in_llm_request(self, base_state, mock_settings):
        """Test that the draft is included in the LLM request."""
        draft = "This is the draft article to review."
        base_state["draft"] = draft
        base_state["sources"] = []
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Reviewed")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", mock_invoke):
            await review_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        user_message = messages[1].content
        assert draft in user_message

    @pytest.mark.asyncio
    async def test_includes_topic_in_llm_request(self, base_state, mock_settings):
        """Test that the topic is included in the LLM request."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "space exploration"

        mock_response = AIMessage(content="Reviewed")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", mock_invoke):
            await review_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        user_message = messages[1].content
        assert "space exploration" in user_message

    @pytest.mark.asyncio
    async def test_includes_review_instruction_in_request(self, base_state, mock_settings):
        """Test that it includes review instruction in the request."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Reviewed")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", mock_invoke):
            await review_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        user_message = messages[1].content
        assert "review" in user_message.lower()

    @pytest.mark.asyncio
    async def test_includes_messages_in_result(self, base_state, mock_settings):
        """Test that it includes messages in the result."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Reviewed")

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await review_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 3

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(self, base_state, mock_settings):
        """Test that it returns an error when LLM fails."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "test"

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(side_effect=Exception("Service unavailable"))):
            result = await review_node(base_state)

        assert result["status"] == "error"
        assert "Review failed" in result["error"]
        assert "Service unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_handles_empty_sources(self, base_state, mock_settings):
        """Test that it handles empty sources list."""
        base_state["draft"] = "Draft"
        base_state["sources"] = []
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Headline\n\nBody")

        with patch("litenews.workflow.nodes.review.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.review.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await review_node(base_state)

        assert result["status"] == "completed"
        assert result["final_article"].sources == []
