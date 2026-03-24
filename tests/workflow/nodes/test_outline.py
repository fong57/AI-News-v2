"""Tests for the outline node."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from litenews.workflow.nodes.outline import outline_node


class TestOutlineNode:
    """Tests for the outline_node function."""

    @pytest.mark.asyncio
    async def test_returns_error_when_no_research_notes(self, base_state):
        """Test that it returns an error when no research notes exist."""
        base_state["research_notes"] = ""

        result = await outline_node(base_state)

        assert result["status"] == "error"
        assert "No research notes for outline" in result["error"]

    @pytest.mark.asyncio
    async def test_creates_outline_from_research_notes(self, base_state, mock_settings):
        """Test that it creates an outline from research notes."""
        base_state["research_notes"] = "Key findings about AI..."
        base_state["topic"] = "artificial intelligence"

        expected_outline = """# AI Developments
        
## Introduction
- Overview of recent AI advancements

## Main Points
- Point 1
- Point 2

## Conclusion
- Future outlook"""

        mock_response = AIMessage(content=expected_outline)

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await outline_node(base_state)

        assert result["outline"] == expected_outline
        assert result["status"] == "outlined"

    @pytest.mark.asyncio
    async def test_includes_topic_in_llm_request(self, base_state, mock_settings):
        """Test that the topic is included in the LLM request."""
        base_state["research_notes"] = "Some research"
        base_state["topic"] = "climate change"

        mock_response = AIMessage(content="Outline")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", mock_invoke):
            await outline_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        user_message = messages[1].content
        assert "climate change" in user_message

    @pytest.mark.asyncio
    async def test_includes_research_notes_in_llm_request(self, base_state, mock_settings):
        """Test that research notes are included in the LLM request."""
        research_notes = "Important findings:\n1. Finding A\n2. Finding B"
        base_state["research_notes"] = research_notes
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Outline")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", mock_invoke):
            await outline_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        user_message = messages[1].content
        assert research_notes in user_message

    @pytest.mark.asyncio
    async def test_includes_messages_in_result(self, base_state, mock_settings):
        """Test that it includes messages in the result."""
        base_state["research_notes"] = "Research"
        base_state["topic"] = "test"

        mock_response = AIMessage(content="Outline")

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", 
                   AsyncMock(return_value=mock_response)):
            result = await outline_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 3

    @pytest.mark.asyncio
    async def test_returns_error_on_llm_exception(self, base_state, mock_settings):
        """Test that it returns an error when LLM fails."""
        base_state["research_notes"] = "Research"
        base_state["topic"] = "test"

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", 
                   AsyncMock(side_effect=Exception("API timeout"))):
            result = await outline_node(base_state)

        assert result["status"] == "error"
        assert "Outline creation failed" in result["error"]
        assert "API timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_uses_outline_system_prompt(self, base_state, mock_settings):
        """Test that the LLM receives the outline system prompt for the article type."""
        base_state["research_notes"] = "Research"
        base_state["topic"] = "test"
        base_state["article_type"] = "其他"

        mock_response = AIMessage(content="Outline")
        mock_invoke = AsyncMock(return_value=mock_response)

        with patch("litenews.workflow.nodes.outline.get_settings", return_value=mock_settings), \
             patch("litenews.workflow.nodes.outline.invoke_llm_with_messages", mock_invoke), \
             patch(
                 "litenews.workflow.nodes.outline.outline_system_prompt",
                 return_value="Test system prompt",
             ):
            await outline_node(base_state)

        call_args = mock_invoke.call_args
        messages = call_args[0][0]
        system_message = messages[0].content
        assert system_message == "Test system prompt"
