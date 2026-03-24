"""Tests for workflow utility functions."""

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from litenews.workflow.utils import (
    create_llm_messages,
    invoke_llm_with_messages,
    create_error_response,
    should_continue,
)


class TestCreateLLMMessages:
    """Tests for create_llm_messages function."""

    def test_creates_system_and_human_messages(self):
        """Test that it creates both system and human messages."""
        system_prompt = "You are a helpful assistant."
        user_content = "Hello, how are you?"

        messages = create_llm_messages(system_prompt, user_content)

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_system_message_content(self):
        """Test that system message has correct content."""
        system_prompt = "Test system prompt"
        user_content = "Test user content"

        messages = create_llm_messages(system_prompt, user_content)

        assert messages[0].content == system_prompt

    def test_human_message_content(self):
        """Test that human message has correct content."""
        system_prompt = "Test system prompt"
        user_content = "Test user content"

        messages = create_llm_messages(system_prompt, user_content)

        assert messages[1].content == user_content

    def test_empty_prompts(self):
        """Test with empty prompts."""
        messages = create_llm_messages("", "")

        assert len(messages) == 2
        assert messages[0].content == ""
        assert messages[1].content == ""


class TestInvokeLLMWithMessages:
    """Tests for invoke_llm_with_messages function."""

    @pytest.mark.asyncio
    async def test_invokes_llm_with_messages(self, mock_settings):
        """Test that it invokes the LLM with provided messages."""
        mock_response = AIMessage(content="Test response")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
        ]

        with patch("litenews.workflow.utils.get_llm", return_value=mock_llm):
            response = await invoke_llm_with_messages(messages, mock_settings)

        assert response == mock_response
        mock_llm.ainvoke.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_uses_default_settings_when_not_provided(self):
        """Test that it fetches settings when not provided."""
        mock_response = AIMessage(content="Test response")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        messages = [HumanMessage(content="Test")]

        with patch("litenews.workflow.utils.get_settings") as mock_get_settings, \
             patch("litenews.workflow.utils.get_llm", return_value=mock_llm):
            await invoke_llm_with_messages(messages, None)

            mock_get_settings.assert_called_once()


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_returns_error_dict(self):
        """Test that it returns a dict with error and status."""
        result = create_error_response("Test error")

        assert isinstance(result, dict)
        assert "error" in result
        assert "status" in result

    def test_error_message_is_included(self):
        """Test that the error message is included."""
        error_msg = "Something went wrong"
        result = create_error_response(error_msg)

        assert result["error"] == error_msg

    def test_status_is_error(self):
        """Test that status is set to 'error'."""
        result = create_error_response("Any error")

        assert result["status"] == "error"

    def test_empty_error_message(self):
        """Test with empty error message."""
        result = create_error_response("")

        assert result["error"] == ""
        assert result["status"] == "error"


class TestShouldContinue:
    """Tests for should_continue function."""

    def test_returns_continue_when_no_error(self, base_state):
        """Test that it returns 'continue' when there's no error."""
        base_state["error"] = ""

        result = should_continue(base_state)

        assert result == "continue"

    def test_returns_error_when_error_exists(self, base_state):
        """Test that it returns 'error' when there's an error."""
        base_state["error"] = "Some error occurred"

        result = should_continue(base_state)

        assert result == "error"

    def test_returns_continue_when_error_key_missing(self):
        """Test that it returns 'continue' when error key is missing."""
        state = {"topic": "test"}

        result = should_continue(state)

        assert result == "continue"

    def test_returns_continue_when_error_is_none(self, base_state):
        """Test that it returns 'continue' when error is None."""
        base_state["error"] = None

        result = should_continue(base_state)

        assert result == "continue"

    def test_returns_error_when_error_is_truthy(self, base_state):
        """Test that any truthy error value returns 'error'."""
        base_state["error"] = "Error!"

        result = should_continue(base_state)

        assert result == "error"
