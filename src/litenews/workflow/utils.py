"""Shared utility functions for the news writing workflow.

This module contains helper functions used across multiple workflow nodes.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from litenews.config.settings import get_settings, Settings
from litenews.llms.base import get_llm
from litenews.state.news_state import NewsState


def workflow_llm_options(state: NewsState, settings: Settings) -> dict[str, str | None]:
    """Provider and optional model from state (after configure); fall back to settings."""
    raw_p = state.get("llm_provider")
    if isinstance(raw_p, str) and raw_p.strip():
        provider: str = raw_p.strip().lower()
    else:
        provider = settings.primary_llm
    raw_m = state.get("llm_model")
    model = (str(raw_m).strip() if raw_m is not None else "") or None
    return {"llm_provider": provider, "llm_model": model}


def create_llm_messages(
    system_prompt: str,
    user_content: str,
) -> list[BaseMessage]:
    """Create a standard message list for LLM invocation.
    
    Args:
        system_prompt: The system prompt to use.
        user_content: The user message content.
        
    Returns:
        list[BaseMessage]: List containing system and human messages.
    """
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]


async def invoke_llm_with_messages(
    messages: list[BaseMessage],
    settings: Settings | None = None,
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> BaseMessage:
    """Invoke the LLM with the given messages.
    
    Args:
        messages: The messages to send to the LLM.
        settings: Optional settings object. If not provided, will fetch from get_settings().
        llm_provider: Optional perplexity/qwen override (defaults to settings.primary_llm).
        llm_model: Optional model id for that provider (defaults to env settings for provider).
        
    Returns:
        BaseMessage: The LLM response message.
    """
    if settings is None:
        settings = get_settings()
    provider = llm_provider if llm_provider else settings.primary_llm
    model_override = llm_model.strip() if llm_model else None
    llm = get_llm(provider=provider, settings=settings, model_override=model_override)
    return await llm.ainvoke(messages)


def create_error_response(error_message: str) -> dict:
    """Create a standardized error response.
    
    Args:
        error_message: The error message to include.
        
    Returns:
        dict: Error response with error message and error status.
    """
    return {"error": error_message, "status": "error"}


def should_continue(state: NewsState) -> Literal["continue", "error"]:
    """Determine if the workflow should continue or stop due to error.
    
    Args:
        state: Current workflow state.
        
    Returns:
        str: "continue" to proceed, "error" to stop.
    """
    if state.get("error"):
        return "error"
    return "continue"
