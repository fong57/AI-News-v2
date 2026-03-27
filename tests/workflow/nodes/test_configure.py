"""Tests for the configure_workflow node."""

from unittest.mock import patch

import pytest

from litenews.config.settings import Settings
from litenews.workflow.nodes.configure import configure_workflow_node


@pytest.mark.asyncio
async def test_configure_applies_defaults():
    settings = Settings(
        pplx_api_key="k",
        dashscope_api_key="k",
        bailian_api_key="k",
        tavily_api_key="k",
        primary_llm="qwen",
    )
    state = {"topic": "t", "article_type": "其他"}
    with patch("litenews.workflow.nodes.configure.get_settings", return_value=settings):
        out = await configure_workflow_node(state)
    assert out["target_word_count"] == 800
    assert out["llm_provider"] == "qwen"
    assert out["llm_model"] == ""
    assert out["status"] == "configured"
    assert out.get("task") == "write"
    assert "error" not in out


@pytest.mark.asyncio
async def test_configure_invalid_task():
    settings = Settings(
        pplx_api_key="k",
        dashscope_api_key="k",
        bailian_api_key="k",
        tavily_api_key="k",
    )
    state = {"article_type": "其他", "task": "bogus"}
    with patch("litenews.workflow.nodes.configure.get_settings", return_value=settings):
        out = await configure_workflow_node(state)
    assert out["status"] == "error"
    assert "task" in out["error"].lower()


@pytest.mark.asyncio
async def test_configure_invalid_provider():
    settings = Settings(
        pplx_api_key="k",
        dashscope_api_key="k",
        bailian_api_key="k",
        tavily_api_key="k",
    )
    state = {"article_type": "其他", "llm_provider": "openai"}
    with patch("litenews.workflow.nodes.configure.get_settings", return_value=settings):
        out = await configure_workflow_node(state)
    assert out["status"] == "error"
    assert "perplexity" in out["error"]


@pytest.mark.asyncio
async def test_configure_missing_perplexity_key():
    settings = Settings(
        pplx_api_key="",
        dashscope_api_key="k",
        bailian_api_key="k",
        tavily_api_key="k",
        primary_llm="perplexity",
    )
    state = {"article_type": "其他", "llm_provider": "perplexity"}
    with patch("litenews.workflow.nodes.configure.get_settings", return_value=settings):
        out = await configure_workflow_node(state)
    assert out["status"] == "error"
    assert "Perplexity" in out["error"]


@pytest.mark.asyncio
async def test_configure_missing_bailian_key():
    settings = Settings(
        pplx_api_key="k",
        dashscope_api_key="k",
        bailian_api_key="",
        tavily_api_key="k",
        primary_llm="bailian",
    )
    state = {"article_type": "其他", "llm_provider": "bailian"}
    with patch("litenews.workflow.nodes.configure.get_settings", return_value=settings):
        out = await configure_workflow_node(state)
    assert out["status"] == "error"
    assert "Bailian" in out["error"]
