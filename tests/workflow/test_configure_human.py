"""Tests for configure_human interrupt resume handling."""

import pytest

from litenews.workflow.nodes.configure_human import (
    _configure_human_resume,
    _validate_merged_configure,
    route_after_configure_human,
)
from litenews.state.news_state import create_initial_state


class _DummySettings:
    min_target_word_count = 100
    max_target_word_count = 10000

    def has_perplexity_key(self) -> bool:
        return True

    def has_qwen_key(self) -> bool:
        return True


@pytest.fixture
def dummy_settings(monkeypatch):
    monkeypatch.setattr(
        "litenews.workflow.nodes.configure_human.get_settings",
        lambda: _DummySettings(),
    )


def _base_state():
    return create_initial_state("Test topic", "懶人包", target_word_count=800, llm_provider="perplexity")


def test_validate_success(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    s["query"] = ""
    out = _validate_merged_configure(s)
    assert out.get("error") is None
    assert out["status"] == "configure_confirmed"
    assert out["topic"] == "Test topic"
    assert out.get("task") == "write"


def test_validate_missing_topic(dummy_settings):
    s = _base_state()
    s["topic"] = "  "
    out = _validate_merged_configure(s)
    assert out.get("error")


def test_resume_accept_string(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    out = _configure_human_resume("accept", s)
    assert out.get("error") is None
    assert out["status"] == "configure_confirmed"


def test_resume_confirm_overrides_topic(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    out = _configure_human_resume(
        {"action": "confirm", "topic": "  New topic  "},
        s,
    )
    assert out.get("error") is None
    assert out["topic"] == "New topic"


def test_resume_invalid_article_type(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    out = _configure_human_resume(
        {"action": "confirm", "article_type": "invalid"},
        s,
    )
    assert out.get("error")


def test_resume_accept_with_task_edit(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    s["task"] = "write"
    out = _configure_human_resume({"action": "accept", "task": "edit"}, s)
    assert out.get("error") is None
    assert out["task"] == "edit"


def test_validate_invalid_task(dummy_settings):
    s = _base_state()
    s["llm_model"] = ""
    s["task"] = "bogus"
    out = _validate_merged_configure(s)
    assert out.get("error")


def test_route_after_configure_human(dummy_settings):
    s = _base_state()
    assert route_after_configure_human(s) == "research"
    s["task"] = "edit"
    assert route_after_configure_human(s) == "edit_human"
    s["error"] = "x"
    assert route_after_configure_human(s) == "error"
